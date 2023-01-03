
from src.models.loss.WeightLossBinaryCrossentropy import WeightLossBinaryCrossentropy
from src.models.pipeline.transform_customer_mapping import transform_customer_mapping
from src.models.pipeline.transform_article_mapping import transform_article_mapping
from src.models.eval_metric.evaluate_metric import transform_logist_label 
from tensorflow.keras.models import Model, load_model, model_from_json
from utils.images_utils import get_image_path,  decode_train_image
from tensorflow.keras.applications.resnet50 import ResNet50 
from tensorflow.keras.layers import  Concatenate, Multiply
from utils.read_utils import  read_yaml_key, read_object
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import  BatchNormalization
from tensorflow.keras.layers import Embedding, Dense
#from tensorflow.keras.utils import plot_model
from config.config import CONFIGURATION_PATH
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Flatten 
from tensorflow.keras.metrics import Mean
from sklearn.pipeline import Pipeline
import utils.write_utils as hlpwrite
import utils.read_utils as hlpread
from tensorflow.keras import Input
from datetime import timedelta
from scipy.stats import mode
import logs.logger as log 
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import re
import gc

class resnet_based_prevence: 

    def __init__(self, is_training , config_path = CONFIGURATION_PATH):

        self.is_training = is_training 
        self.config_path = config_path       
        self.model_threshold = {}
        self.all_models = []
        

    def generate_data_for_nth_ensemble_model(self, train_tran, ensemble_model_number, pos_neg_ratio):
    
        #Split -ve and +ve sample from dataset
        train_tran_pos = train_tran[train_tran.label == 1]
        #train_tran_pos.user_id.nunique()
        train_tran_neg = train_tran[train_tran.label == 0]

        #Count number of +ve sample we have for each user based on that we will get -ve sample for each user for a given ensemble_model_number
        train_tran_neg = (train_tran_neg.merge((train_tran_pos[['user_id','label']]
                                                .groupby('user_id')['label']
                                                .count()
                                                .reset_index(name = 'cnt')
                                            ), 
                                            on = 'user_id', 
                                            how = 'inner')
                        )
        train_tran_neg['total_neg_sample_per_ensemble'] = train_tran_neg['cnt'] * pos_neg_ratio

        #train_tran_neg.groupby('user_id').label.count()

        total_neg_sample_per_ensemble = len(train_tran_pos) * pos_neg_ratio

        #Generate -ve sample based on the total +ve sample we have per user
        df_train_tran = pd.DataFrame()

        group_neg_user = train_tran_neg.groupby('user_id')
        ensemble_number = ensemble_model_number

        for i, x in enumerate(group_neg_user.groups):

            grp_key = group_neg_user.get_group(x)    

            total_neg_sample_per_user = grp_key.iloc[0,grp_key.columns.get_loc("total_neg_sample_per_ensemble")]

            data_start_index = ensemble_number * total_neg_sample_per_user
            data_end_index = data_start_index + total_neg_sample_per_user

            if i == 0:        
                df_train_tran = grp_key[['user_id','item_id','label','image_path']][data_start_index: data_end_index] #grp_key.nth(list(range(0, 10)))
            
            else:
                df_train_tran = pd.concat([df_train_tran, 
                                        (grp_key[['user_id','item_id','label','image_path']][data_start_index: data_end_index])],
                                        axis = 0)
        
        df_train_tran = pd.concat([train_tran_pos[['user_id','item_id','label','image_path']], df_train_tran], axis = 0)
        #df_train_tran = df_train_tran.sort_values(by = 'user_id')
        #Shuffle that will help for data pass for training
        df_train_tran = df_train_tran.sample(frac = 1).reset_index(drop = True)
        
        del [train_tran_pos, train_tran_neg]
        gc.collect()
        
        return df_train_tran

    def model_def(self, parms, unique_user, unique_item):

        #HyperParamaters
        SEED = parms["SEED"]
        L2_reg = parms["L2_reg"]
        CHANNEL = parms["CHANNEL"]
        IMAGE_SIZES = parms["IMAGE_SIZES"]
        EMBEDDING_U = parms["EMBEDDING_U"]
        EMBEDDING_I = parms["EMBEDDING_I"]
        NUM_UNIQUE_ITEMS = unique_item
        NUM_UNIQUE_USERS = unique_user
        EMBEDDING_IMG = parms["EMBEDDING_IMG"]
        LEARNING_RATE = parms["LEARNING_RATE"]
        GLOBAL_BATCH_SIZE = parms["GLOBAL_BATCH_SIZE"]
        INTER_EMBEDDING_I =  parms["INTER_EMBEDDING_I"]
        
        TOTAL_TRAINABLE_LAYERS : 176
        NUMBER_NON_TRAINABLE_LAYERS = TOTAL_TRAINABLE_LAYERS - parms["FINE_TUNE_LAYERS"]

        num_replicas_in_sync : 8 #TPU
        LEARNING_RATE = LEARNING_RATE * num_replicas_in_sync
                                    
        #***************************** Optimizer, Loss Function and Metric *************************

        init_lr = LEARNING_RATE 
        #print(f"Learning rate(lr): {init_lr}")
        params = {}
        params['alpha'] = 0.8 
        params['num_replicas_in_sync'] = num_replicas_in_sync #TPU
        params['global_batch_size'] = GLOBAL_BATCH_SIZE
        params['from_logits'] = True

        fn_loss = WeightLossBinaryCrossentropy(param = params)

        fn_optimizer = Adam(learning_rate = init_lr) 

        #***************************** Define Model ************************* 
        weight_initializers = RandomUniform(minval = NUM_UNIQUE_USERS-1, maxval = 1, seed = SEED)


        #***************************** User Embedding ************************* 
        User_Input = Input(shape = (1,), name = 'User_Input')  

        
        User_Embed = Embedding(input_dim = NUM_UNIQUE_USERS, 
                                input_length = 1,
                                output_dim = EMBEDDING_U,
                                embeddings_initializer = weight_initializers,
                                name = 'User_Embed'
                                )(User_Input)
        
        User_Embed_Batch_Normalize = BatchNormalization(name = 'User_Embed_Batch_Normalize')(User_Embed) 
        user_embedding = Flatten(name = "user_embedding")(User_Embed_Batch_Normalize) 
        

        #***************************** Image Embedding ************************* 
        Image_Input = Input(shape = ((IMAGE_SIZES, IMAGE_SIZES, CHANNEL)), name = 'Image_Input')
        model_RESENT50 = ResNet50(weights = 'imagenet', include_top = False, 
                                    pooling = 'avg',
                                    input_shape = (IMAGE_SIZES, IMAGE_SIZES, CHANNEL)
                                    )
        model_RESENT50.trainable = True  
        number_of_layers = len(model_RESENT50.layers)
        print('Number of layers: ', number_of_layers)

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in range(0, NUMBER_NON_TRAINABLE_LAYERS):
            model_RESENT50.layers[layer].trainable =  False

        non_trainable_layers_cnt = 0
        trainable_layers_cnt = 0

        for layer in range(0,len(model_RESENT50.layers)):

            if model_RESENT50.layers[layer].trainable == True:
                trainable_layers_cnt += 1

            elif model_RESENT50.layers[layer].trainable == False:
                non_trainable_layers_cnt += 1

        print('Number of non trainable layers in ResNet.....', non_trainable_layers_cnt) 
        print('Number of trainable layers in ResNet.....', trainable_layers_cnt)


        Image_RESNET_Output = model_RESENT50(Image_Input)
        
        Image_Embed_Dense = Dense(units = EMBEDDING_IMG,
                                    activation = 'relu',
                                    kernel_regularizer = l2(L2_reg), 
                                    kernel_initializer = weight_initializers,
                                    name = 'Image_Embed_Dense')(Image_RESNET_Output)
        Image_Embed  = BatchNormalization(name = 'image_embedding')(Image_Embed_Dense) 


        #***************************** Item embedding *************************

        Item_Input = Input(shape = (1,), name = 'Item_Input')
        
        Item_Embed = Embedding(input_dim = NUM_UNIQUE_ITEMS,
                                input_length = 1,
                                output_dim = INTER_EMBEDDING_I, 
                                embeddings_initializer = weight_initializers,
                                name = "Item_Embed"
                                )(Item_Input)
        
        Item_Embed_Batch_Normalize = BatchNormalization(name = 'Item_Embed_Batch_Normalize')(Item_Embed)
        Item_Embed_ReShape = Flatten(name = 'Item_Embed_Reshape')(Item_Embed_Batch_Normalize)
        
        Item_Image_Embedding = Concatenate(axis = 1, name = 'Item_Image_Concate')([Item_Embed_ReShape, Image_Embed])
        Item_Image_Embed_Dense = Dense(units = EMBEDDING_I, 
                                        activation = 'relu',
                                        kernel_regularizer = l2(L2_reg), 
                                        kernel_initializer = weight_initializers,
                                        name = 'Item_Image_Embed_Dense')(Item_Image_Embedding) 
        item_embedding = BatchNormalization(name = 'item_embedding')(Item_Image_Embed_Dense)


        #***************************** Model *************************
        dot_user_item = Multiply(name = 'mul_user_item')([user_embedding, item_embedding])
        logits = tf.math.reduce_sum(dot_user_item, 1, name = 'reduce_sum_logits')
        y_hat = logits

        Img_Rec = Model(inputs = [User_Input, Item_Input, Image_Input], outputs = [y_hat], name = 'Image_Recommendation')

        Img_Rec.compile(optimizer = fn_optimizer, 
                       loss = fn_loss
                    )

        return Img_Rec

    def feature_eng(self, X):

        log.write_log('Transform mapping customer/article to user/item started...', log.logging.DEBUG)
        engg = Pipeline( steps = [
                                        ('transform_article_mapping', transform_article_mapping(config_path = self.config_path)),

                                        ('transform_customer_mapping', transform_customer_mapping(hash_conversion = True, config_path = self.config_path)),
        ])

        X = engg.fit_transform(X)  
        log.write_log('Transform mapping customer/article to user/item completed...', log.logging.DEBUG)

        log.write_log(f'Map article id to image path started for {str(X.shape[0])}...', log.logging.DEBUG)
        X['image_path'] = list(map(get_image_path, X['article_id']))
        log.write_log('Map article id to image path completed...', log.logging.DEBUG)

        X = X[X.image_path != ""]

        return X

    def train(self, X):

        #Pipeline to transform customer/article and generate image path
        X = self.feature_eng(X)

        #Load model model paramaters
        parms = read_yaml_key(self.config_path,'image-based-ensemble-models','param')

        ############################   Define model      ############################
        unique_user = X['customer_id'].nunique()
        unique_item = X['article_id'].nunique()
        Img_Rec = self.model_def(parms, unique_user, unique_item)


        GLOBAL_BATCH_SIZE = parms["GLOBAL_BATCH_SIZE"]
        #current_lr = parms['LEARNING_RATE']        
        number_ensemble = read_yaml_key(self.config_path,'image-based-ensemble-models','number_ensemble_models')
        epochs = read_yaml_key(self.config_path,'image-based-ensemble-models','epochs')
        training_model_loss = read_yaml_key(self.config_path,'image-based-ensemble-models','training_model_loss')
        end_of_training_loss  = read_yaml_key(self.config_path,'image-based-ensemble-models','end_of_training_loss')
        saved_training_model = read_yaml_key(self.config_path,'image-based-ensemble-models','saved_training_model')
        pos_neg_ratio = 10
        
        epoch_training_loss = []
        epoch_loss_metric = Mean()         
        for ensemble in range(0, number_ensemble):

            #print(f'Ensemble batch {ensemble}')

            df_train_tran = self.generate_data_for_nth_ensemble_model(X, ensemble, pos_neg_ratio)
        
            train_batch = (tf.data.Dataset
                        .from_tensor_slices((df_train_tran['user_id'],
                                                df_train_tran['item_id'],
                                                df_train_tran['image_path'],
                                                df_train_tran['label']
                                            ))
                        .map(decode_train_image, num_parallel_calls = tf.data.experimental.AUTOTUNE) 
                        .prefetch(GLOBAL_BATCH_SIZE) 
                        .batch(GLOBAL_BATCH_SIZE)                             
                        )

            for epoch in range(0, epochs):

                step_training_loss = []     
                epoch_loss_metric.reset_states()
                batch_cnt = 0        

                
                for  Users, Items, Image_Embeddings, Labels in train_batch:
            
                    loss  = Img_Rec.train_on_batch(x = [Users, Items, Image_Embeddings], y = [Labels])
                    epoch_loss_metric.update_state(loss)
                    step_training_loss.append(loss)

                    """
                    if batch_cnt % 10 == 0:
                        template = ("Epoch {}, Batch {}, Current Batch Loss: {}, Average Loss: {}, Lr: {}")
                        print(template.format(epoch + 1, 
                                            batch_cnt, 
                                            loss, 
                                            epoch_loss_metric.result().numpy(), 
                                            current_lr))
                        """

                    batch_cnt += 1
                    
                    del [Users, Items, Image_Embeddings, Labels]
                    gc.collect()       


                epoch_loss = float(epoch_loss_metric.result().numpy()) 
                epoch_training_loss.append(epoch_loss) 
                #print('Average training losses over epoch done %d: %.4f' % (epoch, epoch_loss,)) 
                
                # Save training loss
                save_file_path = training_model_loss + 'cp-epoch:{epoch:d}-step-loss.npz' 
                save_file_path = save_file_path.format(epoch = epoch, ensemble = 0)   
                hlpwrite.save_compressed_numpy_array_data(save_file_path, step_training_loss)  

                #print('='*50)
                #print('\n')
                #print('\n')
                gc.collect()
            
            # Save training loss per epoch
            save_file_path = end_of_training_loss + 'cp-epoch-loss.npz'
            save_file_path = save_file_path.format(ensemble = 0)  
            hlpwrite.save_compressed_numpy_array_data(save_file_path, epoch_training_loss) 
            

            # Save the ensemble model
            save_file_path = saved_training_model + '/Img_Rec_model.h5'
            save_file_path = save_file_path.format(epoch = epoch, ensemble = 0)   

            if not os.path.exists(os.path.dirname(save_file_path)):
                os.makedirs(os.path.dirname(save_file_path))

            Img_Rec.save(save_file_path)
            print(f"Saved model after end of epoch: {epoch}")

            del [train_batch]
            gc.collect()

    def load_all_image_based_resnet_models(self, n_models = -1 ):

        """
        load models of image based model that is trained on different set of negative sample 
        n_models: number of model to load. By default = -1 meanse all the model in the folder
        """
        if len(self.all_models) == 0 :

            log.write_log('Load model started...', log.logging.DEBUG)
            models_paths = os.path.join( 
                                        hlpread.read_yaml_key(CONFIGURATION_PATH, 'model', 'output_folder'),
                                        hlpread.read_yaml_key(CONFIGURATION_PATH, 'image-based-ensemble-models', 'models-ensemble-outputs-folder')
                                        )

            ensemble = 'ensemble_{ensemble:d}'
            ensemble_models_paths = os.path.join(  models_paths,
                                                    ensemble,
                                                    'Img_Rec_model.h5'
                                                    #'Img_Rec_model.json'
                                                )    

            self.all_models = []
            
            if n_models == -1:
                n_models = 0
                for entry in os.listdir(models_paths):
                    if re.search('ensemble_', entry):
                        n_models += 1

            for i in range(n_models):

                model_path = ensemble_models_paths.format(ensemble = i)  
                if os.path.exists(model_path) == True:

                    #self.all_models.append(model_from_json(read_object(model_path)))
                    self.all_models.append(load_model(model_path, custom_objects = {'WeightLossBinaryCrossentropy': WeightLossBinaryCrossentropy}))
                    self.load_theshold_model(i)
                
                else:
                    log.write_log(f'Ensemble model: {model_path} does not exists.', log.logging.DEBUG) 

            log.write_log('Load model completed...', log.logging.DEBUG)
            
    def load_theshold_model(self, nmodel):

        threshold = hlpread.read_yaml_key(CONFIGURATION_PATH, 'image-based-ensemble-models', 'ensemble-thresholds')
        self.model_threshold[nmodel] = threshold[nmodel]
        
    def stacked_predict(self, predict_batch):
        
        """
        create stacked model input dataset as outputs from the ensemble
        """     
    
        stackX = pd.DataFrame()
        for  Users, Items, Image_Embeddings in predict_batch:
            dfparam = pd.DataFrame(
                                    {
                                        'user_id': Users,
                                        'item_id': Items,  
                                    }
                                )

            for i, model in enumerate(self.all_models):
                yhat = model.predict([Users, Items, Image_Embeddings])
                yhat = tf.nn.sigmoid(yhat)
                dfparam['y_hat_'+ str(i) ] = transform_logist_label(yhat, self.model_threshold[i])

            stackX = pd.concat([stackX, dfparam], axis = 0)

            del [Users, Items, Image_Embeddings]
            gc.collect()

        col = ['y_hat_' + str(x) for x in self.model_threshold.keys()] 
        stackX['y_hat'] = np.squeeze( mode(stackX[col], axis = 1)[0])
        """
        for i, x in enumerate(self.model_threshold.keys()):
            
            col = 'y_hat_' + str(x)
            if i == 0:
                stackX['y_hat'] = yhat[col]
            else:
                stackX['y_hat'] = yhat[col] | stackX['y_hat']
        """

        return stackX
    
    def predict(self, customer_id , relevent_recommend_articleids, n_models = 5): #n_models = -1

        self.load_all_image_based_resnet_models(n_models)

        predict_user_liking = pd.DataFrame()
        predict_user_liking['customer_id'] = [customer_id] * len(relevent_recommend_articleids)
        predict_user_liking['article_id'] = relevent_recommend_articleids
        
        #Pipeline to transform customer/article and generate image path
        predict_user_liking = self.feature_eng(predict_user_liking)
        
        predict_batch = (tf.data.Dataset
                        .from_tensor_slices((predict_user_liking['user_id'],
                                             predict_user_liking['item_id'],
                                             predict_user_liking['image_path']))
                       .map(decode_train_image, num_parallel_calls = tf.data.experimental.AUTOTUNE) 
                       .prefetch(126) 
                       .batch(126)                             
                       )
        
        yhat = self.stacked_predict(predict_batch)


        del [predict_batch]
        gc.collect()

        predict_user_liking = predict_user_liking.merge(yhat, on = ['user_id', 'item_id'], how = 'inner')
        predict_user_liking = predict_user_liking[predict_user_liking.y_hat] #Get list of items user might like

        return predict_user_liking

        #relevent_item = list(yhat[yhat.rank == True].item_id)
        #relevent_item = self.article_mapping_obj.inverse_transform(relevent_item)
        #return relevent_item 
        

       


        