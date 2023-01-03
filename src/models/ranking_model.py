from src.models.pipeline.catagory_leave_one_out_encoder import catagory_leave_one_out_encoder
from src.models.pipeline.bin_feature_based_on_feature import bin_feature_based_on_feature
from src.models.pipeline.transform_customer_mapping import transform_customer_mapping
from src.models.pipeline.merge_catagorical_feature import merge_catagorical_feature
from src.models.pipeline.transform_article_mapping import transform_article_mapping
from src.models.pipeline.catagory_ordinal_encoder import catagory_ordinal_encoder
from utils.read_utils import  read_yaml_key, read_from_parquet, read_object
from src.models.pipeline.transform_color_rgb import transform_color_rgb
from src.models.pipeline.bin_feature import bin_feature
from utils.exception import RecommendationException
from config.config import CONFIGURATION_PATH
from utils.write_utils import save_object
from datetime import datetime, timedelta
from sklearn.pipeline import Pipeline
from feast import FeatureStore
import lightgbm as lgb
import sys
import os
import gc


class ranking_model:
  
  user_features_list = [                          
                          "user_avg_median_purchase_price_last_8week_fv: user_median_purchase_price",  
                          "user_avg_median_purchase_price_last_8week_fv: user_prev_median_purchase_price",
                          # Since two different view have same name we will get error when try to get the value together
                          # "user_avg_median_purchase_price_fv:user_prev_median_purchase_price",                  
                      ]

  user_features_list2 = ["user_avg_median_purchase_price_fv:user_median_purchase_price",
                         "user_avg_median_purchase_price_fv:user_prev_median_purchase_price"
                        ]

  
  item_features_list = [
                        "item_previous_days_sales_count_fv:prev_day_sales_cnt",
                        "item_previous_days_sales_count_fv:prev_1w_sales_cnt",
                        "item_previous_days_sales_count_fv:prev_year_sales_cnt",
                        "item_previous_days_sales_count_fv:sale_count",
                        
                        "item_avg_sales_price_fv:item_median_sales_price", 
                        "item_avg_sales_price_fv:item_prev_mean_sales_price", 
                        
                      ]

  
  
  def __init__(self, saved_model, saved_pipeline, config_path = CONFIGURATION_PATH):
      
      self.config_path = config_path
      self.saved_model_filepath = saved_model
      self.saved_pipeline_filepath = saved_pipeline
      self.fited_pipeline = False
      self.model_trained = False
      self.fs = None

      if os.path.exists(self.saved_model_filepath) == True:

        self.model_trained = True
        self.ranker_bst = lgb.Booster(model_file = self.saved_model_filepath)

      if os.path.exists(self.saved_pipeline_filepath) == True:

        self.fited_pipeline = True
        self.feature_engg = read_object(self.saved_pipeline_filepath)
  
  def init_repo_path(self, is_training):   

    if self.fs == None:

      repo_path =  os.path.join( 
                              #read_yaml_key(self.config_path, 'data_source', 'data_folders'),
                              read_yaml_key(self.config_path, 'data_source', 'feature_folder'),                                
                            )

      #Init FeatureStore
      self.fs = FeatureStore(repo_path = repo_path) 

      # serialize the latest values of features for online serving
      if is_training == False:

        fv = ['user_avg_median_purchase_price_fv',
              'user_avg_median_purchase_price_last_8week_fv', 
              #'customer_elapsed_day_fv', 
              'item_previous_days_sales_count_fv', 
              'item_avg_sales_price_fv' 
              ]

        #For Local testing 
        #last_record_date = datetime(year = 2020, month = 9, day = 22)
        #self.fs.materialize(start_date = last_record_date - timedelta(days = 1), end_date = last_record_date)
    
        #For Live 
        self.fs.materialize_incremental(end_date = datetime.utcnow() - timedelta(days = 1),
                                        feature_views = fv
                                       )

  def get_training_user_features(self, X):
   
    if self.fs == None:
      return X

    X = self.fs.get_historical_features(
                                          entity_df = X, 
                                          features = self.user_features_list
                                        ).to_df()
    #X.columns
    X.rename(columns = {'user_prev_median_purchase_price': 'user_last8week_median_purchase_price'}, 
             inplace = True)


    X = self.fs.get_historical_features(
                                          entity_df = X, 
                                          features = self.user_features_list2
                                        ).to_df()
    X.rename(columns = {'user_prev_median_purchase_price': 'user_overall_median_purchase_price'}, 
                      inplace = True)

    return X

  def get_online_user_features(self, X):

    if self.fs == None:
      return X

    X_User_Elapsed_Feat = self.fs.get_online_features(
                                                    entity_rows = X[['customer_id']].to_dict(orient = 'records'),                                   
                                                    features = self.user_features_list
                                                    ).to_df()
                                                    
    #X_User_Elapsed_Feat.rename(columns = {'user_prev_median_purchase_price': 'user_last8week_median_purchase_price'}, inplace = True)
    #Since in online store we have last value of the feature. So for a give item prev count would be yesterday sales count.
    X_User_Elapsed_Feat.rename(columns = {'user_median_purchase_price': 'user_last8week_median_purchase_price'}, 
             inplace = True)


    X_User_Median_Purchase_Feat = self.fs.get_online_features(
                                                            entity_rows = X[['customer_id']].to_dict(orient = 'records'),
                                                            features = self.user_features_list2
                                                            ).to_df()

    #X_User_Median_Purchase_Feat.rename(columns = {'user_prev_median_purchase_price': 'user_overall_median_purchase_price'}, inplace = True)
    X_User_Median_Purchase_Feat.rename(columns = {'user_median_purchase_price': 'user_overall_median_purchase_price'}, 
                      inplace = True)

    X_return = X_User_Median_Purchase_Feat.merge(X_User_Elapsed_Feat, on = ['customer_id'], how = 'inner')

    X = X.merge(X_return, on = ['customer_id'], how = 'inner')
    return X

  def get_training_item_features(self, X):
    
    if self.fs == None:
      X = self.get_items_previous_sales_details(X)
      return X
    
    X = self.fs.get_historical_features(
                                        entity_df = X, 
                                        features = self.item_features_list
                                        ).to_df()

    return X
  
  def get_online_item_features(self, X):
  
    if self.fs == None:
      X = self.get_items_previous_sales_details(X) 
      return X

    dt_items = X[['article_id']].drop_duplicates().reset_index(drop = True)
    X_Item_Feat = self.fs.get_online_features(
                                    entity_rows = dt_items.to_dict(orient = 'records'),
                                    features = self.item_features_list,
                                    ).to_df()

    #For online feature pre day sales count is the that day sales count
    X['prev_day_sales_cnt'] = X['sale_count']
    X['item_prev_median_sales_price'] = X['item_median_sales_price']
    X = X.merge(X_Item_Feat, on = ['article_id'], how = 'inner')
    return X

  def get_user_features(self, X, is_training):

    self.init_repo_path(is_training)

    if is_training == True:
      X = self.get_training_user_features(X)
    else:
      X = self.get_online_user_features(X)     

    return X

  def get_items_metadata(self, X):

    CLEAN_ARTICLES_MASTER_TABLE = os.path.join( 
                                                read_yaml_key(self.config_path,'data_source','data_folders'),
                                                read_yaml_key(self.config_path,'data_source','processed_data_folder'),
                                                read_yaml_key(self.config_path,'data_source','clean_article_data'),
                                              )

    df_article_master = read_from_parquet(CLEAN_ARTICLES_MASTER_TABLE)

    #df_article_master.head()
    #df_article_master.columns
    X = X.merge(df_article_master[['article_id',                                  
                                   'graphical_appearance_no',
                                   'colour_group_name', 
                                   'perceived_colour_master_name',

                                   'garment_group_name',
                                   'clean_product_group_name', 
                                   'clean_index_group_name',
                                   'clean_index_name', 
                                   'clean_department_name', 
                                   'new_product_type_name'
                                   ]], 
               on = ['article_id'], 
               how = 'inner')

    X.rename(columns = { 
                         'clean_product_group_name': 'product_group_name',
                         'new_product_type_name': 'product_type_name',
                         'clean_index_name':'index_name',
                         'clean_index_group_name':'index_group_name',
                         'clean_department_name':'department_name',
                      }, 
             inplace = True)

    del [df_article_master]
    gc.collect()

    return X

  def get_items_previous_sales_details(self, X):

    ALL_ITEM_SALES_COUNT = os.path.join( 
                                          read_yaml_key(self.config_path,'data_source','data_folders'),
                                          read_yaml_key(self.config_path,'data_source','feature_folder'),
                                          read_yaml_key(self.config_path,'article_feature','article_folder'),
                                          #read_yaml_key(self.config_path,'article_feature','item_sales_count'),
                                          read_yaml_key(self.config_path,'article_feature','item_prev_days_sales_count'),
                                        )

    item_sale_count = read_from_parquet(ALL_ITEM_SALES_COUNT)
    last_record_date = datetime(year = 2020, month = 9, day = 22)
    item_sale_count = item_sale_count[item_sale_count.t_dat == last_record_date - timedelta(days = 1)]

    X = X.merge(item_sale_count[[
                                  'article_id',                                  
                                  #'prev_day_sales_cnt',                                     
                                  'prev_1w_sales_cnt',
                                  'prev_year_sales_cnt',
                                  'sale_count'                                
                                ]], 
               on = ['article_id'], 
               how = 'inner')
    
    del [item_sale_count]
    gc.collect()

    ITEM_AVG_SALES_PRICE = os.path.join( 
                                            read_yaml_key(self.config_path,'data_source','data_folders'),
                                            read_yaml_key(self.config_path,'data_source','feature_folder'),
                                            read_yaml_key(self.config_path,'article_feature','article_folder'),
                                            read_yaml_key(self.config_path,'article_feature','item_avg_median_sales_price'),
                                        )
    item_median_sales_counts = read_from_parquet(ITEM_AVG_SALES_PRICE)
    last_record_date = datetime(year = 2020, month = 9, day = 22)
    item_median_sales_counts = item_median_sales_counts[item_median_sales_counts.t_dat == last_record_date - timedelta(days = 1)]

    X = X.merge(item_median_sales_counts[[
                                        'article_id',                                  
                                        'item_median_sales_price',                                     
                                        #'item_prev_mean_sales_price',                                                               
                                        ]], 
               on = ['article_id'], 
               how = 'inner')

    X.rename(columns = { "sale_count": "prev_day_sales_cnt", "item_median_sales_price": "item_prev_median_sales_price"} , inplace = True)
    

    del item_median_sales_counts
    gc.collect()

    return X

  def get_items_features(self, X, is_training):

    self.init_repo_path(is_training)

    if is_training == True:
      X = self.get_training_item_features(X)
    else:      
      X = self.get_online_item_features(X) 

    X = self.get_items_metadata(X)  

    return X

  def feature_transform(self, X, is_training):

    if self.fited_pipeline == False:

      ordinal_encoder_columns = ['product_group_name']
      ordinal_encoder_label_columns = ['label']

      target_encode_columns = ['product_desc']
      target_encode_label = ['label']
      seed = 1001

      q_list = [0.1, 0.25, 0.5, 0.75, 0.9]
      quartile_features = { 
                            'user_overall_median_purchase_price': {'groupby_col': 'product_desc', 'quartile_list': q_list},
                            'user_last8week_median_purchase_price': {'groupby_col': 'product_desc', 'quartile_list': q_list},
                            'item_prev_median_sales_price': {'groupby_col': 'product_desc', 'quartile_list': q_list},
                            'prev_day_sales_cnt': {'groupby_col': 'product_desc', 'quartile_list': q_list},
                            'prev_1w_sales_cnt': {'groupby_col': 'product_desc', 'quartile_list': q_list},
                            #'prev_2w_sales_cnt': {'groupby_col': 'product_desc', 'quartile_list': q_list},
                            #'prev_3w_sales_cnt': {'groupby_col': 'product_desc', 'quartile_list': q_list},
                            #'prev_4w_sales_cnt': {'groupby_col': 'product_desc', 'quartile_list': q_list},
                          }

      bin_features = {
                      'color': [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                      #'avg_elapse_days_per_tran': [0.25, 0.5, 0.75, 0.9],
                      #'days_pass_since_last_purchase': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 ,0.9],
                    } 

      self.feature_engg = Pipeline( steps = [
                                        ('transform_article_mapping', transform_article_mapping(config_path = self.config_path)),

                                        ('transform_customer_mapping', transform_customer_mapping(hash_conversion = False, config_path = self.config_path)),

                                        ('transform_color_rgb', transform_color_rgb()),

                                        ('merge_catagorical_feature', merge_catagorical_feature()),
      
                                        ('catagory_ordinal_encode', catagory_ordinal_encoder(ordinal_encoder_label_columns, 
                                                                                            ordinal_encoder_columns)),

                                        ('catagory_leave_one_encoder', catagory_leave_one_out_encoder(target_encode_label, 
                                                                                                      target_encode_columns, 
                                                                                                      seed)),
                                                                                                  
                                        ('bin_feature_based_on_other_features', bin_feature_based_on_feature(quartile_features)),
      
                                        ('bin_feature', bin_feature(bin_features))              
                                      ]
                              ,verbose =  True
                        )    

      X = self.feature_engg.fit_transform(X)   
      save_object(self.saved_pipeline_filepath , self.feature_engg)

    else:   
    
        X = self.feature_engg.transform(X)

    return X

  def get_features(self, X, is_training):
    
    X = self.get_items_features(X, is_training = is_training)
    
    X = self.get_user_features(X, is_training = is_training)

    X = self.feature_transform(X, is_training = is_training)

    return X

  def train_model(self, X):

    try:  

        X = self.get_features(X, True) 

        x_train = X[['label',
                    'item_id', 'user_id',
                    'product_group_name_oce',
                    'product_desc_tce',
                    'user_overall_median_purchase_price_bin', #'item_median_sales_price_for_product_type_bin', 
                    'user_last8week_median_purchase_price_bin', 
                    'item_prev_median_sales_price_bin', 
                    'prev_day_sales_cnt_bin', 
                    'prev_1w_sales_cnt_bin',                     
                    'color_bin',
                    'graphical_appearance_no', 
                    #'days_pass_since_last_purchase_bin', 
                   ]]

        x_train = x_train.sort_values(by = ['user_id', 'label'],  ascending = [True, False] , na_position = 'first')

        qids_train = x_train.groupby("user_id")["item_id"].count().reset_index()
        qids_train.columns = ['user_id','cnt']
        #qids_train = qids_train.sort_values('user_id',  ascending = True).cnt.to_pandas().to_numpy() #Code when use cudf dataframe
        qids_train = qids_train.sort_values('user_id',  ascending = True).cnt.to_numpy() #Code when use pandas dataframe

        # Relevance label for train
        y_train = x_train['label'].astype(int)


        # Keeping only the features on which we would train our model 
        ddf_x_train = x_train.drop(["user_id", "item_id", "label"], axis = 1) #, inplace = True
      
        #ddf_x_train = lgb.Dataset(data = ddf_x_train.to_pandas(), label = y_train.to_pandas(), group = qids_train, free_raw_data = False) #Code when use cudf dataframe
        ddf_x_train = lgb.Dataset(data = ddf_x_train, label = y_train, group = qids_train, free_raw_data = False) #Code when use pandas dataframe

        param = read_yaml_key(self.config_path,'lightgbm-param','param')

        self.ranker_bst = lgb.train(params = param, 
                                    num_boost_round = param['n_estimators'], 
                                    train_set = ddf_x_train,                       
                                    keep_training_booster = True
                                    )


        self.ranker_bst.save_model(self.saved_model_filepath)
    
    except Exception as e:
          raise RecommendationException(e, sys) from e  

  def predict(self, X):

      X = self.get_features(X, False)   

      col = ['product_group_name_oce',
             'product_desc_tce',
             'user_overall_median_purchase_price_bin', #'item_median_sales_price_for_product_type_bin', 
             'user_last8week_median_purchase_price_bin', 
             'item_prev_median_sales_price_bin', 
             'prev_day_sales_cnt_bin', 
             'prev_1w_sales_cnt_bin',                     
             'color_bin',
             'graphical_appearance_no', 
             #'days_pass_since_last_purchase_bin', 
            ]

      X['rank'] = self.ranker_bst.predict(data = X[col]) 
      
      X = X[['article_id', 'rank']]
      X.sort_values('rank', ascending = False, inplace = True)

      return X










