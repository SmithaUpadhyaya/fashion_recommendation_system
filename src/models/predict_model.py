from src.models.resnet_image_based_model import resnet_based_prevence
from src.models.ranking_model import ranking_model
from config.config import CONFIGURATION_PATH
from utils.read_utils import  read_yaml_key
import utils.read_utils as hlpread
from datetime import timedelta
import logs.logger as log
import numpy as np
import os

class Recommendation:

    def __init__(self):

        self.popular_items_last_year_list = []

        self.resnet_obj = None
        self.ranking_obj = None
        #self.customer_mapping = None
        #self.article_mapping_obj = None
        self.pairs_items_obj = None
        self.items_paried_to_gether_obj = None
        self.transaction_data_obj = None

    def popular_items_last_year(self, ntop = 15):

        if self.pairs_items_obj == None:            
            POPULAR_ITEMS_LAST_YEAR = os.path.join( 
                                                    hlpread.read_yaml_key(CONFIGURATION_PATH,'model', 'output_folder'),
                                                    hlpread.read_yaml_key(CONFIGURATION_PATH, 'candidate-popular-items-last-year', 'popular-items-last-year-folder'),
                                                    hlpread.read_yaml_key(CONFIGURATION_PATH, 'candidate-popular-items-last-year', 'popular-items-last-year-output')
                                                    )
            self.pairs_items_obj = hlpread.read_object(POPULAR_ITEMS_LAST_YEAR)

        if len(self.popular_items_last_year_list) != ntop:
            
            self.popular_items_last_year_list = list(self.pairs_items_obj.keys())[:ntop]

        return self.popular_items_last_year_list

    def items_paried_together(self, customer_id, popular_items_last_year_list,  ntop = 20 ):

        customer_id = int(customer_id[-16:],16)

        #Candinate 1: Item paried together for last year popular items
        #Get ntop item that are paried together for the popular_items_that are purchase last year.
        #items_paried_to_gether is key value pair where each key is an item and values is list of item that user has purchase together.

        if self.items_paried_to_gether_obj == None:            
            ITEMS_PAIRED_TOGETHER = os.path.join( 
                                                    hlpread.read_yaml_key(CONFIGURATION_PATH,'model', 'output_folder'),
                                                    hlpread.read_yaml_key(CONFIGURATION_PATH,'candidate-item-purchase-together', 'item-purchase-together-folder'),
                                                    hlpread.read_yaml_key(CONFIGURATION_PATH,'candidate-item-purchase-together', 'item-purchase-together-output')
                                                )
            self.items_paried_to_gether_obj = hlpread.read_object(ITEMS_PAIRED_TOGETHER)

        items_paried_to_gether = [self.items_paried_to_gether_obj[x][ : ntop] for x in popular_items_last_year_list]
        items_paried_to_gether = list(np.concatenate(items_paried_to_gether).flat) 

        #Candinate 2: If user has made any recent purchase in last 1 month. Recommend items that are brought together with those items by other users.
        #t_dat = date.today()
        if self.transaction_data_obj == None:
            TRANSACTION_DATA = os.path.join( hlpread.read_yaml_key(CONFIGURATION_PATH,'data_source','data_folders'),
                                             hlpread.read_yaml_key(CONFIGURATION_PATH,'data_source','processed_data_folder'),
                                             hlpread.read_yaml_key(CONFIGURATION_PATH,'data_source','train_data')
                                            )
            self.transaction_data_obj = hlpread.read_from_parquet(TRANSACTION_DATA)
            self.last_tran_date_record =  self.transaction_data_obj.t_dat.max() #This will be replace with the current day in live environment


        article_user_purchase_last_4week_tran = self.transaction_data_obj[
                                                    (self.transaction_data_obj.t_dat >= (self.last_tran_date_record - timedelta(weeks = 4))) &
                                                    (self.transaction_data_obj.customer_id == customer_id)
                                                    ].article_id.unique()
        
        #If user there is no records of the user transaction in last 4 week. We still will bring Top n popular items sold in last 1 week as candinate.
        item_paried_to_gether_for_items_user_purchase_last_4week = []
        if len(article_user_purchase_last_4week_tran) > 0:
            
            item_paried_to_gether_for_items_user_purchase_last_4week = [self.items_paried_to_gether_obj[x][ : ntop] for x in article_user_purchase_last_4week_tran]
            item_paried_to_gether_for_items_user_purchase_last_4week = list(np.concatenate(item_paried_to_gether_for_items_user_purchase_last_4week).flat) 


        #Candinate 3: Popular items that are sold in last 1 week
        vc = self.transaction_data_obj[self.transaction_data_obj.t_dat >= (self.last_tran_date_record - timedelta(weeks = 1))].article_id.value_counts()
        vc = vc.reset_index()
        vc.rename(columns = {'index': 'article_id', 'article_id': 'cnt'}, inplace = True)
        vc = vc[~vc.article_id.isin(items_paried_to_gether)]
        vc = vc[~vc.article_id.isin(item_paried_to_gether_for_items_user_purchase_last_4week)]
        vc.sort_values(by = 'cnt', ascending = False,  inplace = True)

        top_items_sold_last_1week = list(vc[:ntop].article_id)

        #Final Candinate lists
        items_paried_to_gether = popular_items_last_year_list + items_paried_to_gether + item_paried_to_gether_for_items_user_purchase_last_4week + top_items_sold_last_1week 
        items_paried_to_gether = list(set(items_paried_to_gether)) #List does not have function unique

        return items_paried_to_gether

    def candinate_generation(self, customer_id):

        candinate_lists = []

        #popular item sold last year at same time
        candinate_lists = self.popular_items_last_year()
        candinate_lists = self.items_paried_together(customer_id, candinate_lists)

        return candinate_lists
    
    def find_relevent_items_of_candinate(self, customer_id, candinate_items):

        if self.resnet_obj == None:
            self.resnet_obj = resnet_based_prevence(False)

        #Find relevent score/y_hat of candinate items
        df_relevent_items = self.resnet_obj.predict(customer_id, candinate_items)
        #customer_id,article_id, y_hat
        return df_relevent_items

    def rank_relevent_items(self, recommended_items):

        if self.ranking_obj == None:
            
            saved_model = os.path.join(read_yaml_key(CONFIGURATION_PATH,'model','output_folder'),
                                       read_yaml_key(CONFIGURATION_PATH,'lightgbm-param','ranking-model-output-folder'),
                                       #read_yaml_key(CONFIGURATION_PATH,'lightgbm-param','ranking-model-5feature-folder'),
                                       read_yaml_key(CONFIGURATION_PATH,'lightgbm-param','saved_model')
                                      )

            saved_pipeline = os.path.join(read_yaml_key(CONFIGURATION_PATH,'model','output_folder'),
                                          read_yaml_key(CONFIGURATION_PATH,'lightgbm-param','ranking-model-output-folder'),
                                          read_yaml_key(CONFIGURATION_PATH,'lightgbm-param','saved_engg_pipeline')
                                        )

            self.ranking_obj = ranking_model(saved_model, saved_pipeline, CONFIGURATION_PATH)

        ranked_recommended_list = self.ranking_obj.predict(recommended_items)
        
        return ranked_recommended_list

    def predict(self, customer_id):

        recommended_items = []
        
        #Step 1: Transform customer_id to userid i.e convert the customer_id to hash equivalent
        #Handel it in feature pipeline
        #user_id = customer_id.apply(lambda x: int(x[-16:],16) ).astype('int64')
        #if  self.customer_mapping == None:
        #     self.customer_mapping = encoder_customer_userid()
        #user_id = self.customer_mapping.transform(user_id)

        #Step 2: Get the candinates to recommends
        log.write_log(f'Get candinate for customer: {customer_id}...', log.logging.DEBUG)
        candinate_lists = self.candinate_generation(customer_id)

        #Step 3: Of recommends items get the relevent items
        log.write_log(f'Find relevent items from shortlisted candinate for customer: {customer_id}...', log.logging.DEBUG)
        relevent_items = self.find_relevent_items_of_candinate(customer_id, candinate_lists)

        #Step 4: Rank the final items to recommends
        log.write_log(f'Rank the relevent items from shortlisted candinate for customer: {customer_id}...', log.logging.DEBUG)
        recommended_items = self.rank_relevent_items(relevent_items)

        #Step 5: Transform item_id to article_id
        #Ranking model will return dataframe with articleids and itemids
        #if self.article_mapping_obj == None:
        #    self.article_mapping_obj = encode_article_itemid()
        #recommended_items = self.article_mapping_obj.inverse_transform(recommended_items)

        #Step 6: Return top 15 items based on rank
        log.write_log(f'Return Top 10 relevent items to recommend for the customer: {customer_id}...', log.logging.DEBUG)
        return recommended_items[:15]



##################################################################33
"""
Sample Customer ID: 
0003e867a930d0d6842f923d6ba7c9b77aba33fe2a0fbf4672f30b3e622fec55
000525e3fe01600d717da8423643a8303390a055c578ed8a97256600baf54565
fff4d3a8b1f3b60af93e78c30a7cb4cf75edaf2590d3e593881ae6007d775f0f
fffef3b6b73545df065b521e19f64bf6fe93bfd450ab20e02ce5d1e58a8f700b
"""