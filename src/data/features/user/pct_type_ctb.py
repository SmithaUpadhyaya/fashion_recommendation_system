from src.data.features.user.user_avg_median_purchase_price import calculate_avg_median_purchase_price_last_8week
from src.data.features.user.user_avg_median_purchase_price import calculate_avg_median_purchase_price
from utils.read_utils import  read_yaml_key, read_from_parquet
from utils.exception import RecommendationException
from utils.write_utils import save_to_parquet
from utils.analysis_utils import ctb_type_pct
from datetime import timedelta
import logs.logger as log 
import pandas as pd
import sys
import gc
import os

# Calculate what is the percenage of the item per product type user perfer to buy
# Thought process here is first we shoud understand what type of the product_type user's generally purchase. 
# This will help use understand which other items in same product_type can we should recommmend.

def calculate_percentage_of_product_type_user(config_path):

      try:

        log.write_log('Calculating percentage of product type user has purchase started.', log.logging.DEBUG)

        log.write_log('Loading cleaned article data...', log.logging.DEBUG)
        CLEAN_ARTICLES_MASTER_TABLE = os.path.join( 
                                                   read_yaml_key(config_path,'data_source','data_folders'),
                                                   read_yaml_key(config_path,'data_source','processed_data_folder'),                                                
                                                   read_yaml_key(config_path,'data_source','clean_article_data'),
                                                )

        df_article = read_from_parquet(CLEAN_ARTICLES_MASTER_TABLE)

        #Load transation detail to calculate the % of the item user purcase per product type
        log.write_log('Loading cleaned transaction data...', log.logging.DEBUG)
        """
        CLEAN_TRANSACTION_MASTER_TABLE = os.path.join( 
                                            read_yaml_key(config_path,'data_source','data_folders'),
                                            read_yaml_key(config_path,'data_source','processed_data_folder'),
                                            read_yaml_key(config_path,'data_source','clean_transaction_data'),
                                        )  
        train_tran = read_from_parquet(CLEAN_TRANSACTION_MASTER_TABLE)
        """
        TRAINING_DATASET_PATH = os.path.join( 
                                            read_yaml_key(config_path,'data_source','data_folders'),
                                            read_yaml_key(config_path,'data_source','processed_data_folder'),
                                            read_yaml_key(config_path,'data_source','train_data'),
                                        ) 
        train_tran = read_from_parquet(TRAINING_DATASET_PATH)       

        log.write_log('Merging product type information with the transaction data...', log.logging.DEBUG)                 
        train_tran = train_tran.merge(df_article[['article_id', 'new_product_type_name','clean_product_type_name']], on = 'article_id', how = 'left')
        last_tran_date_record = train_tran.t_dat.max() 
        
        category_list = df_article.new_product_type_name.unique()
        log.write_log(f'Number of unique product type {len(category_list)}...', log.logging.DEBUG)

        #Over all % of the type      
        #Since we are only intrested what user like in last 8 week
        log.write_log('Calculate percentage of user purchase per product type for transaction data made till date...', log.logging.DEBUG)       
        pvt_output = ctb_type_pct(train_tran, 'pct_type_ctb', category_list)

        #log.write_log('Add Event timestamp feature that will be used when upload the file at Feature store...', log.logging.DEBUG)
        #pvt_output['event_timestamp'] = pd.date_range(end = pd.Timestamp.now(), periods = len(pvt_output), freq = 'S')
        #Since we have calculated the the percengate as on last t_date
        pvt_output['last_t_dat'] = last_tran_date_record

        #Change the datatype to INT32/FLOAT64. Read FEASTReadme.txt to understand
        pvt_output.last_t_dat = pd.to_datetime(pvt_output.last_t_dat, utc = True)
        pvt_output.pct_type_ctb = pd.to_numeric(pvt_output.pct_type_ctb .astype('float32'))
       

        log.write_log('Save calculated percentage of user purchase per product type...', log.logging.DEBUG)
        USER_PCT_PRODUCT_TYPE = os.path.join( 
                                                read_yaml_key(config_path,'data_source','data_folders'),
                                                read_yaml_key(config_path,'data_source','feature_folder'),
                                                read_yaml_key(config_path,'customer_features','customer_folder'),
                                                read_yaml_key(config_path,'customer_features','user_percentage_product_type_purchase'),
                                                )
        save_to_parquet(pvt_output, USER_PCT_PRODUCT_TYPE)

        log.write_log('Save completed...', log.logging.DEBUG)

        calculate_avg_median_purchase_price(config_path, train_tran, False)
        
        del [df_article, pvt_output]
        gc.collect()
        

        
        log.write_log('Get last 8 week transaction data...', log.logging.DEBUG)
        last_8week_tran = train_tran[train_tran.t_dat >= (last_tran_date_record - timedelta(weeks = 8))]
        log.write_log('Calculate percentage of user purchase per product type for last 8 week transaction data...', log.logging.DEBUG)
        pvt_output_last_8week = ctb_type_pct(last_8week_tran,'pct_type_ctb_last_8week', category_list)

        #log.write_log('Add Event timestamp feature that will be used when upload the file at Feature stoe...', log.logging.DEBUG)
        #pvt_output_last_8week['event_timestamp'] = pd.date_range(end = pd.Timestamp.now(), periods = len(pvt_output_last_8week), freq = 'S')
        pvt_output_last_8week['last_t_dat'] = last_tran_date_record

        #Change the datatype to INT32/FLOAT64. Read FEASTReadme.txt to understand
        pvt_output_last_8week.last_t_dat = pd.to_datetime(pvt_output_last_8week.last_t_dat, utc = True)
        pvt_output_last_8week.pct_type_ctb_last_8week = pd.to_numeric(pvt_output_last_8week.pct_type_ctb_last_8week.astype('float32'))

        log.write_log('Save calculated percentage of user purchase per product type for last 8 week...', log.logging.DEBUG)
        USER_PCT_PRODUCT_TYPE = os.path.join( 
                                                read_yaml_key(config_path,'data_source','data_folders'),
                                                read_yaml_key(config_path,'data_source','feature_folder'),
                                                read_yaml_key(config_path,'customer_features','customer_folder'),
                                                read_yaml_key(config_path,'customer_features','user_percentage_product_type_purchase_last_8week'),
                                                )
        save_to_parquet(pvt_output_last_8week, USER_PCT_PRODUCT_TYPE)

        log.write_log('Calculating percentage of the user purchase completed.', log.logging.DEBUG)

        calculate_avg_median_purchase_price(config_path, last_8week_tran, True)

        del [last_8week_tran, train_tran, pvt_output_last_8week]
        gc.collect()   
        
        return

      except Exception as e:

        raise RecommendationException(e, sys) 