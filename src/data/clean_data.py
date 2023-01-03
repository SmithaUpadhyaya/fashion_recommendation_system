from src.data.customer_user_id_mapping import map_customer_to_user_id
from src.data.article_item_id_mapping import map_article_to_item_id
from src.data.clean_article import clean_article_data
from utils.read_utils import  read_yaml_key, read_csv
from utils.exception import RecommendationException
from utils.write_utils import save_to_parquet
import logs.logger as log
import pandas as pd
import numpy as np
import argparse
import sys
import os
import gc


def clean_customer_data(config_path):

    try:

      """
      clean customer master data
      config_path: str path of the config.yaml file 
      """
      log.write_log('Cleaning customer information started...', log.logging.DEBUG)

      CUSTOMER_MASTER_TABLE = os.path.join( 
                                          read_yaml_key(config_path,'data_source','data_folders'),
                                          read_yaml_key(config_path,'data_source','raw_data_folder'),
                                          read_yaml_key(config_path,'data_source','customer_data'),
                                      )

      categorical_column = lambda x: ('NONE' if pd.isna(x) or len(x) == 0 else 'NONE' if x =='None' else x)

      log.write_log('Reading customer csv file...', log.logging.DEBUG)
      df_customer = read_csv(CUSTOMER_MASTER_TABLE,
                            converters = 
                                        {                                
                                        'fashion_news_frequency': categorical_column   
                                        },
                            usecols = ['customer_id', 'FN', 'Active', 'club_member_status','fashion_news_frequency', 'age'],
                            dtype = 
                                    {
                                      'club_member_status': 'category'                                  
                                    }                          
                          )

      log.write_log('Fill NaN values for boolean feature "FN", "Active" as false...', log.logging.DEBUG)
      df_customer.FN.fillna(0, inplace = True)
      df_customer.FN = df_customer.FN.astype(bool)

      df_customer.Active.fillna(0, inplace = True)
      df_customer.Active = df_customer.Active.astype(bool)

      df_customer.rename(columns = {'FN':'subscribe_fashion_newsletter','Active':'active'}, inplace = True)
      df_customer['fashion_news_frequency'] = df_customer['fashion_news_frequency'].astype('category')

      ###################################### Transform customer_id to hash that will reduce size of the file ###################################### 
      log.write_log('Transform customer_id to hash that will reduce size of the file...', log.logging.DEBUG)
      df_customer.customer_id = df_customer.customer_id.apply(lambda x: int(x[-16:],16) ).astype('int64')


      ######################################     Remove Outlier       ######################################    
      log.write_log(f'Remove Outlier from "Age" feature...', log.logging.DEBUG)

      Q1, Q3 = np.nanpercentile(df_customer.age, [25, 75])  #Since we have nan value in age column np.percentile was always giving nan output
      IQR = Q3 - Q1 
      log.write_log(f'Percentile for the given data. Q1-25%: {Q1}, Q3-75%: {Q3}. Interquartile range: {IQR}', log.logging.DEBUG)
      
      up_lim = Q3 + 1.5 * IQR
      log.write_log(f'Outlier upper limit: {up_lim}', log.logging.DEBUG)

      #Instead of droping the customer we shall mark them as outlier in a seperate column
      outlier_col = 'is_outlier'
      df_customer[outlier_col] = df_customer.age > up_lim 
      df_customer[outlier_col] = df_customer[outlier_col].astype(bool)    
      log.write_log(f'Marked the records as outlier in {outlier_col} feature.', log.logging.INFO)

      ######################################     Missing value in age feature       ######################################  
    
      age_median = df_customer.age.median(skipna = True)
      log.write_log(f'Replace median age {age_median} for missing value in age feature...', log.logging.DEBUG)

      df_customer.age = df_customer.age.fillna(value = age_median)
      df_customer.age = np.uint8(df_customer.age)

      ######################################    Add Event TimeStamp Column        ######################################
      #Since this is is clean master table and not feature. so no need to add timestamp column
      #log.write_log('Add Event timestamp feature that will be used when upload the file at Feature stoe...', log.logging.DEBUG)
      #today_date = datetime.now().date()
      #df_customer['event_timestamp'] = pd.Timestamp(day = today_date.day, month = today_date.month, year = today_date.year)
      #df_customer['event_timestamp'] = pd.date_range(end = pd.Timestamp.now(), periods = len(df_customer), freq = 'S')

      df_customer.reset_index(drop = True, inplace = True)

      ######################################     Save as        ######################################
      log.write_log('Save the cleaned customer as parquet file format...', log.logging.DEBUG)
      CLEAN_CUSTOMER_MASTER_TABLE = os.path.join( 
                                                  read_yaml_key(config_path,'data_source','data_folders'),
                                                  read_yaml_key(config_path,'data_source','processed_data_folder'),
                                                  read_yaml_key(config_path,'data_source','clean_customer_data'),                                            
                                                )
      
      save_to_parquet(df_customer, CLEAN_CUSTOMER_MASTER_TABLE)
      
      del [df_customer]
      gc.collect()

      return

    except Exception as e:

        raise RecommendationException(e, sys) 

def clean_transaction_data(config_path):

    try:

        """
        clean transaction data 
        config_path: str path of the config.yaml file 
        """
        log.write_log('Cleaning transaction information started...', log.logging.DEBUG)

        TRANSACTION_MASTER_TABLE = os.path.join( 
                                            read_yaml_key(config_path,'data_source','data_folders'),
                                            read_yaml_key(config_path,'data_source','raw_data_folder'),
                                            read_yaml_key(config_path,'data_source','transaction_data'),
                                        )

        log.write_log('Reading transaction csv file...', log.logging.DEBUG)
        df_transaction = read_csv(TRANSACTION_MASTER_TABLE)

        log.write_log('Change datatype of the features...', log.logging.DEBUG)
        df_transaction.t_dat = pd.to_datetime(df_transaction.t_dat)
        df_transaction.article_id = pd.to_numeric(df_transaction.article_id, downcast = 'unsigned')
        #df_transaction.price = df_transaction.price.astype('float16')
        df_transaction.price = df_transaction.price.astype('float32') #Since saving to parquet does not support float16 was giving error at time of save

        log.write_log('Transform customer_id to hash that will reduce size of the file...', log.logging.DEBUG)
        df_transaction.customer_id = df_transaction.customer_id.apply(lambda x: int(x[-16:],16) ).astype('int64')
        
        df_transaction.sales_channel_id =  df_transaction.sales_channel_id.apply(lambda x: False if (x == 1) else True)
        df_transaction.rename(columns = {'sales_channel_id':'online_sale'}, inplace = True)

        log.write_log('Dropping any duplicate records from transaction data...', log.logging.DEBUG)
        df_transaction.drop_duplicates(keep = 'first', inplace = True)

        log.write_log('Dropping if there are items where user has purchase same item more them once...', log.logging.DEBUG)
        #Check if there are items where user has purchase same item more them once. 
        df_transaction.drop_duplicates(subset = ['customer_id','article_id'], keep = 'last', inplace = True)

  
        #Convert t_dat time to timestamp feature column that would be help when create feature 
        #Did not work. Lets try if FEAST work with just time value
        #log.write_log('Convert t_dat feature to timestamp column by add 00h00m00s to the transaction date...', log.logging.DEBUG)
        #df_transaction['t_dat_timestamp'] = pd.Timestamp(day = df_transaction.t_dat.dt.day, 
        #                                                 month = df_transaction.t_dat.dt.month, 
        #                                                 year = df_transaction.t_dat.dt.year)

        log.write_log('Assign a unique transaction id...', log.logging.DEBUG)
        df_transaction['unique_transaction_id'] = range(0, len(df_transaction), 1)

        df_transaction.reset_index(drop = True, inplace = True)

        ######################################     Save as        ######################################
        log.write_log('Save the cleaned transaction as parquet file format...', log.logging.DEBUG)
        CLEAN_TRANSACTION_MASTER_TABLE = os.path.join( 
                                                    read_yaml_key(config_path,'data_source','data_folders'),
                                                    read_yaml_key(config_path,'data_source','processed_data_folder'),
                                                    read_yaml_key(config_path,'data_source','clean_transaction_data'),
                                                )  
        save_to_parquet(df_transaction, CLEAN_TRANSACTION_MASTER_TABLE)

        del [df_transaction]
        gc.collect()

        return

    except Exception as e:

        raise RecommendationException(e, sys) 


if __name__ == '__main__':    

    args = argparse.ArgumentParser()
    args.add_argument("--config", default = "config/config.yaml")
    args.add_argument("--data", default = "customer")

    parsed_args = args.parse_args()
        
    log.write_log(f'Read configuration from path: {parsed_args.config}', log.logging.INFO)
    #CONFIGURATION_PATH = parsed_args.config

    log.write_log(f'Clean data of {parsed_args.data}', log.logging.DEBUG)

    if parsed_args.data.upper() == "customer".upper():

        print('Clean customer data started started.')
        clean_customer_data(parsed_args.config)    

        print('Mapping customer to userid started.')
        map_customer_to_user_id(parsed_args.config)


    elif parsed_args.data.upper() == "article".upper():

        print('Clean article data started started.')
        clean_article_data(parsed_args.config)

        print('Mapping article to item started.')
        map_article_to_item_id(parsed_args.config)


    elif parsed_args.data.upper() == "transaction".upper():
        
        clean_transaction_data(parsed_args.config)        
        
    print(f'Sucessfully cleaned {parsed_args.data.upper()} data.')

    
