from utils.read_utils import  read_yaml_key, read_from_parquet
from utils.write_utils import save_to_parquet
from utils.exception import RecommendationException
import logs.logger as log 
import pandas as pd
import sys
import gc
import os

def user_purchase_elapsed_days_from_prev_tran(config_path):
      
      try:
        
        log.write_log('Loading cleaned article data...', log.logging.DEBUG)

        #CLEAN_TRANSACTION_MASTER_TABLE = os.path.join( 
        #                                        read_yaml_key(config_path,'data_source','data_folders'),
        #                                        read_yaml_key(config_path,'data_source','feature_folder'),
        #                                        read_yaml_key(config_path,'transaction_feature','transaction_folder'),
        #                                        read_yaml_key(config_path,'transaction_feature','clean_transaction_folder'),
        #                                ) 

        #train_tran = read_from_parquet(CLEAN_TRANSACTION_MASTER_TABLE)
        
        TRAINING_DATASET_PATH = os.path.join( 
                                              read_yaml_key(config_path,'data_source','data_folders'),
                                              read_yaml_key(config_path,'data_source','processed_data_folder'),
                                              read_yaml_key(config_path,'data_source','train_data'),
                                            ) 
        train_tran = read_from_parquet(TRAINING_DATASET_PATH)

        log.write_log('Calculate average number of days elapsed for any user next transaction...', log.logging.DEBUG)
        #Steps: Calculate average number of days elapsed for any user's next transaction  
        # 1. Count number of items purchase per transaction 
        # 2. Calculate number of days pass after the first transaction

        #Count number of items purchase per transaction 
        grp_user_tran = (train_tran[['customer_id','t_dat']]
                                .groupby(['customer_id','t_dat'])['t_dat']
                                .count()
                                .reset_index(name = "items_per_tran")
                                .sort_values(by = ['customer_id','t_dat'], 
                                                ascending = True)
                        )

        #Calculate number of days pass after each transaction user made
        grp_user_tran['previous_purchase'] = grp_user_tran.groupby('customer_id')['t_dat'].shift(periods = 1)
        grp_user_tran['previous_purchase'] = pd.to_datetime(grp_user_tran['previous_purchase'], errors = 'coerce')
        grp_user_tran['days_pass_since_last_purchase'] = ((grp_user_tran['t_dat'] - grp_user_tran['previous_purchase']).dt.days).fillna(0).astype(int)


        grp_user_time_elpased = grp_user_tran.groupby('customer_id').agg( total_no_items_purchased = ('items_per_tran', 'sum'), 
                                                                          avg_elapse_days_per_tran = ('days_pass_since_last_purchase', 'mean'),
                                                                          number_of_user_tran = ('t_dat', 'count')
                                                                        ).reset_index()   

        global_avg_elapse_days = grp_user_time_elpased['avg_elapse_days_per_tran'].mean()

        grp_user_tran = pd.merge(grp_user_tran, grp_user_time_elpased, how ='inner',on = 'customer_id')

        #Store the generated user feature in a seperate dataset which can later be used
        df_user_feature = grp_user_tran.drop_duplicates(keep = 'first', ignore_index = True)
        df_user_feature['global_avg_elapse_days'] = global_avg_elapse_days

        #Merge with event_timestamp feature of the customer
        """
        CLEAN_CUSTOMER_MASTER_TABLE = os.path.join( 
                                                read_yaml_key(config_path,'data_source','data_folders'),
                                                read_yaml_key(config_path,'data_source','feature_folder'),
                                                read_yaml_key(config_path,'customer_features','customer_folder'),
                                                read_yaml_key(config_path,'customer_features','clean_customer_data'),
                                            )
        df_customer = read_from_parquet(CLEAN_CUSTOMER_MASTER_TABLE)
        grp_user_tran = pd.merge(grp_user_tran, df_customer[['customer_id', 'event_timestamp']], how = 'inner',on = 'customer_id')
        """

        #Since we are calculating elapsed time between current and previous transaction based on t_dat
        #log.write_log('Add Event timestamp feature that will be used when upload the file at Feature stoe...', log.logging.DEBUG)
        #grp_user_tran['event_timestamp'] = pd.date_range(end = pd.Timestamp.now(), periods = len(grp_user_tran), freq = 'S')

        #Change the datatype to INT32/FLOAT64. Read FEASTReadme.txt to understand
        grp_user_tran.t_dat = pd.to_datetime(grp_user_tran.t_dat, utc = True)
        grp_user_tran.items_per_tran = pd.to_numeric(grp_user_tran.items_per_tran.astype('int32'))
        grp_user_tran.days_pass_since_last_purchase = pd.to_numeric(grp_user_tran.days_pass_since_last_purchase.astype('int32'))
        grp_user_tran.total_no_items_purchased = pd.to_numeric(grp_user_tran.total_no_items_purchased.astype('int32'))
        grp_user_tran.avg_elapse_days_per_tran = pd.to_numeric(grp_user_tran.avg_elapse_days_per_tran.astype('float32'))
        grp_user_tran.number_of_user_tran = pd.to_numeric(grp_user_tran.number_of_user_tran.astype('int32'))

        log.write_log('Saving users elapsed days to file started...', log.logging.DEBUG)

        CUSTOMER_ELAPSED_DAY = os.path.join( 
                                                read_yaml_key(config_path,'data_source','data_folders'),
                                                read_yaml_key(config_path,'data_source','feature_folder'),
                                                read_yaml_key(config_path,'customer_features','customer_folder'),
                                                read_yaml_key(config_path,'customer_features','user_purchase_elapsed_days_from_previous_tran'),
                                            )
        save_to_parquet(grp_user_tran, CUSTOMER_ELAPSED_DAY)

        log.write_log('Saving mapping for customer to unique interger user id completed.', log.logging.DEBUG)

        del [grp_user_tran, df_user_feature, grp_user_time_elpased, train_tran]#, df_customer]
        gc.collect()

        return
      except Exception as e:

        raise RecommendationException(e, sys) 