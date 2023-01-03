from utils.read_utils import  read_from_parquet, read_yaml_key, read_from_pickle
from utils.exception import RecommendationException
from utils.write_utils import save_to_parquet
import logs.logger as log 
import sys
import os
import gc

def map_customer_to_user_id(config_path):

    try:

        """
        create mapping of unique customer to user id
        This will also help when we need to transform customerid to interger for train/predict model
        """
        CLEAN_CUSTOMER_MASTER_TABLE = os.path.join( 
                                                  read_yaml_key(config_path,'data_source','data_folders'),
                                                  read_yaml_key(config_path,'data_source','processed_data_folder'),
                                                  read_yaml_key(config_path,'data_source','clean_customer_data'),                                            
                                                )

        df_customer = read_from_parquet(CLEAN_CUSTOMER_MASTER_TABLE)

        log.write_log('Create Customer to unique User id mapping...', log.logging.DEBUG)

        """
        #Old code when we had created mapping pkl file that store customer to user mapping
        customerid_mapping_path = os.path.join( read_yaml_key(config_path,'data_source','data_folders'),
                                                read_yaml_key(config_path, 'data_source', 'interim_data_folder'),
                                                read_yaml_key(config_path, 'customer_map', 'customerid_to_userid')
                                                )
        customer_to_user_mapping = read_from_pickle(customerid_mapping_path, compression = '')
        df_customer['user_id'] = df_customer['customer_id'].map(customer_to_user_mapping)
        df_customer['user_id_aug'] = df_customer['user_id']
        df_customer['user_id_aug'] = range(len(customer_to_user_mapping.keys()) + 1, df_customer.shape[0] + len(customer_to_user_mapping.keys()) + 1, 1)
        df_customer['user_id'] = df_customer['user_id'].fillna(df_customer.pop('user_id_aug'))
        df_customer['user_id'] = df_customer['user_id'].astype(int)
        """
        df_customer['user_id'] = range(0, df_customer.user_id.nunique())
        
        #df_customer = df_customer[['customer_id', 'user_id', 'event_timestamp']]
        df_customer = df_customer[['customer_id', 'user_id']]

        log.write_log('Saving mapping to file started...', log.logging.DEBUG)

        CUSTOMER_USER_MAPPING = os.path.join( 
                                            read_yaml_key(config_path,'data_source', 'data_folders'),
                                            read_yaml_key(config_path,'data_source', 'processed_data_folder'),                                               
                                            read_yaml_key(config_path,'data_source', 'customer_user_mapping_data'),
                                            )
        save_to_parquet(df_customer, CUSTOMER_USER_MAPPING)

        log.write_log('Saving mapping for customer to unique interger user id completed.', log.logging.DEBUG)

        del [df_customer]
        gc.collect()

        return

    except Exception as e:

        raise RecommendationException(e, sys) 