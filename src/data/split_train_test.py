from utils.read_utils import  read_yaml_key, read_from_parquet
from utils.exception import RecommendationException
from utils.write_utils import save_to_parquet
from datetime import datetime, timedelta
import logs.logger as log 
import argparse
import sys
import os
import gc


def split_train_test(config_path):

    #Will hold 1 days as test dataset and will use rest of the data for train and feature enginering
    
    #Read clean transaction data
    log.write_log('Loading cleaned transaction file...', log.logging.DEBUG)
    CLEAN_TRANSACTION_MASTER_TABLE = os.path.join( 
                                                    read_yaml_key(config_path,'data_source','data_folders'),
                                                    read_yaml_key(config_path,'data_source','processed_data_folder'),                 
                                                    read_yaml_key(config_path,'data_source','clean_transaction_data'),
                                                ) 

    train = read_from_parquet(CLEAN_TRANSACTION_MASTER_TABLE)  

    last_tran_date_record = datetime(2020, 9, 22)
    Split_t_dat = (last_tran_date_record - timedelta(days = 1))

    log.write_log('Split and save 1 day data for test dataset...', log.logging.DEBUG)
    test = train[(train.t_dat > Split_t_dat)]
    log.write_log(f'Number of records in test dataset {test.shape}.', log.logging.DEBUG)

    TEST_DATASET_PATH = os.path.join( 
                                    read_yaml_key(config_path,'data_source','data_folders'),
                                    read_yaml_key(config_path,'data_source','processed_data_folder'),
                                    read_yaml_key(config_path,'data_source','test_data'),
                                    ) 
    #log.write_log(f'Save test dataset at path: {TEST_DATASET_PATH}.', log.logging.DEBUG)
    save_to_parquet(test, TEST_DATASET_PATH)

    
    log.write_log('Split and save data for train dataset...', log.logging.DEBUG)
    train = train[(train.t_dat <= Split_t_dat)]
    log.write_log(f'Number of records in train dataset {train.shape}.', log.logging.DEBUG)
    TRAINING_DATASET_PATH = os.path.join( 
                                            read_yaml_key(config_path,'data_source','data_folders'),
                                            read_yaml_key(config_path,'data_source','processed_data_folder'),
                                            read_yaml_key(config_path,'data_source','train_data'),
                                        ) 
    save_to_parquet(train, TRAINING_DATASET_PATH)
    

    del [train, test]
    gc.collect()

    return


if __name__ == '__main__':

    try:

        args = argparse.ArgumentParser()
        args.add_argument("--config", default = "config/config.yaml")
        
        parsed_args = args.parse_args()

        log.write_log(f'Read configuration from path: {parsed_args.config}', log.logging.INFO)
        #CONFIGURATION_PATH = parsed_args.config

        log.write_log(f'Started creating features for article...', log.logging.DEBUG)

        print('Generate train and test data split..')

        split_train_test(parsed_args.config)        

        print('Data split completed.')

    except Exception as e:

        raise RecommendationException(e, sys)