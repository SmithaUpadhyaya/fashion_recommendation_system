from utils.exception import RecommendationException
from utils.write_utils import save_to_parquet
from utils.read_utils import  read_from_parquet, read_yaml_key
from datetime import datetime, timedelta
import logs.logger as log 
import pandas as pd
import argparse
import sys
import os
import gc

def generate_negative_candinate(dtran_data):

    try:

        log.write_log('Converting the items purchase by a customer to set...', log.logging.DEBUG)
        #Convert purchase items to sets
        train_purchase_item = (dtran_data.groupby(by = ['t_dat','customer_id'])['article_id']
                                    .apply(set)
                                    .reset_index(name = 'purchased_item'))
        #train_purchase_item.head()


        #For a give day convert other transaction made by other user as -ve transaction for the current user
        log.write_log('For a given day converting other transaction made by other user as -ve transaction for the current user...', log.logging.DEBUG)
        train_other_item_user_purchase = (dtran_data.groupby('t_dat')['article_id']
                                                    .apply(set)
                                                    .reset_index(name = 'other_item_user_purchase')
                                        )
        #train_other_item_user_purchase.head()

        #Merge the columns of the item purchase by current user and items purchase by other user on a given transaction date
        log.write_log('Merge the item sets purchase by current user and items purchase by other user on a given transaction date...', log.logging.DEBUG)
        train_label = train_purchase_item.merge(train_other_item_user_purchase, how = 'inner', on = 't_dat')
        train_label.head()


        del [dtran_data, train_purchase_item ,train_other_item_user_purchase]
        gc.collect()

        # Remove transaction that current user made on a give day from other item purchase. 
        # This will ensure that the +ve transaction does not consider as negative sample for current user as negative sample for current user
        log.write_log('Remove transaction that current user made on a give day as negative sample...', log.logging.DEBUG)
        train_label['neg_sample'] = train_label['other_item_user_purchase'] - train_label['purchased_item']
        train_label.drop(columns = ['other_item_user_purchase'], inplace = True)


        #Explode the postive transaction of the user 
        log.write_log('Explode the postive transaction of the user...', log.logging.DEBUG)
        pos_transaction = train_label[['t_dat', 'customer_id', 'purchased_item']].explode('purchased_item')
        pos_transaction.rename(columns = {'purchased_item': 'article_id'}, inplace = True)
        pos_transaction['label'] = 1
        #pos_transaction.head()


        #Explode the negative transaction of the user
        log.write_log('Explode the negative transaction of the user...', log.logging.DEBUG)
        neg_transaction = train_label[['t_dat', 'customer_id', 'neg_sample']].explode('neg_sample')
        neg_transaction.rename(columns= {'neg_sample': 'article_id'}, inplace = True)
        neg_transaction['label'] = 0
        #neg_transaction.head()


        del train_label
        gc.collect()

        #Concate the +ve and -ve samples and create train transaction data
        log.write_log('Concate the +ve and -ve samples and create train transaction data...', log.logging.DEBUG)
        train_tran = pd.concat([pos_transaction, neg_transaction], ignore_index = True, sort = True)

        #Sort Values
        log.write_log('Sort transaction data by transaction date...', log.logging.DEBUG)
        train_tran.sort_values(by = ['t_dat'], inplace = True)

        log.write_log('Generating transaction data with negative sample completed...', log.logging.DEBUG)

        return train_tran

    except Exception as e:

        raise RecommendationException(e, sys) 


def generate_train_data(config_path, weekday = 4):

    try:

        log.write_log('Read train dataset started...', log.logging.DEBUG)

        TRAINING_DATASET_PATH = os.path.join( 
                                                read_yaml_key(config_path,'data_source','data_folders'),
                                                read_yaml_key(config_path,'data_source','processed_data_folder'),
                                                read_yaml_key(config_path,'data_source','train_data'),
                                            ) 

        train = read_from_parquet(TRAINING_DATASET_PATH)

        last_tran_date_record = datetime(2020, 9, 22)
        train = train[train.t_dat >= (last_tran_date_record - timedelta(weeks = weekday))] 

        train = generate_negative_candinate(train)

        TRAINING_NEGATIVE_SAMPLE_DATASET_PATH = os.path.join( 
                                                read_yaml_key(config_path,'data_source','data_folders'),
                                                read_yaml_key(config_path,'data_source','processed_data_folder'),
                                                read_yaml_key(config_path,'data_source','training_data'),
                                            )

        save_to_parquet(train, TRAINING_NEGATIVE_SAMPLE_DATASET_PATH)

        del [train]
        gc.collect()

        return

    except Exception as e:

        raise RecommendationException(e, sys) 

def generate_test_data(config_path):

    try:

        TEST_DATASET_PATH = os.path.join( 
                                        read_yaml_key(config_path,'data_source','data_folders'),
                                        read_yaml_key(config_path,'data_source','processed_data_folder'),
                                        read_yaml_key(config_path,'data_source','test_data'),
                                        ) 

        test = read_from_parquet(TEST_DATASET_PATH)

        test = generate_negative_candinate(test)


        TEST_NEGATIVE_SAMPLE_DATASET_PATH = os.path.join( 
                                                read_yaml_key(config_path,'data_source','data_folders'),
                                                read_yaml_key(config_path,'data_source','processed_data_folder'),
                                                read_yaml_key(config_path,'data_source','testing_data'),
                                            )

        save_to_parquet(test, TEST_NEGATIVE_SAMPLE_DATASET_PATH)

        del [train]
        gc.collect()

        return

    except Exception as e:

        raise RecommendationException(e, sys) 

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument("--config", default = "config/config.yaml")
    args.add_argument("--data", default = "train")
    args.add_argument("--weekdata", default = "4")

    parsed_args = args.parse_args()

    print('Generation of training dataset started.')

    if parsed_args.data.upper() == "train".upper():

        generate_train_data(parsed_args.config, int(parsed_args.weekdata))

    if parsed_args.data.upper() == "test".upper():

        generate_test_data(parsed_args.config)

    print('Generation of training dataset completed.')
