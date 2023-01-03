from utils.read_utils import  read_yaml_key, read_from_parquet
from utils.exception import RecommendationException
from utils.write_utils import save_to_parquet
import logs.logger as log 
import pandas as pd
import sys
import gc
import os

#def calculate_avg_medina_purchase_price(config_path):
def calculate_avg_median_purchase_price(config_path, train_tran, islast_8week_tran = False):

    try:

        if islast_8week_tran == True:
            log.write_log('Calculating average purchase price of each user in last 8 week started...', log.logging.DEBUG)
        else:
            log.write_log('Calculating average purchase price of each user till date started...', log.logging.DEBUG)

        """
        log.write_log('Loading cleaned transaction data...', log.logging.DEBUG)
        #CLEAN_TRANSACTION_MASTER_TABLE = os.path.join( 
        #                                    read_yaml_key(config_path,'data_source','data_folders'),
        #                                    read_yaml_key(config_path,'data_source','feature_folder'),
        #                                    read_yaml_key(config_path,'transaction_feature','transaction_folder'),
        #                                    read_yaml_key(config_path,'transaction_feature','clean_transaction_folder'),
        #                            )  
        #train_tran = read_from_parquet(CLEAN_TRANSACTION_MASTER_TABLE) 

        TRAINING_DATASET_PATH = os.path.join( 
                                              read_yaml_key(config_path,'data_source','data_folders'),
                                              read_yaml_key(config_path,'data_source','processed_data_folder'),
                                              read_yaml_key(config_path,'data_source','train_data'),
                                            ) 
        train_tran = read_from_parquet(TRAINING_DATASET_PATH)

        """
        #This tell us what is the purchase price of the user at the given t_dat. 
        # To know if the what is the average price of the user is able to spend on a give date.
        """
        #Old logic where we just calculate the customer sales for a given day. We want to know for all the item
        
        df_avg_user_purchase = train_tran.groupby(['customer_id','t_dat']).agg( 
                                                                                t_date_avg_purchase_price = ('price', 'mean'),
                                                                                t_date_median_purchase_price = ('price', 'median'),
                                                                                t_date_purchase_cnt = ('t_dat', 'count')
                                                                                ).reset_index()

        """
        #New logic that will calculate mean and median purchase price of a given date.
        pvt_output = pd.pivot_table(data = train_tran, 
                                    index = ['t_dat'], 
                                    columns = ['customer_id'], 
                                    values = 'price',
                                    aggfunc = ['mean', 'median'],
                                    #fill_value = 0,
                                    #margins = True, margins_name = 'Total',                               
                                )
        pvt_output_mean = pvt_output.xs(
                                key='mean', 
                                axis=1, 
                                level=0, 
                                drop_level=True)

        pvt_output_mean.reset_index(inplace = True)

        pvt_output_mean = pd.melt(pvt_output_mean, 
                                    id_vars = 't_dat', 
                                    value_vars = pvt_output_mean.columns)

        pvt_output_mean.rename(columns = {'value': 'user_mean_purchase_price'}, inplace = True)
        #pvt_output_mean.head()


        pvt_output_median = pvt_output.xs(
                                    key = 'median', 
                                    axis = 1, 
                                    level = 0, 
                                    drop_level = True
                                )

        pvt_output_median.reset_index(inplace = True)

        pvt_output_median = pd.melt(pvt_output_median, 
                                    id_vars = 't_dat', 
                                    value_vars = pvt_output_median.columns)

        pvt_output_median.rename(columns = {'value': 'user_median_purchase_price'}, inplace = True)
        #pvt_output_median.head()

        del [pvt_output, train_tran]
        gc.collect()

        user_median_mean_purchase_price = pd.concat([pvt_output_median,pvt_output_mean], axis = 1, ignore_index = True)
        user_median_mean_purchase_price.columns = ['t_dat', 'customer_id', 'user_median_purchase_price', 't_dat_1', 'customer_id_1', 'user_mean_purchase_price']

        #Due to huge size of the data instead of using merge with join we shall use concate which will be much easier
        user_median_mean_purchase_price.drop(['t_dat_1', 'customer_id_1'], axis = 1, inplace = True)
        #user_median_mean_purchase_price.head()

        #user_median_mean_purchase_price = pvt_output_median.merge(pvt_output_mean, on = ['t_dat', 'customer_id'], how = 'inner')
        #user_median_mean_purchase_price.sort_values(['customer_id','t_dat'], inplace = True)

        del [pvt_output_median, pvt_output_mean]
        gc.collect()

        user_median_mean_purchase_price['user_mean_purchase_price'] = user_median_mean_purchase_price.groupby('customer_id')['user_mean_purchase_price'].ffill().reset_index(drop = True) #.fillna(method ='ffill', inplace = True)
        user_median_mean_purchase_price['user_median_purchase_price'] = user_median_mean_purchase_price.groupby('customer_id')['user_median_purchase_price'].ffill().reset_index(drop = True)

        user_median_mean_purchase_price['user_prev_mean_purchase_price'] = user_median_mean_purchase_price.groupby('customer_id')['user_mean_purchase_price'].shift()
        user_median_mean_purchase_price['user_prev_median_purchase_price'] = user_median_mean_purchase_price.groupby('customer_id')['user_median_purchase_price'].shift()

        user_median_mean_purchase_price.t_dat = pd.to_datetime(user_median_mean_purchase_price.t_dat, utc = True) 
       
        #log.write_log('Add Event timestamp feature that will be used when upload the file at Feature store...', log.logging.DEBUG)
        #df_avg_user_purchase['event_timestamp'] = pd.date_range(end = pd.Timestamp.now(), periods = len(df_avg_user_purchase), freq = 'S')

        #Change the datatype to INT32/FLOAT64. Read FEASTReadme.txt to understand
        #df_avg_user_purchase.t_dat = pd.to_datetime(df_avg_user_purchase.t_dat, utc = True)
        #df_avg_user_purchase.t_date_avg_purchase_price = pd.to_numeric(df_avg_user_purchase.t_date_avg_purchase_price.astype('float32'))
        #df_avg_user_purchase.t_date_median_purchase_price = pd.to_numeric(df_avg_user_purchase.t_date_median_purchase_price.astype('float32'))
        #df_avg_user_purchase.t_date_purchase_cnt = pd.to_numeric(df_avg_user_purchase.t_date_purchase_cnt.astype('int32'))

        if islast_8week_tran == True:
            log.write_log('Saving users average & median purchase price to file started...', log.logging.DEBUG)

            USER_AVG_MEDIAN_PURCHASE_PRICE_LAST_8WEEK = os.path.join( 
                                                                read_yaml_key(config_path,'data_source','data_folders'),
                                                                read_yaml_key(config_path,'data_source','feature_folder'),
                                                                read_yaml_key(config_path,'customer_features','customer_folder'),
                                                                read_yaml_key(config_path,'customer_features','user_avg_median_purchase_price_last_8week'),
                                                                )
            save_to_parquet(user_median_mean_purchase_price, USER_AVG_MEDIAN_PURCHASE_PRICE_LAST_8WEEK)

        else:

            log.write_log('Saving users average & median purchase price to file started...', log.logging.DEBUG)

            USER_AVG_MEDIAN_PURCHASE_PRICE = os.path.join( 
                                                            read_yaml_key(config_path,'data_source','data_folders'),
                                                            read_yaml_key(config_path,'data_source','feature_folder'),
                                                            read_yaml_key(config_path,'customer_features','customer_folder'),
                                                            read_yaml_key(config_path,'customer_features','user_avg_median_purchase_price'),
                                                        )

            save_to_parquet(user_median_mean_purchase_price, USER_AVG_MEDIAN_PURCHASE_PRICE)

        log.write_log('Calculating users average & median purchase price for transaction for till date completed.', log.logging.DEBUG)


        del [user_median_mean_purchase_price]
        gc.collect()
        
        #df_avg_purchase_price_per_transaction = train_tran.groupby(['customer_id','t_dat']).agg(avg_price_per_order = ('price', 'mean'),
        #                                                                                        scaled_avg_price_per_order = ('scale_price' , 'mean')
        #                                                                                        ).reset_index()

        return

    except Exception as e:

        raise RecommendationException(e, sys) 
