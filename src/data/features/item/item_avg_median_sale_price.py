from utils.read_utils import  read_yaml_key, read_from_parquet
from utils.exception import RecommendationException
from utils.write_utils import save_to_parquet
import logs.logger as log 
import pandas as pd
import sys
import gc
import os

def calculate_avg_median_sale_price_item(config_path):

    try:

        log.write_log('Calculate average & median of sale price for each item started.', log.logging.DEBUG)

        log.write_log('Loading cleaned train transaction data...', log.logging.DEBUG)
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

        log.write_log('Compute average sale price, median sale price and sales count of items as on the transaction date...', log.logging.DEBUG)
        
        #This tell us what is the sale price of the item at the given t_dat. To know if the sales price of the item effect the purchase of the item by users
        
        """
        #Old logic where we just calculate the article_id sales for a given day. We want to know for all the item
        df_item_feature = train_tran.groupby(['article_id','t_dat']).agg(                                                       
                                                                        avg_sale_price = ('price', 'mean'),
                                                                        median_sale_price = ('price','median'),
                                                                        t_dat_sales_cnt = ('t_dat', 'count')
                                                                        ).reset_index()
        """

        #New logic that will calculate mean and median of all the items as on given date.
        pvt_output = pd.pivot_table(data = train_tran, 
                                            index = ['t_dat'], 
                                            columns = ['article_id'], 
                                            values = 'price',
                                            aggfunc = ['mean', 'median'],
                                            #fill_value = 0,
                                            #margins = True, margins_name = 'Total',                               
                                    )
        #https://towardsdatascience.com/functions-that-generate-a-multiindex-in-pandas-and-how-to-remove-the-levels-7aa15ac7ca95

        pvt_output_mean = pvt_output.xs(
                                        key='mean', 
                                        axis=1, 
                                        level=0, 
                                        drop_level=True)

        pvt_output_mean.reset_index(inplace = True)

        pvt_output_mean = pd.melt(pvt_output_mean, 
                                    id_vars = 't_dat', 
                                    value_vars = pvt_output_mean.columns)

        pvt_output_mean.rename(columns = {'value': 'item_mean_sales_price'}, inplace = True)

        #pvt_output_mean.shape
        #pvt_output_mean.columns


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

        pvt_output_median.rename(columns = {'value': 'item_median_sales_price'}, inplace = True)

        #pvt_output_median.shape
        #pvt_output_median.columns

        item_median_mean_sales_price = pvt_output_median.merge(pvt_output_mean, on = ['t_dat', 'article_id'], how = 'inner')
        item_median_mean_sales_price.sort_values(['article_id','t_dat'], inplace = True)

        del [pvt_output, train_tran, pvt_output_median, pvt_output_mean]
        gc.collect()


        #method = 'ffill': Fill NAN with last valid sales value when the item was sold.
        #There are case when the item sold on a particular day is less/cheaper then previous day.

        item_median_mean_sales_price['item_mean_sales_price'] = item_median_mean_sales_price.groupby('article_id')['item_mean_sales_price'].ffill().reset_index(drop = True) #.fillna(method ='ffill', inplace = True)
        item_median_mean_sales_price['item_median_sales_price'] = item_median_mean_sales_price.groupby('article_id')['item_median_sales_price'].ffill().reset_index(drop = True)


        #For a given date we would like to know the yesterday/previous sales value of the item
        item_median_mean_sales_price['item_prev_mean_sales_price'] = item_median_mean_sales_price.groupby('article_id')['item_mean_sales_price'].shift()
        item_median_mean_sales_price['item_prev_median_sales_price'] = item_median_mean_sales_price.groupby('article_id')['item_median_sales_price'].shift()

        #item_median_mean_sales_price.head()

        #log.write_log('Add Event timestamp feature that will be used when upload the file at feature store...', log.logging.DEBUG)
        #df_item_feature['event_timestamp'] = pd.date_range(end = pd.Timestamp.now(), periods = len(df_item_feature), freq = 'S')
        
        #Change the datatype to INT32/FLOAT64. Read FEASTReadme.txt to understand
        item_median_mean_sales_price.t_dat = pd.to_datetime(item_median_mean_sales_price.t_dat, utc = True)
        #df_item_feature.avg_sale_price = pd.to_numeric(df_item_feature.avg_sale_price .astype('float32'))
        #df_item_feature.median_sale_price = pd.to_numeric(df_item_feature.median_sale_price .astype('float32'))
        #df_item_feature.t_dat_sales_cnt = pd.to_numeric(df_item_feature.t_dat_sales_cnt .astype('int32'))
        
        ITEM_AVG_SALES_PRICE = os.path.join( 
                                                read_yaml_key(config_path,'data_source','data_folders'),
                                                read_yaml_key(config_path,'data_source','feature_folder'),
                                                read_yaml_key(config_path,'article_feature','article_folder'),
                                                read_yaml_key(config_path,'article_feature','item_avg_median_sales_price'),
                                            )
        
        save_to_parquet(df_item_feature, ITEM_AVG_SALES_PRICE)
        log.write_log('Calculate average & median of sale price for each item completed.', log.logging.DEBUG)
        
        del [df_item_feature, train_tran]
        gc.collect()

        return

    except Exception as e:

        raise RecommendationException(e, sys) 