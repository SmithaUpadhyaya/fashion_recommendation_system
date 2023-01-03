from utils.read_utils import  read_yaml_key, read_from_parquet
from utils.exception import RecommendationException
from utils.write_utils import save_to_parquet
from datetime import timedelta
import logs.logger as log 
import pandas as pd
import sys
import os
import gc

def calculate_previous_day_item_sales(tran_data):

    log.write_log('Count sales of the item in previous transaction date started...', log.logging.DEBUG)

    """
    article_id_popularity_cnt_prev_day = (tran_data
                                          .groupby(['article_id', 't_dat'])
                                          .agg(sale_count = ('t_dat', 'count'))
                                          .groupby(level = 0, group_keys = False)
                                          .rolling(1, min_periods = 1, closed = 'left')
                                          .sum()
                                          .droplevel(0)
                                          .reset_index()
                                          )  
    """
    article_id_popularity_cnt_prev_day = (tran_data
                                     .groupby('article_id')
                                     ['sale_count']  
                                     .rolling(1, min_periods = 1, closed = 'left')
                                     .sum()
                                     .reset_index()
                                    ) 
    
    article_id_popularity_cnt_prev_day.rename(columns = {'sale_count': 'prev_day_sales_cnt'}, inplace = True)
    article_id_popularity_cnt_prev_day.drop(columns = ['level_1'], inplace = True)
    article_id_popularity_cnt_prev_day['prev_day_sales_cnt'] = article_id_popularity_cnt_prev_day['prev_day_sales_cnt'].fillna(value = 0)
    article_id_popularity_cnt_prev_day.prev_day_sales_cnt = pd.to_numeric(article_id_popularity_cnt_prev_day.prev_day_sales_cnt.astype('int32'), downcast = 'unsigned')

    log.write_log('Count sales of the item in previous transaction date completed.', log.logging.DEBUG)

    return article_id_popularity_cnt_prev_day

def calculate_previous_1week_item_sales(tran_data):

    log.write_log('Count sales of the item in previous 1 week transaction date started...', log.logging.DEBUG)

    """
    article_id_popularity_cnt_prev_1w = (tran_data
                                        .groupby(['article_id', 't_dat'])
                                        .agg(sale_count = ('t_dat', 'count'))
                                        .groupby(level = 0, group_keys = False)
                                        .rolling(7, min_periods = 1, closed = 'left')
                                        .sum()
                                        .droplevel(0)
                                        .reset_index())  
    """

    article_id_popularity_cnt_prev_1w = (tran_data
                                        .groupby('article_id')
                                        ['sale_count']  
                                        .rolling(7, min_periods = 1, closed = 'left')
                                        .sum()
                                        .reset_index()
                                        ) 

    article_id_popularity_cnt_prev_1w.rename(columns = {'sale_count':'prev_1w_sales_cnt'}, inplace = True)
    article_id_popularity_cnt_prev_1w.drop(columns = ['level_1'], inplace = True)
    article_id_popularity_cnt_prev_1w['prev_1w_sales_cnt'] = article_id_popularity_cnt_prev_1w['prev_1w_sales_cnt'].fillna(value = 0)
    article_id_popularity_cnt_prev_1w.prev_1w_sales_cnt = pd.to_numeric(article_id_popularity_cnt_prev_1w.prev_1w_sales_cnt.astype('int32'), downcast = 'unsigned')


    log.write_log('Count sales of the item in previous 1 week transaction date completed.', log.logging.DEBUG)

    return article_id_popularity_cnt_prev_1w

def calculate_previous_2week_item_sales(tran_data):

    log.write_log('Count sales of the item in previous 2 week transaction date started...', log.logging.DEBUG)

    """
    article_id_popularity_cnt_prev_2w = (tran_data
                                        .groupby(['article_id', 't_dat'])
                                        .agg(sale_count = ('t_dat', 'count'))
                                        .groupby(level = 0, group_keys = False)
                                        .rolling(14, min_periods = 1, closed = 'left')
                                        .sum()
                                        .droplevel(0)
                                        .reset_index())
    """
    article_id_popularity_cnt_prev_2w = (tran_data
                                        .groupby('article_id')
                                        ['sale_count']
                                        .rolling(14, min_periods = 1, closed = 'left')
                                        .sum()
                                        .reset_index()
                                    )

    article_id_popularity_cnt_prev_2w.rename(columns = {'sale_count':'prev_2w_sales_cnt'}, inplace = True)
    article_id_popularity_cnt_prev_2w.drop(columns = ['level_1'], inplace = True)
    article_id_popularity_cnt_prev_2w['prev_2w_sales_cnt'] = article_id_popularity_cnt_prev_2w['prev_2w_sales_cnt'].fillna(value = 0)
    article_id_popularity_cnt_prev_2w.prev_2w_sales_cnt = pd.to_numeric(article_id_popularity_cnt_prev_2w.prev_2w_sales_cnt.astype('int32'), downcast = 'unsigned')

    log.write_log('Count sales of the item in previous 2 week transaction date completed.', log.logging.DEBUG)

    return article_id_popularity_cnt_prev_2w

def calculate_previous_3week_item_sales(tran_data):

    log.write_log('Count sales of the item in previous 3 week transaction date started...', log.logging.DEBUG)
    """
    article_id_popularity_cnt_prev_3w = (tran_data
                                        .groupby(['article_id', 't_dat'])
                                        .agg(sale_count = ('t_dat', 'count'))
                                        .groupby(level = 0, group_keys = False)
                                        .rolling(21, min_periods = 1, closed = 'left')
                                        .sum()
                                        .droplevel(0)
                                        .reset_index()) 
    """
    article_id_popularity_cnt_prev_3w = (tran_data
                                        .groupby('article_id')
                                        ['sale_count']
                                        .rolling(21, min_periods = 1, closed = 'left')
                                        .sum()
                                        .reset_index())

    article_id_popularity_cnt_prev_3w.rename(columns = {'sale_count':'prev_3w_sales_cnt'}, inplace = True)
    article_id_popularity_cnt_prev_3w.drop(columns = ['level_1'], inplace = True)
    article_id_popularity_cnt_prev_3w['prev_3w_sales_cnt'] = article_id_popularity_cnt_prev_3w['prev_3w_sales_cnt'].fillna(value = 0)
    article_id_popularity_cnt_prev_3w.prev_3w_sales_cnt = pd.to_numeric(article_id_popularity_cnt_prev_3w.prev_3w_sales_cnt.astype('int32'), downcast = 'unsigned')


    log.write_log('Count sales of the item in previous 3 week transaction date completed.', log.logging.DEBUG)

    return article_id_popularity_cnt_prev_3w

def calculate_previous_4week_item_sales(tran_data):

    log.write_log('Count sales of the item in previous 4 week transaction date started...', log.logging.DEBUG)

    """
    article_id_popularity_cnt_prev_4w = (tran_data
                                        .groupby(['article_id', 't_dat'])
                                        .agg(sale_count = ('t_dat', 'count'))
                                        .groupby(level = 0, group_keys = False)
                                        .rolling(28, min_periods = 1, closed = 'left')
                                        .sum()
                                        .droplevel(0)
                                        .reset_index()) 
    """
    article_id_popularity_cnt_prev_4w = (tran_data
                                        .groupby('article_id')
                                        ['sale_count']
                                        .rolling(28, min_periods = 1, closed = 'left')
                                        .sum()
                                        .reset_index()
                                        )

    article_id_popularity_cnt_prev_4w.rename(columns = {'sale_count':'prev_4w_sales_cnt'}, inplace = True)
    article_id_popularity_cnt_prev_4w.drop(columns = ['level_1'], inplace = True)
    article_id_popularity_cnt_prev_4w['prev_4w_sales_cnt'] = article_id_popularity_cnt_prev_4w['prev_4w_sales_cnt'].fillna(value = 0)
    article_id_popularity_cnt_prev_4w.prev_4w_sales_cnt = pd.to_numeric(article_id_popularity_cnt_prev_4w.prev_4w_sales_cnt.astype('int32'), downcast = 'unsigned')


    log.write_log('Count sales of the item in previous 4 week transaction date completed.', log.logging.DEBUG)

    return article_id_popularity_cnt_prev_4w

def calculate_previous_5week_item_sales(tran_data):

    log.write_log('Count sales of the item in previous 5 week transaction date started...', log.logging.DEBUG)
    """
    article_id_popularity_cnt_prev_5w = (tran_data
                                        .groupby(['article_id', 't_dat'])
                                        .agg(sale_count = ('t_dat', 'count'))
                                        .groupby(level = 0, group_keys = False)
                                        .rolling(35, min_periods = 1, closed = 'left')
                                        .sum()
                                        .droplevel(0)
                                        .reset_index()) 
    """
    article_id_popularity_cnt_prev_5w = (tran_data
                                        .groupby('article_id')
                                        ['sale_count']
                                        .rolling(35, min_periods = 1, closed = 'left')
                                        .sum()
                                        .reset_index()
                                        ) 

    article_id_popularity_cnt_prev_5w.rename(columns = {'sale_count':'prev_5w_sales_cnt'}, inplace = True)
    article_id_popularity_cnt_prev_5w.drop(columns = ['level_1'], inplace = True)
    article_id_popularity_cnt_prev_5w['prev_5w_sales_cnt'] = article_id_popularity_cnt_prev_5w['prev_5w_sales_cnt'].fillna(value = 0)
    article_id_popularity_cnt_prev_5w.prev_5w_sales_cnt = pd.to_numeric(article_id_popularity_cnt_prev_5w.prev_5w_sales_cnt.astype('int32'), downcast = 'unsigned')

    log.write_log('Count sales of the item in previous 5 week transaction date completed.', log.logging.DEBUG)

    return article_id_popularity_cnt_prev_5w

def calculate_previous_year_item_sales(tran_data):

    log.write_log('Count sales of the item in previous year transaction date started...', log.logging.DEBUG)
    
    article_id_popularity_cnt_prev_year = (tran_data
                                            .groupby('article_id')
                                            ['sale_count']
                                            .rolling(365, min_periods = 1, closed = 'left')
                                            .sum()
                                            .reset_index()
                                    )

    article_id_popularity_cnt_prev_year.rename(columns = {'sale_count':'prev_year_sales_cnt'}, inplace = True)
    article_id_popularity_cnt_prev_year.drop(columns = ['level_1'], inplace = True)
    article_id_popularity_cnt_prev_year['prev_year_sales_cnt'] = article_id_popularity_cnt_prev_year['prev_year_sales_cnt'].fillna(value = 0)
    article_id_popularity_cnt_prev_year.prev_year_sales_cnt = pd.to_numeric(article_id_popularity_cnt_prev_year.prev_year_sales_cnt.astype('int32'), downcast = 'unsigned')

    log.write_log('Count sales of the item in previous year transaction date completed.', log.logging.DEBUG)

    return article_id_popularity_cnt_prev_year

def calculate_item_prev_day_sales(config_path):

    try:
        log.write_log('Calculate item previous day sales count started...', log.logging.DEBUG)

        log.write_log('Loading cleaned train transaction data...', log.logging.DEBUG)
        TRAINING_DATASET_PATH = os.path.join( 
                                                read_yaml_key(config_path,'data_source','data_folders'),
                                                read_yaml_key(config_path,'data_source','processed_data_folder'),
                                                read_yaml_key(config_path,'data_source','train_data'),
                                            ) 
        train_tran = read_from_parquet(TRAINING_DATASET_PATH)

        #Calculate counts for all the items 
        pvt_output = pd.pivot_table(data = train_tran, 
                                    index = ['t_dat'], 
                                    columns = ['article_id'], 
                                    values = 'customer_id',
                                    aggfunc = 'count',
                                    fill_value = 0,
                                    #margins = True, margins_name = 'Total',                               
                                )
        pvt_output = pvt_output.reset_index(level = 0)
        col = pvt_output.columns
        col = col[1:]
        pvt_output = pd.melt(pvt_output, 
                         id_vars = 't_dat', 
                         value_vars = col)
        train_tran = pvt_output
        train_tran.rename(columns = {'value':'sale_count'}, inplace = True)

        """Temp file that only stored last sales of all the item on last training date
        ALL_ITEM_SALES_COUNT = os.path.join( 
                                                read_yaml_key(config_path,'data_source','data_folders'),
                                                read_yaml_key(config_path,'data_source','feature_folder'),
                                                read_yaml_key(config_path,'article_feature','article_folder'),
                                                read_yaml_key(config_path,'article_feature','item_sales_count'),
                                            )
        save_to_parquet(train_tran, ALL_ITEM_SALES_COUNT)

        #save_to_parquet(pvt_output, 'item_sales_cnt_t_date_vise_full.parquet')
        """

        #Consider for only last 8 week records. So for first 1 week record last 5 week will be 8+5 = 13
        #last_tran_date_record = train_tran.t_dat.max()
        #train_tran = train_tran[train_tran.t_dat >= (last_tran_date_record - timedelta(weeks = 13))]

        train_tran.sort_values(['article_id','t_dat'], inplace = True)
        train_tran.reset_index(inplace = True, drop = True)

        article_id_popularity_cnt_prev_day = calculate_previous_day_item_sales(train_tran)
        article_id_popularity_cnt = article_id_popularity_cnt_prev_day

        article_id_popularity_cnt_prev_1w = calculate_previous_1week_item_sales(train_tran)

        log.write_log('Merge the dataframe with main dataframe started...', log.logging.DEBUG)        
        #article_id_popularity_cnt = article_id_popularity_cnt.merge(article_id_popularity_cnt_prev_1w, on = ['article_id', 't_dat'], how = 'inner')
        article_id_popularity_cnt = article_id_popularity_cnt.merge(article_id_popularity_cnt_prev_1w[['prev_1w_sales_cnt']], 
                                                                    left_index = True,
                                                                    right_index = True,
                                                                    how = 'inner'
                                                                    )
        log.write_log('Merge the dataframe with main dataframe completed...', log.logging.DEBUG)

        article_id_popularity_cnt_prev_2w = calculate_previous_2week_item_sales(train_tran)

        log.write_log('Merge the dataframe with main dataframe started...', log.logging.DEBUG)
        #article_id_popularity_cnt = article_id_popularity_cnt.merge(article_id_popularity_cnt_prev_2w, on = ['article_id', 't_dat'], how = 'inner')
        article_id_popularity_cnt = article_id_popularity_cnt.merge(article_id_popularity_cnt_prev_2w[['prev_2w_sales_cnt']], 
                                                            left_index = True,
                                                            right_index = True,
                                                            how = 'inner')
        log.write_log('Merge the dataframe with main dataframe completed...', log.logging.DEBUG)

        
        article_id_popularity_cnt_prev_3w = calculate_previous_3week_item_sales(train_tran)

        log.write_log('Merge the dataframe with main dataframe started...', log.logging.DEBUG)
        #article_id_popularity_cnt = article_id_popularity_cnt.merge(article_id_popularity_cnt_prev_3w, on = ['article_id', 't_dat'], how = 'inner')
        article_id_popularity_cnt = article_id_popularity_cnt.merge(article_id_popularity_cnt_prev_3w[['prev_3w_sales_cnt']], 
                                                                    left_index = True,
                                                                    right_index = True,
                                                                    how = 'inner')
        log.write_log('Merge the dataframe with main dataframe completed...', log.logging.DEBUG)


        article_id_popularity_cnt_prev_4w = calculate_previous_4week_item_sales(train_tran)

        log.write_log('Merge the dataframe with main dataframe started...', log.logging.DEBUG)
        #article_id_popularity_cnt = article_id_popularity_cnt.merge(article_id_popularity_cnt_prev_4w, on = ['article_id', 't_dat'], how = 'inner')
        article_id_popularity_cnt = article_id_popularity_cnt.merge(article_id_popularity_cnt_prev_4w[['prev_4w_sales_cnt']], 
                                                                    left_index = True,
                                                                    right_index = True,
                                                                    how = 'inner')
        log.write_log('Merge the dataframe with main dataframe completed...', log.logging.DEBUG)


        article_id_popularity_cnt_prev_5w = calculate_previous_5week_item_sales(train_tran)
        
        log.write_log('Merge the dataframe with main dataframe started...', log.logging.DEBUG)
        #article_id_popularity_cnt = article_id_popularity_cnt.merge(article_id_popularity_cnt_prev_5w, on = ['article_id', 't_dat'], how = 'inner')
        article_id_popularity_cnt = article_id_popularity_cnt.merge(article_id_popularity_cnt_prev_5w[['prev_5w_sales_cnt']], 
                                                                    left_index = True,
                                                                    right_index = True,
                                                                    how = 'inner')
        log.write_log('Merge the dataframe with main dataframe completed...', log.logging.DEBUG)
        

        article_id_popularity_cnt_prev_year = calculate_previous_year_item_sales(train_tran)

        log.write_log('Merge the dataframe with main dataframe started...', log.logging.DEBUG)
        article_id_popularity_cnt = article_id_popularity_cnt.merge(article_id_popularity_cnt_prev_year[['prev_year_sales_cnt']], 
                                                                    left_index = True,
                                                                    right_index = True,
                                                                    how = 'inner')
        log.write_log('Merge the dataframe with main dataframe completed...', log.logging.DEBUG)

        article_id_popularity_cnt = article_id_popularity_cnt.merge(train_tran[['sale_count', 't_dat']], 
                                                                    left_index = True,
                                                                    right_index = True,
                                                                    how = 'inner')

        article_id_popularity_cnt.t_dat = pd.to_datetime(article_id_popularity_cnt.t_dat, utc = True)
        article_id_popularity_cnt.fillna(value = 0, inplace = True)

        #Set th order of the columns
        article_id_popularity_cnt = article_id_popularity_cnt[['t_dat', 'article_id',
                                                              'sale_count',
                                                              'prev_day_sales_cnt',
                                                              'prev_1w_sales_cnt',
                                                              'prev_2w_sales_cnt',
                                                              'prev_3w_sales_cnt',
                                                              'prev_4w_sales_cnt',
                                                              'prev_5w_sales_cnt',
                                                              'prev_year_sales_cnt'
                                                            ]]

        #Change the datatype to INT32/FLOAT64. Read FEASTReadme.txt to understand
        """       
        article_id_popularity_cnt.prev_day_sales_cnt = pd.to_numeric(article_id_popularity_cnt.prev_day_sales_cnt .astype('int32'))
        article_id_popularity_cnt.prev_1w_sales_cnt = pd.to_numeric(article_id_popularity_cnt.prev_1w_sales_cnt .astype('int32'))
        article_id_popularity_cnt.prev_2w_sales_cnt = pd.to_numeric(article_id_popularity_cnt.prev_2w_sales_cnt .astype('int32'))
        article_id_popularity_cnt.prev_3w_sales_cnt = pd.to_numeric(article_id_popularity_cnt.prev_3w_sales_cnt .astype('int32'))
        article_id_popularity_cnt.prev_4w_sales_cnt = pd.to_numeric(article_id_popularity_cnt.prev_4w_sales_cnt .astype('int32'))
        article_id_popularity_cnt.prev_5w_sales_cnt = pd.to_numeric(article_id_popularity_cnt.prev_5w_sales_cnt .astype('int32'))
        """

        log.write_log('Save the gernerated features...', log.logging.DEBUG)
        ITEM_PREVIOUS_DAYS_SALES_COUNT = os.path.join( 
                                                        read_yaml_key(config_path,'data_source','data_folders'),
                                                        read_yaml_key(config_path,'data_source','feature_folder'),
                                                        read_yaml_key(config_path,'article_feature','article_folder'),
                                                        read_yaml_key(config_path,'article_feature','item_prev_days_sales_count'),
                                                    )
        save_to_parquet(article_id_popularity_cnt, ITEM_PREVIOUS_DAYS_SALES_COUNT)


        del [article_id_popularity_cnt, 
             article_id_popularity_cnt_prev_year,
             article_id_popularity_cnt_prev_day,
             article_id_popularity_cnt_prev_5w, 
             article_id_popularity_cnt_prev_4w, 
             article_id_popularity_cnt_prev_3w, 
             article_id_popularity_cnt_prev_2w, 
             article_id_popularity_cnt_prev_1w,
             ]

        gc.collect()

        log.write_log('Calculate item previous day sales count completed.', log.logging.DEBUG)

        return

    except Exception as e:

        raise RecommendationException(e, sys)