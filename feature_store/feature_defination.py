from utils.read_utils import  read_yaml_key
from feast.types import PrimitiveFeastType
from datetime import timedelta
from feast import FeatureStore
from feast import FeatureView
from feast import FileSource
from feast import Field
from feast import Entity
import logs.logger as log
import argparse
import os

if __name__ == '__main__': 

    args = argparse.ArgumentParser()
    args.add_argument("--config", default = "config/config.yaml")
    parsed_args = args.parse_args()

    config_path = parsed_args.config

    log.write_log(f'Read configuration from path: {parsed_args.config}', log.logging.INFO)

    repo_path =  os.path.join( 
                                read_yaml_key(config_path, 'feature_store', 'feature_store_folder'),
                            )

    fs = FeatureStore(repo_path = repo_path) 

    log.write_log(f'Registering features in feature store ', log.logging.INFO)

    

    #************************* Define an entity for the USER FEATURE *************************
    user_entity = Entity(name = "User", 
                         join_keys = ["customer_id"], #"join_keys" parm for FEAST 0.23.0
                         description = "Customer/User related features",                
    )

    #************************* Define Feature Views for each features of the users *************************
    

    ############### Feature View:  User average median purchase price ###############  

    USER_AVG_MEDIAN_PURCHASE_PRICE = os.path.join( 
                                                read_yaml_key(config_path,'data_source','data_folders'),
                                                read_yaml_key(config_path,'data_source','feature_folder'),
                                                read_yaml_key(config_path,'customer_features','customer_folder'),
                                                read_yaml_key(config_path,'customer_features','user_avg_median_purchase_price'),
                                                )

    user_avg_median_purchase_price_source = FileSource(
                                                        path = USER_AVG_MEDIAN_PURCHASE_PRICE, 
                                                        timestamp_field = "t_dat",                                
                                                     )

    user_avg_median_purchase_price_fv = FeatureView(
                                                        name = "user_avg_median_purchase_price_fv",
                                                        ttl = timedelta(hours=5,minutes=30),#timedelta(seconds=86400 * 1), 
                                                        entities = [user_entity],
                                                        schema = [                              
                                                                    Field(name = "user_mean_purchase_price", dtype = PrimitiveFeastType.FLOAT32),
                                                                    Field(name = "user_median_purchase_price", dtype = PrimitiveFeastType.FLOAT32), 
                                                                    Field(name = "user_prev_mean_purchase_price", dtype =PrimitiveFeastType.FLOAT32),     
                                                                    Field(name = "user_prev_median_purchase_price", dtype = PrimitiveFeastType.FLOAT32), 
                                                                ],    
                                                        source = user_avg_median_purchase_price_source
                                                    )                                        

    ############### Feature View:  User Avg Median Purchase Price Last 8 Week ###############

    USER_AVG_MEDIAN_PURCHASE_PRICE_LAST_8WEEK = os.path.join( 
                                                            read_yaml_key(config_path,'data_source','data_folders'),
                                                            read_yaml_key(config_path,'data_source','feature_folder'),
                                                            read_yaml_key(config_path,'customer_features','customer_folder'),
                                                            read_yaml_key(config_path,'customer_features','user_avg_median_purchase_price_last_8week'),
                                                            )

    user_avg_median_purchase_price_last_8week_source = FileSource(
                                                                path = USER_AVG_MEDIAN_PURCHASE_PRICE_LAST_8WEEK, 
                                                                timestamp_field = "t_dat",                                
                                                                )

    user_avg_median_purchase_price_last_8week_fv = FeatureView(
                                                                name = "user_avg_median_purchase_price_last_8week_fv",
                                                                ttl = timedelta(hours=5,minutes=30), #timedelta(seconds=86400 * 1), 
                                                                entities = [user_entity],
                                                                schema = [                              
                                                                            Field(name = "user_mean_purchase_price", dtype = PrimitiveFeastType.FLOAT32),
                                                                            Field(name = "user_median_purchase_price", dtype = PrimitiveFeastType.FLOAT32), 
                                                                            Field(name = "user_prev_mean_purchase_price", dtype = PrimitiveFeastType.FLOAT32),     
                                                                            Field(name = "user_prev_median_purchase_price", dtype = PrimitiveFeastType.FLOAT32),                                                                
                                                                        ],    
                                                                source = user_avg_median_purchase_price_last_8week_source
                                                            )                                                           

    #*************************************************************************************************************************************************


    #************************* Define an entity for the ITEM FEATURE *************************
    item_entity = Entity(name = "Item", 
                        join_keys = ["article_id"], #"join_keys" parm for FEAST 0.23.0
                        description = "Article/Items related features",                
    )

    #************************* Define Feature Views for each features of the items *************************

    
    
    ############### Feature View: Item average median sale price ############### 

    ITEM_AVG_SALES_PRICE = os.path.join( 
                                            read_yaml_key(config_path,'data_source','data_folders'),
                                            read_yaml_key(config_path,'data_source','feature_folder'),
                                            read_yaml_key(config_path,'article_feature','article_folder'),
                                            read_yaml_key(config_path,'article_feature','item_avg_median_sales_price'),
                                        )

    item_avg_sales_price_source = FileSource(
                                            path = ITEM_AVG_SALES_PRICE, 
                                            timestamp_field = "t_dat",                                
                                            )

    item_avg_sales_price_fv = FeatureView(
                                            name = "item_avg_sales_price_fv",
                                            ttl = timedelta(hours=5,minutes=30), #timedelta(seconds=86400 * 1), 
                                            entities = [item_entity],
                                            schema = 
                                            [ 
                                                Field(name = 'item_mean_sales_price', dtype =  PrimitiveFeastType.FLOAT32),
                                                Field(name = 'item_median_sales_price', dtype =  PrimitiveFeastType.FLOAT32),                                                 
                                                Field(name = 'item_prev_mean_sales_price', dtype =  PrimitiveFeastType.FLOAT32), 
                                                Field(name = 'item_prev_median_sales_price', dtype =  PrimitiveFeastType.FLOAT32),
                                            ],    
                                            source = item_avg_sales_price_source
                                        )                                        


    ############### Feature View: Item previous days sales counts ############### 

    ITEM_PREVIOUS_DAYS_SALES_COUNT = os.path.join( 
                                                    read_yaml_key(config_path,'data_source','data_folders'),
                                                    read_yaml_key(config_path,'data_source','feature_folder'),
                                                    read_yaml_key(config_path,'article_feature','article_folder'),
                                                    read_yaml_key(config_path,'article_feature','item_prev_days_sales_count'),
                                                )

    item_previous_days_sales_count_source = FileSource(
                                                        path = ITEM_PREVIOUS_DAYS_SALES_COUNT, 
                                                        timestamp_field = "t_dat",                                
                                                    ) 

    item_previous_days_sales_count_fv = FeatureView(
                                                    name = "item_previous_days_sales_count_fv",
                                                    ttl = timedelta(hours=5,minutes=30), #timedelta(seconds=86400 * 1), 
                                                    entities = [item_entity],
                                                    schema = 
                                                    [                    
                                                        Field(name = 'sale_count', dtype = PrimitiveFeastType.INT32),                                 
                                                        Field(name = 'prev_day_sales_cnt', dtype = PrimitiveFeastType.INT32), 
                                                        Field(name = 'prev_1w_sales_cnt', dtype =  PrimitiveFeastType.INT32), 
                                                        Field(name = 'prev_2w_sales_cnt', dtype =  PrimitiveFeastType.INT32), 
                                                        Field(name = 'prev_3w_sales_cnt', dtype =  PrimitiveFeastType.INT32), 
                                                        Field(name = 'prev_4w_sales_cnt', dtype =  PrimitiveFeastType.INT32), 
                                                        Field(name = 'prev_5w_sales_cnt', dtype =  PrimitiveFeastType.INT32),
                                                        Field(name = 'prev_year_sales_cnt', dtype =  PrimitiveFeastType.INT32),
                                                    ],    
                                                    source = item_previous_days_sales_count_source
                                        )                                           

  



    #************************* Apply the feature_view  *************************
    # register entity and feature view
    log.write_log(f'Applying features views in feature store...', log.logging.INFO)
    fs.apply([
                user_entity ,  item_entity,                
                user_avg_median_purchase_price_last_8week_fv, user_avg_median_purchase_price_fv,
                item_avg_sales_price_fv, item_previous_days_sales_count_fv,
            ]
            ) 

    log.write_log(f'Applying features views in feature completed', log.logging.INFO)
















