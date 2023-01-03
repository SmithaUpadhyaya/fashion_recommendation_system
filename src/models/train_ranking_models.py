from utils.read_utils import  read_yaml_key, read_from_parquet
from src.models.ranking_model import ranking_model
from datetime import datetime, timedelta, timezone
import logs.logger as log
import argparse
import os

################### Train the model ##############################
if __name__ == '__main__':  
    
    args = argparse.ArgumentParser()
    args.add_argument("--config", default = "config/config.yaml")
    parsed_args = args.parse_args()

    log.write_log(f'Read configuration from path: {parsed_args.config}', log.logging.INFO)
    config_path = parsed_args.config
    #config_path = "config/config.yaml"

    TRAIN_DATA = os.path.join(                         
                            read_yaml_key(config_path,'data_source','data_folders'),
                            read_yaml_key(config_path,'data_source','processed_data_folder'),
                            read_yaml_key(config_path,'data_source','training_data'),
                            )

    X = read_from_parquet(TRAIN_DATA)

    #Use last 4Week of data
    last_tran_date_record = X.t_dat.max()
    split_t_dat = (last_tran_date_record - timedelta(weeks = 4))
    X = X[(X.t_dat >= split_t_dat)]

    saved_model = os.path.join(read_yaml_key(config_path,'model','output_folder'),read_yaml_key(config_path,'lightgbm-param','ranking-model-output-folder'),read_yaml_key(config_path,'lightgbm-param','saved_model'))
    saved_pipeline = os.path.join(read_yaml_key(config_path,'model','output_folder'),read_yaml_key(config_path,'lightgbm-param','ranking-model-output-folder'),read_yaml_key(config_path,'lightgbm-param','saved_engg_pipeline'))
    model_obj = ranking_model(saved_model, saved_pipeline, config_path)

    model_obj.train_model(X)
    
    TEST_DATASET = os.path.join(read_yaml_key(config_path,'data_source','data_folders'), read_yaml_key(config_path,'data_source','processed_data_folder'),read_yaml_key(config_path,'data_source','testing_data'))
    X_test = read_from_parquet(TEST_DATASET)    
    #is_training = True #Change this to False when go live
    #X_test = model_obj.get_features(X_test, is_training) 
    rank = model_obj.predict(X_test)

    print(rank)


############################################################

#model_obj.init_repo_path(is_training)
#model_obj.init_repo_path(True)
#materialize_in: Consider last end date as start date.
#start = datetime(year = 2020, month = 9, day = 21, hour = 0, minute = 0, second = 0)
#end = datetime(year = 2020, month = 9, day = 21, hour = 23, minute = 59, second = 59)
#model_obj.fs.materialize(start_date = start, end_date = end )

#end = datetime(year = 2020, month = 9, day = 22, hour = 23, minute = 59, second = 59)
#model_obj.fs.materialize_incremental(end_date = end)
#Things tried for materilization 

"""
import pytz
from pytz import timezone

#last_record_date = datetime(year = 2020, month = 9,day = 22, tzinfo = timezone.utc) + timedelta(seconds=0, days=0, milliseconds=0,minutes=0)
last_record_date = datetime(year = 2020, month = 9,day = 22, hour = 0, minute = 0, second = 0, microsecond = 0).strftime("%Y-%m-%dT%H:%M:%S")

start = datetime(year = 2020, month = 9, day = 19, hour = 0, minute = 0, second = 0)#.isoformat()
end = datetime(year = 2020, month = 9, day = 20, hour = 23, minute = 59, second = 59)#.isoformat()

start = datetime(year = 2020, month = 9, day = 20, hour = 0, minute = 0, second = 0)#, tzinfo = pytz.utc)#timezone('Asia/Kolkata'))#.strftime("%Y-%m-%dT%H:%M:%S")
end = datetime(year = 2020, month = 9, day = 22, hour = 23, minute = 59, second = 59)#, tzinfo = pytz.utc)#.strftime("%Y-%m-%dT%H:%M:%S")

model_obj.fs.materialize_incremental(end_date = end)

feature_views = ['customer_elapsed_day_fv','item_previous_days_sales_count_fv', 'item_avg_sales_price_fv' ]

model_obj.fs.materialize(start_date = start, end_date = end )#, feature_views=feature_views)
model_obj.fs.materialize(start_date = start, 
                        end_date = end,
                        feature_views = ['user_avg_median_purchase_price_last_8week_fv', 'customer_elapsed_day_fv', 'user_avg_median_purchase_price_fv','item_previous_days_sales_count_fv', 'item_avg_sales_price_fv' ]
                        )
model_obj.fs.materialize_incremental(end_date = datetime(year = 2020, month = 9,day = 22),
                            feature_views = ['user_avg_median_purchase_price_last_8week_fv', 'customer_elapsed_day_fv', 'user_avg_median_purchase_price_fv','item_previous_days_sales_count_fv', 'item_avg_sales_price_fv' ]
                        )
model_obj.fs.materialize_incremental(end_date = datetime.utcnow(),
                            feature_views = ['user_avg_median_purchase_price_last_8week_fv', 'customer_elapsed_day_fv', 'user_avg_median_purchase_price_fv','item_previous_days_sales_count_fv', 'item_avg_sales_price_fv' ]
                        )


X_test = model_obj.get_features(X_test, False)  
X_Item_Feat = model_obj.get_online_item_features(X_test.head(100))
dt_items = X_test[['article_id']].drop_duplicates().reset_index(drop = True)
X_Item_Feat = model_obj.fs.get_online_features(  entity_rows = dt_items.to_dict(orient = 'records'), features = model_obj.item_features_list ).to_df()

X_test2 = model_obj.get_items_features(X_test.head(100), is_training = False)
X_test2 = model_obj.get_items_features(X_test, is_training = False)
X_test= model_obj.get_user_features(model_obj, is_training = False)
###################################################################

# config_path = r"F:\github_workspace\recommendation\config\config.yaml"
#config_path = "config/config.yaml"
#X = read_from_parquet('data\\feature_eng_dataset.parquet')    
#feature_engg = read_object('models\\ranking_model\\feature_engg_pipeline.json')
#X = feature_engg.transform(X.head(3))

import sqlite3
import pandas as pd
con_online = sqlite3.connect("data\\features\\online_store.db")
print("\n--- Schema of online store ---")
#"SELECT * FROM feature_repo_driver_hourly_stats"
print(
    pd.read_sql_query(
        "SELECT name FROM sqlite_schema WHERE type ='table' AND name NOT LIKE 'sqlite_%';"        
        , con_online))
pd.read_sql_query(
        "SELECT * FROM feature_repo_item_avg_sales_price_fv"
        , con_online)

con_online.close()
"""