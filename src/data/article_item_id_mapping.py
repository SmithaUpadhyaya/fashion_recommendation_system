from utils.read_utils import  read_from_parquet, read_yaml_key, read_from_pickle
from utils.exception import RecommendationException
from utils.write_utils import save_to_parquet
import logs.logger as log 
import sys
import os
import gc

def map_article_to_item_id(config_path):

    try:   
        """
        create mapping of unique article to item id
        This will also help when we need to transform article to interger for train/predict model
        """
        CLEAN_ARTICLES_MASTER_TABLE = os.path.join( 
                                                   read_yaml_key(config_path,'data_source','data_folders'),
                                                   read_yaml_key(config_path,'data_source','processed_data_folder'), 
                                                   read_yaml_key(config_path,'data_source','clean_article_data'),
                                                )

        df_article = read_from_parquet(CLEAN_ARTICLES_MASTER_TABLE)

        log.write_log('Create Article to unique item id mapping...', log.logging.DEBUG)

        #Old code when we had created mapping pkl file that store article to item mapping
        """
        article_mapping_path = os.path.join( 
                                            read_yaml_key(config_path, 'data_source', 'data_folders'),
                                            read_yaml_key(config_path, 'data_source', 'interim_data_folder'),
                                            read_yaml_key(config_path, 'article_map', 'articleid_to_itemid')
                                            )
        article_to_item_mapping = read_from_pickle(article_mapping_path, compression = '')
        df_article['item_id'] = df_article['article_id'].map(article_to_item_mapping)
        df_article['item_id_aug'] = df_article['item_id']
        df_article['item_id_aug'] = range(len(article_to_item_mapping.keys()) + 1, df_article.shape[0] + len(article_to_item_mapping.keys()) + 1, 1)
        df_article['item_id'] = df_article['item_id'].fillna(df_article.pop('item_id_aug'))
        df_article['item_id'] = df_article['item_id'].astype(int)
        """
        df_article['item_id'] = range(0, df_article.item_id.nunique())

        #df_article = df_article[['article_id', 'item_id', 'event_timestamp']]
        df_article = df_article[['article_id', 'item_id']]
        
        log.write_log('Saving mapping to file started...', log.logging.DEBUG)

        ARTICLE_ITEM_MAPPING = os.path.join( 
                                                read_yaml_key(config_path,'data_source','data_folders'),
                                                read_yaml_key(config_path,'data_source','processed_data_folder'),                               
                                                read_yaml_key(config_path,'data_source','article_item_mapping_data'),
                                            )
        
        save_to_parquet(df_article, ARTICLE_ITEM_MAPPING)

        log.write_log('Saving mapping for artcle to unique interger item id completed.', log.logging.DEBUG)

        del [df_article]
        gc.collect()

        return

    except Exception as e:

        raise RecommendationException(e, sys) 