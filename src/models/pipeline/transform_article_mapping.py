
from utils.read_utils import read_from_parquet, read_yaml_key
from sklearn.base import BaseEstimator, TransformerMixin
from utils.exception import RecommendationException
from config.config import CONFIGURATION_PATH
import logs.logger as log 
#import cudf
import sys
import os

class transform_article_mapping(BaseEstimator, TransformerMixin):


    def __init__(self, config_path = CONFIGURATION_PATH) -> None:

        super().__init__()

        self.ARTICLE_ITEM_MAPPING = os.path.join( 
                                        read_yaml_key(config_path,'data_source','data_folders'),
                                        read_yaml_key(config_path,'data_source','processed_data_folder'),                               
                                        read_yaml_key(config_path,'data_source','article_item_mapping_data'),
                                )

        
    def fit(self, X, Y = None):        
        
        self.df_article_item_mapping = self.read_mapping_data()
        log.write_log('Read the article to item mapping from the file completed...', log.logging.DEBUG)

        return self
    
    def transform(self, X = None):
            
        try:

            log.write_log('Transform mapping article to item...', log.logging.DEBUG)
            X = X.merge(self.df_article_item_mapping, on = ['article_id'], how = 'inner')

            return X
        
        except Exception as e:
            raise RecommendationException(e, sys) from e 

    def read_mapping_data(self):
        
        #In case we want to replace file storage with database this is the only place we would need to changes

        log.write_log('Read the article to item mapping from the file started...', log.logging.DEBUG)
        return read_from_parquet(self.ARTICLE_ITEM_MAPPING)