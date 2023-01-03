from utils.read_utils import read_from_parquet, read_yaml_key
from sklearn.base import BaseEstimator, TransformerMixin
from utils.exception import RecommendationException
from config.config import CONFIGURATION_PATH
import logs.logger as log 
#import cudf
import sys
import os


class transform_customer_mapping(BaseEstimator, TransformerMixin):


    def __init__(self, hash_conversion = False , config_path = CONFIGURATION_PATH) -> None:
       
        super().__init__()

        self.CUSTOMER_USER_MAPPING = os.path.join( 
                                                    read_yaml_key(config_path,'data_source', 'data_folders'),
                                                    read_yaml_key(config_path,'data_source', 'processed_data_folder'),                                               
                                                    read_yaml_key(config_path,'data_source', 'customer_user_mapping_data'),
                                                    )
        self.hash_conversion = hash_conversion
      


    def fit(self, X, Y = None):

        self.df_customer_user_mapping = self.read_mapping_data()
        log.write_log('Read the customer to user mapping from the file completed...', log.logging.DEBUG)

        return self

    def transform(self, X = None):

        try:
            
            #Step 1: convert the customer_id to hash equivalent
            #We did it as it was original customer_id was string and required more space. By transfomr to hash we change it to int, which in turn required less space.
            if self.hash_conversion == True:
                log.write_log('Convert customer_id to hash...', log.logging.DEBUG)
                X['customer_id'] = X['customer_id'].apply(lambda x: int(x[-16:],16) ).astype('int64')
        
            #Step 2: mapping customer_id to user_id
            log.write_log('Transform mapping customer to user started...', log.logging.DEBUG)
            X = X.merge(self.df_customer_user_mapping, on = ['customer_id'], how = 'inner')
            log.write_log('Transform mapping customer to user completed...', log.logging.DEBUG)

            return X
        
        except Exception as e:
            raise RecommendationException(e, sys) from e  


    def read_mapping_data(self):

        #In case we want to replace file storage with database this is the only place we would need to changes

        log.write_log('Read the customer to user mapping from the file started...', log.logging.DEBUG)
        return read_from_parquet(self.CUSTOMER_USER_MAPPING)  