from sklearn.base import BaseEstimator, TransformerMixin
from utils.exception import RecommendationException
import logs.logger as log 
import sys
import re

class merge_catagorical_feature(BaseEstimator, TransformerMixin):

    """
    Class will merge catagorical feature 
    """
    
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, Y = None):
        
        return self
    
    def transform(self, X = None):
            
        try:

            X = self.generate_product_desc(X)
            
            log.write_log(f'Transform by merge catagorical feature started...', log.logging.INFO)         

            X['product_desc'] = (X['garment_group_name'].astype(str) + '_' + X['product_type_name'].astype(str))
            X['color'] = X['Red'] + X['Blue'] + X['Green']

        except Exception as e:
            raise RecommendationException(e, sys) from e  

        return X

    def generate_product_desc(self, X):

        try:

            log.write_log(f'Transform product catagorical feature by merging them into as "product_desc" started...', log.logging.INFO)

            list_garment_group_name = []
            for (garment_group_name, index_name, index_grp, department_name) in zip(X['garment_group_name'], X['index_name'], X['index_group_name'], X['department_name']):

                #Add garment_group_name
                merge_str = garment_group_name

                #Merger index_grp
                pattern = index_grp.split(' ')
                for str in pattern:
                    if re.search(str, merge_str) == None:
                        merge_str += " "+ str


                #Merge index_name
                pattern = index_name.split(' ')
                for str in pattern:
                    if re.search(str, merge_str) == None:
                        merge_str += " "+ str

                #Merge department_name
                pattern = department_name.split(' ')
                for str in pattern:
                    if re.search(str, merge_str) == None:
                        merge_str += " "+ str

                list_garment_group_name.append(merge_str)

            X['garment_group_name'] = list_garment_group_name
            #X['product_group_name'] = X['clean_product_group_name']
            #X['product_type_name'] = X['new_product_type_name']

            return X
        except Exception as e:
            raise RecommendationException(e, sys) from e


#####################################################################################

#Sample code 

#merge_catagorical_feature_obj = merge_catagorical_feature()
#df_train = merge_catagorical_feature_obj.transform(df_train)
#df_test = merge_catagorical_feature_obj.transform(df_test)
