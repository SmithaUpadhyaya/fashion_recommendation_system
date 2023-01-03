from sklearn.base import BaseEstimator, TransformerMixin
from utils.exception import RecommendationException
import logs.logger as log
import pandas as pd
#import cudf
import sys

class bin_feature(BaseEstimator, TransformerMixin):
    
    """
    Class will bin the feature based on the quartile values
    """
    
    def __init__(self, quartile_feature_dict) -> None:
        
        super().__init__()

        self.quartile_feature_dict = quartile_feature_dict
        self.feature_quartile_output = {}
        
       
    def fit(self, X, Y = None):
        
        try:

            log.write_log(f'Fit bin features started...', log.logging.INFO)

            for key in self.quartile_feature_dict.keys():
                
                bin_col = key  
                log.write_log(f'Calculating quartile for feature "{bin_col}"...', log.logging.DEBUG)

                self.calculate_quartile(bin_col, X)

        except Exception as e:
            raise RecommendationException(e, sys) from e  
        
        return self
    
    def transform(self, X = None):
        
        try:

            log.write_log(f'Transform features...', log.logging.INFO)

            #binned_feature = cudf.DataFrame() #Code when use cudf dataframe
            binned_feature = pd.DataFrame()  #Code when use pandas dataframe

            for i, key in enumerate(self.quartile_feature_dict.keys()):
                
                bin_col = key    
                log.write_log(f'Map quartile to bin for calculated for feature "{bin_col}"...', log.logging.DEBUG)

                if i == 0:
                    binned_feature = self.map_quartile_to_bin(bin_col, X)
                else:
                    #binned_feature = cudf.concat([binned_feature, self.map_quartile_to_bin(bin_col, X)], axis = 1) #Code when use cudf dataframe
                    binned_feature = pd.concat([binned_feature, self.map_quartile_to_bin(bin_col, X)], axis = 1)  #Code when use pandas dataframe
            
            #X = pd.concat([X, binned_feature], axis = 1) #Code when use pandas dataframe
            #X = cudf.concat([X, binned_feature], axis = 1) #Code when use cudf dataframe 
            
            return X #binned_feature

        except Exception as e:
            raise RecommendationException(e, sys) from e 
    
    def calculate_quartile(self, bin_col, X):
    
        try:

            quartile_list = self.quartile_feature_dict[bin_col]
            quartile = X[bin_col].quantile(quartile_list)
            quartile = quartile.reset_index()       

            log.write_log(f'Stored the calculated quartile for "{bin_col}" to dict...', log.logging.DEBUG)
            self.feature_quartile_output[bin_col] = quartile
        
        except Exception as e:
            raise RecommendationException(e, sys) from e

        return
    
    def map_quartile_to_bin(self, bin_col, X):

        try:  

            X[bin_col + '_bin'] = -1
            
            log.write_log(f'Get list of quartile bin features for dictonary...', log.logging.DEBUG)
            quartile_list = self.quartile_feature_dict[bin_col]

            quartile_value = self.feature_quartile_output[bin_col]
            quartile_value = quartile_value.set_index('index')

            for i, q_value in enumerate(quartile_list):
                
                log.write_log(f'Bin for quartile {q_value} ...', log.logging.DEBUG)
                q_cur = quartile_value.loc[q_value, bin_col]
                
                if i == 0:
                    
                    X.loc[X[bin_col] <= q_cur, bin_col + '_bin'] = i
                    
                else:
                    
                    q_prev = quartile_value.loc[quartile_list[i-1],bin_col]
                    
                    X.loc[(X[bin_col] > q_prev) & 
                                (X[bin_col] <= q_cur), bin_col + '_bin'] = i

            X.loc[X[bin_col] > q_cur, bin_col + '_bin'] = i + 1

        except Exception as e:
            raise RecommendationException(e, sys) from e
        
        return X[bin_col + '_bin']

#####################################################################################

#Sample Code

#bin_obj = bin_feature({
#                        'color': [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
#                        'avg_elapse_days_per_tran': [0.25, 0.5, 0.75, 0.9],
#                        'days_pass_since_last_purchase': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 ,0.9],
#                      })
#bin_obj.fit(df_train)
#bin_obj.transform(df_train)  