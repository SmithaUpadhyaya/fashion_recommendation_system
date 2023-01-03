from sklearn.base import BaseEstimator, TransformerMixin
from utils.exception import RecommendationException
import logs.logger as log 
import pandas as pd
#import cudf
import sys

class bin_feature_based_on_feature(BaseEstimator, TransformerMixin):
    
    """
    Class will bin the feature using other feature
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
                log.write_log(f'Get the calculated quartile feature for "{bin_col}"...', log.logging.DEBUG) 

                groupby_col = self.quartile_feature_dict[bin_col]['groupby_col']
                quartile = self.feature_quartile_output[bin_col]

                log.write_log(f'Merge quartile bin to main dataframe "{bin_col}"...', log.logging.DEBUG)
                X = X.merge(quartile, how ='left', on = groupby_col)
                
                log.write_log(f'Map quartile to bin for calculated for feature "{bin_col}"...', log.logging.DEBUG)
                if i == 0:
                    binned_feature = self.map_quartile_to_bin(bin_col, X)
                else:
                    #binned_feature = cudf.concat([binned_feature, self.map_quartile_to_bin(bin_col, X)], axis = 1) #Code when use cudf dataframe
                    binned_feature = pd.concat([binned_feature, self.map_quartile_to_bin(bin_col, X)], axis = 1) #Code when use pandas dataframe
                
                #X = pd.concat([X, binned_feature], axis = 1) #Code when use pandas dataframe
                #X = cudf.concat([X, binned_feature], axis = 1) #Code when use cudf dataframe 
            
            log.write_log(f'Drop quartile bin features...', log.logging.DEBUG)
            X = self.drop_feature_quartile(X) 
        
        except Exception as e:
            raise RecommendationException(e, sys) from e  

        return X #binned_feature
           
    """
    e.g Bin the previous sales count of the item based on the prodcut type using quantile. 
    Since each product type have different distrubution. 
    """
    def calculate_quartile(self, bin_col, X):
    
        try:
            
            groupby_col = self.quartile_feature_dict[bin_col]['groupby_col']
            quartile_list = self.quartile_feature_dict[bin_col]['quartile_list']
                
            #quartile = cudf.DataFrame() #Code when use cudf dataframe
            quartile = pd.DataFrame() #Code when use pandas dataframe
            
            for i, q_value in enumerate(quartile_list):

                log.write_log(f'Calculate quartile: {q_value} for "{bin_col}" grouped by col: "{groupby_col}"...', log.logging.DEBUG)

                q = X[[groupby_col, bin_col]].groupby([groupby_col]).quantile(q_value)
                q = q.reset_index()
                q.columns = [groupby_col, bin_col + str(q_value)]

                #bin_col_list = bin_col_list + [bin_col + str(q_value)]
                #X = X.merge(q, how ='left', on = groupby_col)

                log.write_log(f'Merge calculated quartile to main dataset...', log.logging.DEBUG)
                if i == 0:
                    quartile = q
                else:
                    #quartile = cudf.merge(quartile, q, how ='left', on = groupby_col) #Code when use cudf dataframe
                    quartile = pd.merge(quartile, q, how ='left', on = groupby_col) #Code when use pandas dataframe
            
            log.write_log(f'Stored the calculated quartile for "{bin_col}" to dict...', log.logging.DEBUG)      
            self.feature_quartile_output[bin_col] = quartile
        
        except Exception as e:
            raise RecommendationException(e, sys) from e  

        return

    
    def map_quartile_to_bin(self, bin_col, X):
            
        try:

            X[bin_col + '_bin'] = -1

            log.write_log(f'Get list of quartile bin features for dictonary...', log.logging.DEBUG)
            quartile_list = self.quartile_feature_dict[bin_col]['quartile_list']
            
            for i, q_value in enumerate(quartile_list):

                log.write_log(f'Bin for quartile {q_value} ...', log.logging.DEBUG)
                
                if i == 0:
                    X.loc[X[bin_col] <= X[bin_col + str(q_value)], bin_col + '_bin'] = i
                else:
                    X.loc[(X[bin_col] > X[bin_col + str(quartile_list[i-1])]) & 
                                (X[bin_col] <= X[bin_col + str(q_value)]), bin_col + '_bin'] = i

            X.loc[X[bin_col] > X[bin_col + str(q_value)], bin_col + '_bin'] = i + 1
        
        except Exception as e:
            raise RecommendationException(e, sys) from e  

        return X[bin_col + '_bin']
        
    def drop_feature_quartile(self, X):
        
        try:

            for key in self.quartile_feature_dict.keys():
                
                bin_col = key
                log.write_log(f'Drop quartile bin features for "{bin_col}"...', log.logging.DEBUG)
            
                quartile_list = self.quartile_feature_dict[key]['quartile_list']
                X = X.drop(columns = [bin_col + str(x) for x in quartile_list])

        except Exception as e:
            raise RecommendationException(e, sys) from e  

        return X


#####################################################################################

#Sample Code

#q_list = [0.1, 0.25, 0.5, 0.75, 0.9]
#quartile_features = { 
#                          'item_median_sales_price_for_product_type': {'groupby_col': 'product_desc', 'quartile_list': q_list},
#                          'user_last8week_median_sale_price': {'groupby_col': 'product_desc', 'quartile_list': q_list},
#                          'item_median_sale_price': {'groupby_col': 'product_desc', 'quartile_list': q_list},
#                          'prev_day_sales_cnt': {'groupby_col': 'product_desc', 'quartile_list': q_list},
#                          'prev_1w_sales_cnt': {'groupby_col': 'product_desc', 'quartile_list': q_list},
#                          'prev_2w_sales_cnt': {'groupby_col': 'product_desc', 'quartile_list': q_list},
#                          'prev_3w_sales_cnt': {'groupby_col': 'product_desc', 'quartile_list': q_list},
#                          'prev_4w_sales_cnt': {'groupby_col': 'product_desc', 'quartile_list': q_list},
#                        }

#bin_quartile_obj = bin_feature_based_on_feature(quartile_features)

#bin_quartile_obj.fit(df_train)
#bin_quartile_obj.transform(df_train)

