from sklearn.base import BaseEstimator, TransformerMixin
from utils.exception import RecommendationException
import category_encoders as ce
import logs.logger as log 
import pandas as pd
#import cudf
import sys

class catagory_leave_one_out_encoder(BaseEstimator, TransformerMixin):

    """
    Class will apply target encoding on catagorical feature
    """

    def __init__(self, label_column_name, columns_list, seed) -> None:

        super().__init__()

        self.columns_list = columns_list
        self.label_column_name = label_column_name
        self.seed = seed
        self.target_ce = ce.leave_one_out.LeaveOneOutEncoder(verbose = 1, return_df = True, random_state = self.seed)
        

    def fit(self, X, Y = None):        
        
        try:
            
            log.write_log(f'Fit catagory leave one out encoder features started...', log.logging.DEBUG)
            log.write_log(f'Features to fit {self.columns_list}...', log.logging.DEBUG)

            #self.target_ce.fit(X[self.columns_list].to_pandas(), X[self.label_column_name].to_pandas()) #Code when use cudf dataframe
            self.target_ce.fit(X[self.columns_list], X[self.label_column_name]) #Code when use pandas dataframe

        except Exception as e:
            raise RecommendationException(e, sys) from e  

        return self


    def transform(self, X, Y = None):

        try:

            log.write_log(f'Transform catagory leave one out encoder features started...', log.logging.DEBUG)
            log.write_log(f'Features to transform {self.columns_list}...', log.logging.DEBUG)

            #target_catagory_output = self.target_ce.transform(X[self.columns_list].to_pandas()) #Code when use cudf dataframe. , X[self.label_column_name].to_pandas()
            target_catagory_output = self.target_ce.transform(X[self.columns_list])  #Code when use pandas dataframe. , X[self.label_column_name]   
            
            target_catagory_output[self.columns_list] = target_catagory_output[self.columns_list].astype('float16') 
            #target_catagory_output = cudf.DataFrame.from_pandas(target_catagory_output) #Code when use cudf dataframe
            
            log.write_log(f'Merge transformed feature to main dataset...', log.logging.DEBUG)
            target_catagory_output.columns = [x + '_tce' for x in self.columns_list]
            
            X = pd.concat([X, target_catagory_output], axis = 1) #Code when use pandas dataframe
            #X = cudf.concat([X, target_catagory_output], axis = 1) #Code when use cudf dataframe

        except Exception as e:
            raise RecommendationException(e, sys) from e  

        return X #target_catagory_output


#####################################################################################

#Sample code 

#leave_one_ce = catagory_leave_one_out_encoder('label', ['product_desc'], 1001)
#leave_one_ce.fit(df_train)
#leave_one_ce.transform(df_train)

