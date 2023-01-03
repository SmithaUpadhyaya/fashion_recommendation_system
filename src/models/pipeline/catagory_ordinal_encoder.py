from sklearn.base import BaseEstimator, TransformerMixin
from utils.exception import RecommendationException
import category_encoders as ce
import logs.logger as log
import pandas as pd
#import cudf
import sys


class catagory_ordinal_encoder(BaseEstimator, TransformerMixin):

    """
    Class will convert catagorical feature into integer
    """

    def __init__(self, label_column_name, columns_list) -> None:
    
        super().__init__()

        self.columns_list = columns_list
        self.label_column_name = label_column_name
        self.ordinal_ce = ce.ordinal.OrdinalEncoder(verbose = 1, return_df = True)
    

    def fit(self, X, Y = None):

        try:

            log.write_log(f'Fit catagory ordinal encoder features started...', log.logging.DEBUG)
            log.write_log(f'Features to fit {self.columns_list}...', log.logging.DEBUG)

            #self.ordinal_ce.fit(X[self.columns_list].to_pandas(), X[self.label_column_name].to_pandas()) #Code when use cudf dataframe
            self.ordinal_ce.fit(X[self.columns_list], X[self.label_column_name]) #Code when use pandas dataframe

        except Exception as e:
            raise RecommendationException(e, sys) from e  

        return self


    def transform(self, X = None):

        try:

            log.write_log(f'Transform catagory ordinal encoder features started...', log.logging.DEBUG)
            log.write_log(f'Features to transform {self.columns_list}...', log.logging.DEBUG)

            ordinal_catagory_output = self.ordinal_ce.transform(X[self.columns_list]) #Code when use pandas dataframe. , X[self.label_column_name]
            #ordinal_catagory_output = self.ordinal_ce.transform(X[self.columns_list].to_pandas())   #Code when use cudf dataframe. , X[self.label_column_name].to_pandas()  
            #ordinal_catagory_output = cudf.DataFrame.from_pandas(ordinal_catagory_output) #Code when use cudf dataframe
            
            log.write_log(f'Merge transformed feature to main dataset...', log.logging.DEBUG)
            ordinal_catagory_output.columns = [x + '_oce' for x in self.columns_list]
            
            X = pd.concat([X, ordinal_catagory_output], axis = 1) #Code when use pandas dataframe
            #X = cudf.concat([X, ordinal_catagory_output], axis = 1)#Code when use cudf dataframe
        
        except Exception as e:
            raise RecommendationException(e, sys) from e  
        
        return X #ordinal_catagory_output


#####################################################################################

#Sample code 
#ordinal_ec = Catagory_Ordinal_Encoder('label', ['product_group_name'])
#ordinal_ec.fit(df_train)
#ordinal_ec.transform(df_train)


