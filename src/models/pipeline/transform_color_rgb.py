from sklearn.base import BaseEstimator, TransformerMixin
from utils.exception import RecommendationException
import logs.logger as log 
import pandas as pd 
import webcolors
import sys
import re

class transform_color_rgb(BaseEstimator, TransformerMixin):

    """
    Transform color of the garment/article to RGB format
    """
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, Y = None):
        
        return self

    def transform(self, X = None):
            
        try:

            log.write_log('Transform article color to RGB format started...', log.logging.DEBUG)

            log.write_log('Transform color into RGB format...', log.logging.DEBUG)
            data = X.apply(lambda row: self.transform_rgb(row.colour_group_name, row.perceived_colour_master_name), axis = 1)

            df_rgb = pd.DataFrame(list(data), columns = ['Red', 'Green', 'Blue'])
            X['Red'] = df_rgb.Red
            X['Green'] = df_rgb.Green
            X['Blue'] = df_rgb.Blue
            
            log.write_log('Transform article color to RGB format completed.', log.logging.DEBUG)

            return X

        except Exception as e:
            raise RecommendationException(e, sys) from e  


    def clean_color(self, color_name):
    
        #Remove any special char
        pat = r'[^a-zA-Z]'
        color_name = re.sub(pat, ' ', color_name)

        #Remove extra space
        color_name = ' '.join(color_name.split())

        #Remove "Other"
        color_name = color_name.replace("Other", "")

        #Convert to lower
        color_name = color_name.lower()

        #Remove blank space
        color_name = color_name.replace(" ","")
        
        return color_name

    def get_color_name(self, color_name, master_color_name):
        
        closest_color_rgb = actual_color_rgb = None
        
        try:        
            closest_color_rgb = actual_color_rgb = webcolors.name_to_rgb(self.clean_color(color_name))   
        except ValueError:
            actual_color_rgb = None
            
            try:            
                closest_color_rgb = webcolors.name_to_rgb(self.clean_color(master_color_name))            
            except ValueError:          
                #Based on the article master database we see that most of the item they have are in black color. Also number of sales we see is are in black color  
                closest_color_rgb = webcolors.name_to_rgb('black') 
        
        if actual_color_rgb == None:
            return closest_color_rgb
        
        return actual_color_rgb

    def transform_rgb(self, color_name, master_color_name):

        rgb = self.get_color_name(color_name, master_color_name)
        #Divide it by 255 to normalize the 
        return rgb.red/255, rgb.green/255 , rgb.blue/255

    
        
