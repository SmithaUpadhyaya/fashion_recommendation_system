from utils.read_utils import read_yaml_key, read_csv
from utils.exception import RecommendationException
from utils.write_utils import save_to_parquet
#from config.config import CONFIGURATION_PATH
import utils.analysis_utils as util
import logs.logger as log 
import pandas as pd
import numpy as np
import sys
import re
import os
import gc

def clean_text(text):
        #return remove_extra_space(remove_stopword(to_lowercase(remove_numeric(text))))
        return remove_extra_space(remove_stopword(remove_numeric(text)))

def remove_stopword(text):
    stopword = ['and']

    word_tokens = text.split() 
    filtered_word = [w for w in word_tokens if not w.lower() in stopword]
    return ' '.join(filtered_word)

def to_lowercase(text):
    return text.lower().strip()

def remove_numeric(text):
    pat = r'[^a-zA-Z]'
    return re.sub(pat, ' ', text)

def remove_extra_space(text):
    return ' '.join(text.split())

def merge_similar_product_type(df_article):

    unique_product_type = df_article.product_type_name.unique()
    colindex = ['product_type_name']  

    #Merge all product type of "hat" ,"cap", "Beanie" as "Hat Cap"
    #Beanie look like some kind of hat
    filters = ["Hat", "Cap","Beanie"]
    items = util.filter_types(filters, unique_product_type)
    rowindex = df_article[df_article.product_type_name.isin(items)].index.tolist()
    df_article.loc[rowindex, colindex] = "Hat Cap"

    #Merge all product type of "Wood balls","Side table" etc as "Cloths Care "
    filters = ["Wood balls","Side table", "Stain remover spray","Sewing kit", "Clothing mist", "Zipper head"]
    items = util.filter_types(filters, unique_product_type)
    rowindex = df_article[df_article.product_type_name.isin(items)].index.tolist()
    df_article.loc[rowindex, colindex] = "Cloths Care"

    #Merge all product type of "Wood balls","Side table" as "Cloths Care "
    filters = ["Ring", "Necklace", "Bracelet", "Earring"]
    items = util.filter_types(filters, unique_product_type)
    rowindex = df_article[df_article.product_type_name.isin(items)].index.tolist()
    df_article.loc[rowindex, colindex] = "Jewellery"

    #Merge all product type of "Polo shirt" as "Shirt"
    filters = ["Polo shirt"]
    items = util.filter_types(filters, unique_product_type)
    rowindex = df_article[df_article.product_type_name.isin(items)].index.tolist()
    df_article.loc[rowindex, colindex] = "Shirt"

    #Merge all product type of "Bag" as "Bag"
    filters = ["Bag", 'Backpack']
    items = util.filter_types(filters, unique_product_type)
    rowindex = df_article[df_article.product_type_name.isin(items)].index.tolist()
    df_article.loc[rowindex, colindex] = "Bag"

    #Merge all product type like "Swimwear bottom","Swimsuit","Bikini top" as "Swimwear"
    filters = ["Swimwear"]
    items = util.filter_types(filters, unique_product_type)
    rowindex = df_article[df_article.product_type_name.isin(items)].index.tolist()
    df_article.loc[rowindex, colindex] = "Swimwear"

    #Merge all product type like "Underwear bottom","Underwear Tights","Underwear body", "Long John" as "Underwear"
    filters = ["Underwear", "Long John"]
    items = util.filter_types(filters, unique_product_type)
    rowindex = df_article[df_article.product_type_name.isin(items)].index.tolist()
    df_article.loc[rowindex, colindex] = "Underwear"

    #Merge all product type like "Hair","Alice band", "Hairband" as "Hair Accessories"
    filters = ["Hair", "Alice band", "Hairband"]
    items = util.filter_types(filters, unique_product_type)
    rowindex = df_article[df_article.product_type_name.isin(items)].index.tolist()
    df_article.loc[rowindex, colindex] = "Hair Accessories"

    #Merge all product type like "Sandals","Heeled sandals","Wedge","Pumps", "Slippers" as "Sandals"
    #Ballerinas type of scandel that are used for ballerine performance
    filters = ["Sandals", "Heels", "Flip flop", "Wedge", "Pump", "Slipper", "Ballerinas"]
    items = util.filter_types(filters, unique_product_type)
    rowindex = df_article[df_article.product_type_name.isin(items)].index.tolist()
    df_article.loc[rowindex, colindex]= "Sandals"

    #Merge all product type like "Toy","Soft Toys" as "Toys"
    filters = ["Toy"]
    items = util.filter_types(filters, unique_product_type)
    rowindex = df_article[df_article.product_type_name.isin(items)].index.tolist()
    df_article.loc[rowindex, colindex] = "Toy"

    #Merge all product type like "Flat Shoe","Other Shoe","Boots", "Sneakers" as "shoe"
    filters = ["shoe","Boot", "Leg warmer", "Sneakers"]
    items = util.filter_types(filters, unique_product_type)
    rowindex = df_article[df_article.product_type_name.isin(items)].index.tolist()
    df_article.loc[rowindex, colindex] = "Shoe"

    #Merge all product type like "Dog wear" as "Dog wear"
    filters = ["Dog wear"]
    items = util.filter_types(filters, unique_product_type)
    rowindex = df_article[df_article.product_type_name.isin(items)].index.tolist()
    df_article.loc[rowindex, colindex] = "Dog wear"

    #Merge all product type like "Sunglasses","EyeGlasses" as "Glasses"
    filters = ["glasses"]
    items = util.filter_types(filters, unique_product_type)
    rowindex = df_article[df_article.product_type_name.isin(items)].index.tolist()
    df_article.loc[rowindex, colindex] = "Glasses"

    #Merge all product type like "Cosmetics","Chem. Cosmetics" as "Cosmetics" 
    filters = ["Cosmetics"]
    items = util.filter_types(filters, unique_product_type)
    rowindex = df_article[df_article.product_type_name.isin(items)].index.tolist()
    df_article.loc[rowindex, colindex] = "Cosmetics"

    #Merge all product type like "earphone case","case" as "Mobile Accesories" 
    filters = ["earphone case", "case"]
    items = util.filter_types(filters, unique_product_type)
    rowindex = df_article[df_article.product_type_name.isin(items)].index.tolist()
    df_article.loc[rowindex, colindex] = "Mobile Accesories"

    #Merge all product type like "Bra extender" as "Bra" 
    filters = ["Bra extender"]
    items = util.filter_types(filters, unique_product_type)
    rowindex = df_article[df_article.product_type_name.isin(items)].index.tolist()
    df_article.loc[rowindex, colindex] = "Bra"

    #Merge all product type like "Nipple covers" not under Nightwear department_name as "Bar"
    filters = ["Nipple covers"]
    rowindex = df_article[(df_article.department_name != 'Nightwear') & 
                        (df_article.product_type_name.isin(filters))].index.tolist()
    df_article.loc[rowindex, colindex] = "Bra"


    #Merge all product type like "Tailored Waistcoat","Outdoor Waistcoat" as "Waistcoat" 
    filters = ["Waistcoat"]
    items = util.filter_types(filters, unique_product_type)
    rowindex = df_article[df_article.product_type_name.isin(items)].index.tolist()
    df_article.loc[rowindex, colindex] = "Waistcoat"

    #Product type like "Outdoor trousers" as "trousers" 
    filters = ["Outdoor trousers"]
    items = util.filter_types(filters, unique_product_type)
    rowindex = df_article[df_article.product_type_name.isin(items)].index.tolist()
    df_article.loc[rowindex, colindex] = "Trousers"

    #UnderDress are wore along with some Dress. 
    #So lets group them with Dress segment as if user has purchase and dress in past they might need underdress to wore along with it
    filters = ["Underdress"]
    items = util.filter_types(filters, unique_product_type)
    rowindex = df_article[df_article.product_type_name.isin(items)].index.tolist()
    df_article.loc[rowindex, colindex] = "Dress"

    #Product type like "Towel","Blanket","Cushion","Waterbottle","Robe" as "trousers" 
    filters = ["Towel","Blanket","Cushion","Waterbottle","Robe"]
    items = util.filter_types(filters, unique_product_type)
    rowindex = df_article[df_article.product_type_name.isin(items)].index.tolist()
    df_article.loc[rowindex, colindex] = "House Accesories"

    return df_article

def clean_department_name(df_article):

    unique_article_department = df_article.department_name.unique()

    log.write_log('Remove all the article with the word "inactive" from the department_name. Since those item are not longer available ', log.logging.DEBUG)
    drop_departments =  df_article[df_article.department_name.isin(util.filter_types(['inactive'], unique_article_department))].index
    df_article.drop(drop_departments, inplace = True)


    log.write_log('Renaming department_name value from  Tights basic to Tights...', log.logging.DEBUG)
    
    colindex = ['department_name']
    filters = ["Tights basic"]
    rowindex = df_article[df_article.department_name.isin(filters)].index.tolist()
    df_article.loc[rowindex, colindex] = "Tights"

    log.write_log('Replacing certain department_name to meaning name...', log.logging.DEBUG)
    unique_article_department = df_article.department_name.unique()


    ###### Under "index_name" replace "Socks & Tights"  to "Socks" if department_name contain "Socks" else "Tights" #####
    #Replace "index_name" feature based on department_name. i.e if department_name == "Socks" then Socks else if department_name == "Tights" the  "Tights"
    #                       "Lingeries/Tights" -> Socks 
    #                       "Lingeries/Tights" -> Tights 
    #Since we are going to merge these column to one. And in certain case index_name provide more detail then department

    colindex = ['index_name']  
    filters = ["Socks"]
    items = util.filter_types(filters, unique_article_department)
    rowindex = df_article[df_article.department_name.isin(items)].index.tolist()
    df_article.loc[rowindex,colindex] = "Socks"

    colindex = ['index_name']  
    filters = ["Tights"]
    items = util.filter_types(filters, unique_article_department)
    rowindex = df_article[df_article.department_name.isin(items)].index.tolist()
    df_article.loc[rowindex,colindex] = "Tights"

  

    #* Replace "S&T" in department_name as "Socks"
    #* Replace "Acc" in department_name as "Accessories"
    #* Replace "Accessories Other" , "Other Accessories" in department_name as "Accessories"
    #* Rename all "Jersey Basic" and other similar type in department_name to "Jersey"

    #Replace "S&T" in department_name as "Socks"
    colindex = ['department_name']  
    filters = ["S&T"]
    items = util.filter_types(filters, unique_article_department)
    rowindex = df_article[df_article.department_name.isin(items)].index.tolist()
    df_article.loc[rowindex,colindex] = df_article.loc[rowindex,colindex].apply(lambda x: x.replace("S&T","Socks",regex = True))


    #Replace "Acc" in department_name as "Accessories"
    colindex = ['department_name']  
    filters = ["Acc"]
    items = util.filter_types(filters, unique_article_department)
    rowindex = df_article[df_article.department_name.isin(items)].index.tolist()
    df_article.loc[rowindex,colindex] = df_article.loc[rowindex,colindex].apply(lambda x: x.replace("Acc","Accessories",regex = True))

    #Replace "Accessories Other" , "Other Accessories" in department_name as "Accessories"
    colindex = ['department_name']  
    filters = ["Accessories Other", "Other Accessories"]
    items = util.filter_types(filters, unique_article_department)
    rowindex = df_article[df_article.department_name.isin(items)].index.tolist()
    df_article.loc[rowindex,colindex] = "Accessories Other"


    #Rename all "Jersey Basic" and other similar type in department_name to "Jersey"
    colindex = ['department_name']
    filters = ["Jersey Basic","Heavy Basic Jersey","Light Basic Jersey" ]
    rowindex = df_article[df_article.department_name.isin(filters)].index.tolist()
    df_article.loc[rowindex,colindex] = "Jersey"

    filters = ["Jersey Fancy", "Jersey fancy", "Jercy Fancy DS", "Tops Fancy Jersey", "Jersey License" ,"AK Tops Jersey & Woven"]
    rowindex = df_article[df_article.department_name.isin(filters)].index.tolist()
    df_article.loc[rowindex,colindex] = "Jersey Fancy"

    filters = ["Heavy Basic Jersey"]
    rowindex = df_article[df_article.department_name.isin(filters)].index.tolist()
    df_article.loc[rowindex,colindex] = "Jersey Heavy"

    filters = ["Jersey/Knitwear Premium", "Projects Jersey & Knitwear"]
    rowindex = df_article[df_article.department_name.isin(filters)].index.tolist()
    df_article.loc[rowindex,colindex] = "Jersey Knitwear"

    filters = ["Underwear Jersey"]
    rowindex = df_article[df_article.department_name.isin(filters)].index.tolist()
    df_article.loc[rowindex,colindex] = "Jersey Underwear"



    log.write_log('Remove the word "DS" from department_name...', log.logging.DEBUG)

    unique_article_department = df_article.department_name.unique()

    colindex = ['department_name']  
    filter_DS = []
    filter_word = "DS"
    for name in unique_article_department:
        if filter_word in name:
            filter_DS = filter_DS + [name]

    rowindex = (df_article[df_article.department_name.isin(filter_DS)].index.tolist())
    df_article.loc[rowindex,colindex] = df_article.loc[rowindex,colindex].apply(lambda x: x.replace("DS", "", regex = True))

    log.write_log('Remove UW/NW from department_name...', log.logging.DEBUG)
    unique_article_department = df_article.department_name.unique()

    #"UW/NW" is either "Underwear" or "Nightwear" or Just remove that if those detail is available in index_name , index_group_name

    #Lets Replace department_name which contain "UW/NW" based on the product_type_name. 
    #For product_type_name = "Underwear" lets remove the word "UW/NW" since we are going to merge these columne it will reflect the item
    colindex = ['department_name']  
    filters = ["UW/NW"]
    items = util.filter_types(filters, unique_article_department)

    rowindex = (df_article[(df_article.department_name.isin(items)) &
                        (df_article.product_type_name == "Underwear")
                        ].index.tolist())
    df_article.loc[rowindex,colindex] = df_article.loc[rowindex,colindex].apply(lambda x: x.replace("UW/NW", "", regex = True))

    #For product_type_name != "Underwear" rename the "UW/NW" = "Nightwear"
    rowindex = (df_article[(df_article.department_name.isin(items)) &
                        (df_article.product_type_name != "Underwear")
                        ].index.tolist())

    df_article.loc[rowindex,colindex] = df_article.loc[rowindex,colindex].apply(lambda x: x.replace("UW/NW", "Nightwear", regex = True))

    log.write_log('Rename UW from department_name to Underwear...', log.logging.DEBUG)
    colindex = ['department_name']  
    rowindex = df_article[df_article.department_name == 'UW'].index.tolist()
    df_article.loc[rowindex, colindex] = df_article.loc[rowindex, colindex].apply(lambda x: x.replace("UW", "Underwear", regex = True))



    log.write_log('Clean "index_name" based on the value of "department_name"...', log.logging.DEBUG)

    unique_article_department = df_article.department_name.unique()
    
    #By defination "lingerie" means women's underwear and nightclothes. This could either be a Nightwear or Swimwera
    #To simplify lets rename "Lingerie/Tights" as "Lingerie" for item under department_name = "Nightwear"
    #Replace index_name = "Lingeries" where department_name contain = "Lingerie"
    colindex = ['index_name']  
    filters = ["Lingerie"]
    items = util.filter_types(filters, unique_article_department)
    rowindex = df_article[df_article.department_name.isin(items)].index.tolist()
    df_article.loc[rowindex, colindex] = "Lingerie"


    log.write_log('Rename Lingerie/Tights as "Lingerie" for item under department_name = "Nightwear"', log.logging.DEBUG)
    #By defination "lingerie" means women's underwear and nightclothes. This could either be a Nightwear or Swimwera
    #To simplify lets rename "Lingerie/Tights" as "Lingerie" for item under department_name = "Nightwear"
    colindex = ['index_name']  
    filters = ["Nightwear", "Swimwear"]
    rowindex = df_article[df_article.department_name.isin(filters)].index.tolist()
    df_article.loc[rowindex, colindex] = "Lingerie"


    return df_article

def clean_product_type_name(df_article):

    # Drop the data row where "product_type_name" = "Unknown"
    log.write_log('Dropping the data row where product_type_name =  Unknown...' , log.logging.DEBUG)

    df_article.drop(df_article[df_article.product_type_name == "Unknown"].index, 
                    inplace = True, 
                    axis = 0)

    #### Group similar product_type_name #### 
    #To reduce cardinality in number of unique product type name we have.
    #Steps:
    #1. Create a new column "product_type_name_org" and copy the orignal value and update "product_type_name".
    #2. Group similar product based on the catagory they belong to
    
    log.write_log(f'Creating new column "product_type_name_org" that will be grouping of similar product type...', log.logging.DEBUG)

    colname = 'product_type_name_org'      
    df_article[colname] = df_article.product_type_name #Copy the original value to new column "_org"
    df_article = merge_similar_product_type(df_article)


    log.write_log('Discard product type which has less then 5 items', log.logging.DEBUG)
    
    class_label_cnt = df_article.product_type_name.value_counts()
    threshold = 5
    df_article = df_article[~df_article.product_type_name.isin(class_label_cnt[class_label_cnt <= threshold].index)]

    return df_article

def clean_article_data(config_path):

    try:

        log.write_log('Cleaning article information started...', log.logging.DEBUG)
        ARTICLES_MASTER_TABLE = os.path.join( 
                                            read_yaml_key(config_path,'data_source','data_folders'),
                                            read_yaml_key(config_path,'data_source','raw_data_folder'),
                                            read_yaml_key(config_path,'data_source','article_data'),
                                        )
        
        log.write_log('Reading article csv file...', log.logging.DEBUG)
        df_article = read_csv(ARTICLES_MASTER_TABLE)

        df_article = clean_product_type_name(df_article)

        df_article = clean_department_name(df_article)

        log.write_log('Drop the item where product_type_name = "Nipple covers" in department_name = "Nightwear"  and index_name = "Lingeries/Tights"', log.logging.DEBUG)
        #As item image does not match the item description
        df_article.drop(df_article[(df_article.department_name == 'Nightwear') & 
                                (df_article.product_type_name == 'Nipple covers') &
                                (df_article.index_name == 'Lingeries/Tights')
                                ].index, 
                        axis = 0, 
                        inplace = True)


        log.write_log('Remove article_id where "bar" product_type_name is defined in "MensWear" catagory', log.logging.DEBUG)
        rowindex = df_article[(df_article.product_type_name == "Bra") & 
                            (df_article.index_name == "Menswear")].index.tolist()

        df_article.drop(rowindex, axis = 0, inplace = True)


        log.write_log('Clean product_group_name = Underwear/nightwear to either Underwear or Nightwear', log.logging.DEBUG)
        #Sleeping sack are items for baby that are used in Nightwear.
        colindex = 'product_group_name'
        rowindex = (df_article[df_article.product_group_name == "Underwear/nightwear"].index.tolist())
        df_article.loc[rowindex,colindex] =  "Nightwear"


        log.write_log('Clean text/catagory information...', log.logging.DEBUG)
        
        catagory_column = ['product_type_name', 'product_group_name', 'index_group_name', 'index_name', 'department_name']
        for col in catagory_column:
            df_article["clean_"+ col] = df_article[col].apply(clean_text)


        ######################################     Mark items which does not have images        ######################################
        log.write_log('Mark items which does not have images...', log.logging.DEBUG)
        MISSING_ITEM_IMAGES =  os.path.join( 
                                            read_yaml_key(config_path,'data_source','data_folders'),
                                            read_yaml_key(config_path,'data_source','interim_data_folder'),
                                            read_yaml_key(config_path,'data_source','item_image_missing'),
                                        )

        item_missing_image = np.load(MISSING_ITEM_IMAGES)['arr_0']
        #Mark article_id 0925656001, 0617835001 from article list since image is not clearly visible.
        item_missing_image = np.append(item_missing_image, [925656001, 617835001])
        df_article['is_item_image_missing'] = df_article.article_id.isin(item_missing_image)

        ######################################    Reduce cardinality in product type names        ######################################
        # Most of the unique product type is more compare the the number of article that belong to that catagory
        # So to reduce cardinlaity of the product type. 
        # Lets group item/article which has least number of items in that catagory as Other 

        df_article['new_product_type_name'] = util.cumulatively_categorise(df_article.clean_product_type_name, 
                                                                           threshold = 0.85, 
                                                                           return_categories_list = False)



        ######################################    Add Event TimeStamp Column        ######################################
        #Since this is is clean master table and not feature. so no need to add timestamp column
        #log.write_log('Add Event timestamp feature that will be used when upload the file at Feature stoe...', log.logging.DEBUG)
        #df_article['event_timestamp'] = pd.date_range(end = pd.Timestamp.now(), periods = len(df_article), freq = 'S')

        df_article.reset_index(drop = True, inplace = True)
        
        ######################################     Save as        ######################################
        log.write_log('Save the cleaned article as parquet file format...', log.logging.DEBUG)
        CLEAN_ARTICLES_MASTER_TABLE = os.path.join( 
                                                    read_yaml_key(config_path,'data_source','data_folders'),
                                                    read_yaml_key(config_path,'data_source','processed_data_folder'),
                                                    read_yaml_key(config_path,'data_source','clean_article_data'),
                                            )

        #CLEAN_ARTICLES_MASTER_TABLE = os.path.splitext(CLEAN_ARTICLES_MASTER_TABLE)[0] + '.pkl'
        #save_to_pickle(df_article, CLEAN_ARTICLES_MASTER_TABLE)    
        save_to_parquet(df_article, CLEAN_ARTICLES_MASTER_TABLE)

        del [df_article, item_missing_image]
        gc.collect()

        return

    except Exception as e:

        raise RecommendationException(e, sys) 




    

    