from utils.read_utils import read_yaml_key
from matplotlib import pyplot as plt
#from IPython.display import display
import config.config as fg #import CONFIGURATION_PATH, BASE_FOLDER_PATH_FOR_IMAGE
import logs.logger as log 
import tensorflow as tf
from PIL import Image
import numpy as np
import math
import os

SHOW_ITEMS_IMAGE = False
IMAGE_SIZES = 244  
BASEWIDTH = 300
CHANNEL = 3

def image_path(itemid):    
    
    """
    Given itemid generate the path where the image will be stored
    """  

    #log.write_log(f'Current base folder: {BASE_FOLDER_PATH_FOR_IMAGE}...', log.logging.DEBUG)
    
    if fg.BASE_FOLDER_PATH_FOR_IMAGE == "":

        log.write_log('Get the Base folder path for the image directory started...', log.logging.DEBUG)

        current_dir = os.path.dirname(os.path.realpath('__file__'))
        PROJECT_ROOT = os.path.abspath(os.path.join(current_dir))   

        log.write_log(f'Current PROJECT_ROOT folder: {PROJECT_ROOT}...', log.logging.DEBUG)

        while os.path.exists(os.path.join(PROJECT_ROOT, fg.CONFIGURATION_PATH)) == False:            

            PROJECT_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir))
            log.write_log(f'Next PROJECT_ROOT folder: {PROJECT_ROOT}...', log.logging.DEBUG)

        
        fg.BASE_FOLDER_PATH_FOR_IMAGE = os.path.join(PROJECT_ROOT, 
                                                    read_yaml_key(os.path.join(PROJECT_ROOT, fg.CONFIGURATION_PATH),  'data_source', 'data_folders'),
                                                    read_yaml_key(os.path.join(PROJECT_ROOT, fg.CONFIGURATION_PATH),  'data_source', 'raw_data_folder'),
                                                    read_yaml_key(os.path.join(PROJECT_ROOT, fg.CONFIGURATION_PATH),  'data_source', 'image_data')
                                                    )

        log.write_log('Get the Base folder path for the image directory completed...', log.logging.DEBUG)
    
    return os.path.join(fg.BASE_FOLDER_PATH_FOR_IMAGE,
                        str(itemid).zfill(10)[:3],
                        str(itemid).zfill(10) + ".jpg")

def get_relative_image_path(itemid):

    BASE_FOLDER_PATH = os.path.join(".",
                                    read_yaml_key(fg.CONFIGURATION_PATH, 'data_source', 'data_folders'),
                                    read_yaml_key(fg.CONFIGURATION_PATH, 'data_source', 'raw_data_folder'),
                                    read_yaml_key(fg.CONFIGURATION_PATH, 'data_source', 'image_data'),
                                    str(itemid).zfill(10)[:3],
                                    str(itemid).zfill(10) + ".jpg"
                                )

    return BASE_FOLDER_PATH


def get_image_path(itemid):

    img_path = image_path(itemid)
        
    #Case when the image for the article is not available skip it
    if os.path.exists(img_path) == False:
        return ""
    else:
        return img_path 

def show_image(img_path):    

    """    
    given image path show the image    
    """

    fig = plt.figure(figsize = (7, 7)) 
    
    if os.path.exists(img_path) == False:
        print(f'File {img_path} not found.')
        return
    
    image = Image.open(img_path)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Item: {os.path.basename(img_path)}, shape: {np.asarray(image).shape}')

def show_item_img(items, no_of_img_per_row = 4, figsiz = (25, 25)):        
        
    """
    Given list of items ids, show the item image 
    """

    columns = no_of_img_per_row
    rows = math.ceil(len(items)/columns)

    i = 1
    
    fig = plt.figure(figsize = figsiz)    
    
    for item in items:        
        
        img_path = image_path(item)
        if os.path.exists(img_path) == False:
            print(f"File {img_path} not found.")
            continue
            
        basewidth = BASEWIDTH
        image = Image.open(img_path)
        wpercent = (basewidth / float(image.size[0]))
        hsize = int((float(image.size[1]) * float(wpercent)))
        img_resize = image.resize((basewidth, hsize), Image.ANTIALIAS)        
        
        fig.add_subplot(rows, columns, i)

        # showing image
        plt.imshow(img_resize)
        plt.axis('off')
        plt.title(os.path.basename(img_path))
        i += 1

def show_item_img_detail(items, show_item_detail = False, no_of_img_per_row = 4,figsiz = (25, 25)):    
    
    """
    Given items dataframe, show the item image and details of the item
    """

    columns = no_of_img_per_row
    rows = math.ceil(items.shape[0]/columns)

    i = 1
    
    if show_item_detail == False:
        fig = plt.figure(figsize = figsiz)    
    
    for _,item in items.iterrows():
        
        #print("*"*60)
        img_path = image_path(item['article_id'])
        if os.path.exists(img_path) == False:
            print(f"File {img_path} not found.")
            continue
            
        basewidth = BASEWIDTH
        image = Image.open(img_path)
        wpercent = (basewidth / float(image.size[0]))
        hsize = int((float(image.size[1]) * float(wpercent)))
        img_resize = image.resize((basewidth, hsize), Image.ANTIALIAS)
        
        if show_item_detail == False:
            # Adds a subplot at the 1st position
            fig.add_subplot(rows, columns, i)

            # showing image
            plt.imshow(img_resize)
            plt.axis('off')
            plt.title(os.path.basename(img_path))
            i += 1
            
        else:
            
            #display(img_resize)
        
            print("="*10 + "Item details"+"="*10)        
            print(f"image size: {image.size}")
        
            item = item.to_dict() #Convert the DataFrame row to to_dict()
            for label,value in item.items():
                print(f"{label:30}: {value}")                
            print("\n"*2)

def get_image(image_path):   
    
    
    image_size = (IMAGE_SIZES ,IMAGE_SIZES ) 
    bits = tf.io.read_file(image_path)   
 
    image =  tf.image.decode_jpeg(bits, channels = CHANNEL)

    #Scale the image [0,1]
    image = tf.cast(image, tf.float32) / 255.0    
        
    #Most of the images has blank space around the images. So lets crop the image
    image = tf.image.central_crop(image, 0.80)
    
    #Resize the image as per the RESNET50 model
    image = tf.image.resize(image, image_size)         
           
    return image
   
def decode_train_image(user, item, image_path, label):

    return user, item, get_image(image_path), label

def decode_train_image(user, item, image_path):

    return user, item, get_image(image_path)