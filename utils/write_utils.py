from utils.exception import RecommendationException
import pandas as pd
import numpy as np
import logging
#import cudf
import yaml
import dill
import sys
import os

def save_to_parquet(df, file_path, replace = False):

    """
    Save dataframe to parquet. Will check if directory exists, if not will create directory before saving.
    df: pandas/cudf dataframe to save to file.
    file_path: str relative path to save the file.
    replace: bool to check if the file exists, if yes will delete it.
    """    
    if replace:
        if os.path.exists(file_path):
            os.remove(file_path)

    os.makedirs(os.path.dirname(file_path), exist_ok = True)

    df.to_parquet(file_path)

def save_to_pickle(df, file_path, replace = False):

    """
    Save dataframe to pickle. Will check if directory exists, if not will create directory before saving.
    df: pandas/cudf dataframe to save to file.
    file_path: str relative path to save the file.
    replace: bool to check if the file exists, if yes will delete it.
    """

    if replace:
        if os.path.exists(file_path):
            os.remove(file_path)

    os.makedirs(os.path.dirname(file_path), exist_ok = True)

    df.to_pickle(file_path, compression = 'gzip', protocol = 4)

def save_yaml_data(file_path, content, replace = False):
    
    """
    Save to yaml file. Will check if directory exists, if not will create directory before saving.
    file_path: str relative path to save the file.
    content: object file content.
    replace: bool to check if the file exists, if yes will delete it.
    """    
    
    if replace:
        if os.path.exists(file_path):
            os.remove(file_path)

    os.makedirs(os.path.dirname(file_path), exist_ok = True)

    with open(file_path, "w") as file:
        yaml.dump(content, file)
    
def save_compressed_numpy_array_data(file_path, array):

    """
    Save numpy array data as compressed file format(.npz).
    file_path: str location of file to save.
    array: np.array data to save.
    """    
    
    dir_path = os.path.dirname(file_path)        
    os.makedirs(dir_path, exist_ok = True)
    np.savez_compressed(file_path, array)

def save_object(file_path: str, obj: object) -> None:
       
    logging.info("save_object method of main_utils class started.")

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "wb") as file_obj:
        dill.dump(obj, file_obj)

    logging.info("save_object method of main_utils class ended.")

    
    