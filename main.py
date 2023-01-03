from utils.read_utils import read_yaml_key, read_from_parquet
from utils.images_utils import get_relative_image_path
from src.models.predict_model import Recommendation
from config.config import CONFIGURATION_PATH
import streamlit as st
import os
import gc

model_obj = Recommendation()

@st.cache
def load_customer_id():

    PATH = os.path.join(
                         read_yaml_key(CONFIGURATION_PATH, 'data_source','data_folders'),
                         'customer_lists.parquet'
                        )

    unique_customer_id = read_from_parquet(PATH)
    customer_lists = (unique_customer_id[['customer_id','article_id']]
                          .groupby('customer_id')['article_id']
                          .apply(list)
                          .reset_index()
                        )
    customer_lists = customer_lists.merge(unique_customer_id[['customer_id','org_customer_id']], on = ['customer_id'], how = 'inner')
    customer_lists = customer_lists.drop_duplicates(['customer_id','org_customer_id'], keep ='last')

    del [unique_customer_id]
    gc.collect()

    return customer_lists

def recommend(customer_id):

    #Recommended Items
    recommend_items = model_obj.predict(customer_id)
    
    #Map the recommended items to the image path
    recommend_items['image_path'] = list(map(get_relative_image_path, recommend_items['article_id']))

    return recommend_items

def main():

    st.markdown("<h1 style='text-align: center; font-size: 65px; color: #4682B4;'>{}</h1>".format('Recommender System'), 
    unsafe_allow_html=True)
    st.image("./references/banner.png")
   
    customer_lists = load_customer_id()

    selected_customer = st.selectbox(
        "Type or select a customer from the dropdown",
        customer_lists.org_customer_id.values
    )

    if st.button('Recommend'):

        output = recommend(selected_customer)
        items = output.article_id.values
        image_path = output.image_path.values

        st.header("Recommendation")
        
        cnt = 0 
        cols = st.columns(5)
        for i in range(5):
            cols[i].header(items[cnt])
            cols[i].image(image_path[cnt])
            cnt += 1        
        
        cols = st.columns(5)
        for i in range(5):
            cols[i].header(items[cnt])
            cols[i].image(image_path[cnt])
            cnt += 1

        cols = st.columns(5)
        for i in range(5):
            cols[i].header(items[cnt])
            cols[i].image(image_path[cnt])
            cnt += 1
          

if __name__ == "__main__":
    main()

