# **H&M Fashion Recommendation System** #

Recommender systems became one of the essential areas in the machine learning field. In our daily life, recommender systems affect us in one way or another, as they exist in almost all the current websites such as social media, shopping websites, and entertainment websites. Buying clothes online became a widespread trend nowadays.  In this regard, item images play a crucial role, no one wants to buy a new shirt without seeing how it looks like.

In fashion recommendation adding the items images to the model has proven to show a significant lift in the recommendation performance when the model is trained not only on the relational data describing the interaction between users and items but also on how it looks.

</br>
</br>

# **Problem Statement** #
Product recommendations are key to enhance customer exeperiance and help them to find the right product from huge corpus of products. When cusomer find the right product that are mostly like going to add the item to cart and which help in company revenue. 

</br>
</br>

# **Solution** #

Recomendation for given customer is divided in to three part: </br>
1) Candinate Generation</br>
2) Find the relevent item that customer might like from the generated candinates</br>
3) Rank the generated candinates</br>


- **Candinate Generation model**
    </br>For candinate generation have use strategy:</br>
    * Strategy 1: Recommend items purchase by similar user in last 1 week.</br>
    * Strategy 2: Recommend most popularly item in last 1 week and other item that are purchase together with those by users.</br>
    * Strategy 3: Recommend item popularly last year at same time. </br></br>

- **Find the relevent item for the query user:** 
</br>Generated candinate from the previous step is then pass to model that will classify items that are relevent to the query user based on there previous purchase.

    Model is trained to learn lantent feature of user and item using the item images. Model implemented is based on  research paper <a href="https://arxiv.org/abs/2205.02923">"End-to-End Image-Based Fashion Recommendation"</a> by Shereen Elsayed, Lukas Brinkmeyer and Lars Schmidt-Thieme. Most of the proposed model for images based relied on using recommendation system pre-trained networks for items images features extraction. Author of this research paper proposed to extract the latent item’s image features and refine the image features further and get better representation by jointly train the whole image network simultaneously with the recommender model. The proposed model utilizes item's image features extracted by a calibrated ResNet50 component.

</br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src="references\Image-EtE.png"/>
</br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Source of image: Taken from the<a href="https://arxiv.org/abs/2205.02923">research paper</a>

</br>   So the main idea is while we are training the model to learn latent feature of the user and item by their interation data, we also back train the last layers of the image extraction model to fine tune the image embedding. This will futher help to learn user and item feature. 

- **Rank the generated candinates:**
</br>Filtered candinate is then passed to the ranking model to sort the most relevent item at the top.
 


# **Chalanges** #

1) **Generation negative candidates**: 
</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Given only user transaction data we had only positive candidates i.e item's user's has purchase. We assume that the item user has purchase are the item's that they like. The strategy that I have experimened where </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1. Randomly selecting the item's from item's corpus in the same catagory of the item user has purchased. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2. Convent the items that are purchase on the same day by other user's as negative candidates of the user. Number of negative candidates to selected is based on the percentage of the item catagory the user has purchase. e.g if user has purchase total of 10 items for a given transaction date, out of which 3 items where jeans and 7 items where top. Then for negative candidates we shall has 30% of items </br>

</br> 
</br>

2) **Imbalance dataset**:
</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Number of negative candidates was in ratio of 10:1. So for a given 1 positive candinates we have 10 negative candidates. This ratio is based on the negative candiates we generated. Approach that I have used is to ensemble different resampled datasets. 
So we building 10 models that use all the positive candidates and n-differing negative candiates per model. Output score of the ensemble models is then passed to Dense layer to predict if the user will like the item or not

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="references\Stacked_Ensemble_Merge_model.png"/>

</br>
</br>

# **Tech Stack Used** #
</br>
1) Python 3.7.9 </br>
2) Deep learning algorithm </br>
3) Ranking LightGBM </br> 
4) Streamlit </br>
5) Feast </br>
</br>

# **Data Source** #
                 
Step 1: Downloaded the data : [kaggle link](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)</br>
Step 2: Extract and copy the files in the data/raw/ folder</br>
</br>

# **Requirements** #
Refer requirements.txt 
</br>
</br>

# **Demo** #
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="./references/demo.gif" />








