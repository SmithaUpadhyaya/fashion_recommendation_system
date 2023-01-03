from matplotlib import pyplot as plt
from collections import Counter
from datetime import timedelta
import pandas as pd
import math


def random_sample_by_catagory(df ,filter, group_by = '', no_sample = 4):
    
    if filter == '':
        
        if group_by == '':
            return  df.sample(n = no_sample, replace = False)
        else:
            return  df.groupby(group_by).apply(pd.DataFrame.sample, n = no_sample, replace = True ).reset_index(drop=True)

    else:
        query = ' & '.join([f'{k}=="{v}"' for k, v in filter.items()])
        #print(query)
        
        if group_by == '':
            return  df.query(query).sample(n = no_sample, replace = False)
        else:
            return  df.query(query).groupby(group_by).apply(pd.DataFrame.sample, n = no_sample, replace = True ).reset_index(drop=True)


def filter_types(filters, list_to_filter_from):   

    """
    Will get all the items with department name containing the filter department name. 
    Why use department_name? Because found that it provide some more detail on the item. 
    Like if the item is for kid for which age group the item belong to.
    """

    filter_name = []

    for dept in list_to_filter_from:
        
        dept_name = [dept for s in filters if s.lower() in dept.lower()]
        filter_name = filter_name + dept_name

    return filter_name


#Source code https://towardsdatascience.com/dealing-with-features-that-have-high-cardinality-1c9212d7ff1b
def cumulatively_categorise(column, threshold = 0.75, return_categories_list = True):   
    
    #Find the threshold value using the percentage and number of instances in the column
    threshold_value = int(threshold*len(column))
    #Initialise an empty list for our new minimised categories
    categories_list = []
    #Initialise a variable to calculate the sum of frequencies
    s = 0
    #Create a counter dictionary of the form unique_value: frequency
    counts = Counter(column)

    #Loop through the category name and its corresponding frequency after sorting the categories by descending order of frequency
    for i,j in counts.most_common():
        #Add the frequency to the global sum
        s += dict(counts)[i]
        #Append the category name to the list
        categories_list.append(i)
        #Check if the global sum has reached the threshold value, if so break the loop
        if s >= threshold_value:
            break
    #Append the category Other to the list
    categories_list.append('Other')

    #Replace all instances not in our new categories by Other  
    new_column = column.apply(lambda x: x if x in categories_list else 'Other')

    #Return transformed column and unique values if return_categories=True
    if(return_categories_list):
        return new_column, categories_list
    #Return only the transformed column if return_categories=False
    else:
        return new_column



def plot_frequency(df, col, top = 30 ,title = ""):

    """
    Method return total number of item user has purchase. 
    Input Param:
    * df: dataframe
    * col: column to group by and count the frequency
    * top: plot to top n computed counts
    * title: title of the plot
    Output Explain:
    * Count of 1 indicate user has purchase only 1 item till date. 
    * Count of 2 indicate use has purchase 2 items either or same date or different date
    """
    
    grp = (df
            .groupby(col)[col]#.groupby(col)['t_dat']
            .count()
            .reset_index(name = 'total_transaction')
            )

    grp.sort_values(by = 'total_transaction', 
                    ascending = False, 
                    inplace = True)

    grp = (grp['total_transaction']
            .value_counts()
            .reset_index()
            .sort_values('index')
            )
    
    total = grp.total_transaction.sum()
    
    plt.figure(figsize = (20,5))

    pps = plt.bar(grp['index'].astype(str).values[0:top], 
                    grp['total_transaction'].values[0:top])

    for p in pps:
        height = p.get_height()
        per = round((height/total)*100,1)

        plt.annotate(f'%: {per}', #f'Cnt: {height}, %: {per}',\n% of outlier: {per_outlier}
                        xy = (p.get_x() + p.get_width() / 2, height),
                        xytext = (0, 5), # 3 points vertical offset
                        textcoords = "offset points",
                        ha = 'center', 
                        va = 'bottom')


    plt.title('Top '+ str(top) +' frequency of ' + title)
    plt.xlabel('Number frequency')
    plt.ylabel('Frequency')




#Source Code: https://github.com/JacobCP/kaggle-handm-helpers/blob/master/fe.py
def day_week_numbers(dates):
    
    """
    assign week numbers to dates, such that:
    - week numbers represent consecutive actual weeks on the calendar. 
    - the latest date in the dates provided is the last day of the latest week number
    args:
    dates: pd.Series of date strings
    returns:
    day_weeks: pd.Series of week numbers each day in original pd.Series is in
    """
    
    #pd_dates = cudf.to_datetime(dates)
    pd_dates = pd.to_datetime(dates)
   
    #unique_dates = cudf.Series(pd_dates.unique())
    unique_dates = pd_dates.drop_duplicates()
    numbered_days = unique_dates - unique_dates.min() + timedelta(1)
    numbered_days = numbered_days.dt.days
    extra_days = numbered_days.max() % 7
    numbered_days -= extra_days
    day_weeks = (numbered_days / 7).apply(lambda x: math.ceil(x))
    
    
    day_weeks_map = pd.DataFrame(
        {"week_number": day_weeks, "unique_dates": unique_dates}
    )

    
    return day_weeks_map 

#method will calculate the % of the type it contribute to overall purchase of the user
def ctb_type_pct(df, new_col_name , category_list):
    pvt_output = pd.pivot_table(data = df, 
                                index = ['customer_id'], 
                                columns = ['new_product_type_name'], 
                                values = 'article_id',
                                aggfunc = 'count',
                                fill_value = 0,
                                margins = True, margins_name = 'Total',                               
                                )

    pvt_output.drop(index = ['Total'], inplace = True) #Drop row with index 'Total'
    
    #What % of there overall user purchase does the items type contribute of user transaction
    for col in pvt_output.columns:
        if col != 'Total':
            pvt_output[col] = round(pvt_output[col]/pvt_output['Total'], 2) #len(category_list)


    #Transform the data from wide to long
    pvt_output = pd.melt(pvt_output.reset_index(level = 0), 
                         id_vars = 'customer_id', 
                         value_vars = category_list)
    
    #pct_items_ctb_last_6week: stands for percentage(pct) of the product type contribution(ctb) in the total purchase made by the user in last 6 week/overall of data
    pvt_output.columns = ['customer_id','new_product_type_name', new_col_name] #'pct_type_ctb_last_6week'
    pvt_output = pvt_output[pvt_output[new_col_name] > 0]
    
    
    return pvt_output
