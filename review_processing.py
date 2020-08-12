from utils import *
import pandas as pd
import pickle

reviews_df=pd.read_pickle("raw_reviews_with_content_df.pkl")
print('reading the database')

#explosion of the review_df, one row per reeview
total_shoes_df=pd.DataFrame(columns=['brand','name','product_id','expert_id','id','score','type','content','sentiment','main_sentiment'])
print('extraction of the contents...')
for i,product_id in enumerate(reviews_df['product_id']):
    shoe = create_shoe_df(product_id,reviews_df,i)
    shoe = shoe_df_processing(shoe, get_sentiment_analysis, main_sent)
    total_shoes_df=pd.concat([total_shoes_df,shoe])

'''#saving
total_shoes_df.to_pickle('raw_total_shoes_df.pkl') 
'''