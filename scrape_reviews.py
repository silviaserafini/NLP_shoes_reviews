
import json
from bs4 import BeautifulSoup
import html2text
import re
import string
import pandas as pd
import numpy as np
from utils import populate_database, get_shoe_reviews
import pickle


#population of the database
reviews = populate_database()

reviews_df = pd.DataFrame(reviews)

#extration of the reviews text from json
count=0
reviews_df['reviews']=reviews_df['product_id'].apply(get_shoe_reviews)

#saving the database
#reviews_df.to_pickle('/content/drive/My Drive/Colab Notebooks/raw_reviews_with_content_df.pkl') 