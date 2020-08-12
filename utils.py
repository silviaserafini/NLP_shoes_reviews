from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import keras.backend as K
import time
import texthero as hero
import pickle
import nltk
import html2text
import re
import string
import pandas as pd
import numpy as np
import nltk.sentiment.vader
import requests
nltk.download('vader_lexicon')

shoes_counter=0
def make_request(url):
  print('processing', url)
  json_response=requests.get(url).json()
  time.sleep(1)
  return extractShoeData(json_response)

def get_shoe_reviews(code):
    MAX_N_PAGES = 20
    page_n = 1
    data=[]
    global shoes_counter
    shoes_counter+=1
    TOTAL_SHOES_NUMBER=2250
    while page_n < MAX_N_PAGES :
        print('code=',code,'page_n=',page_n,'missing:',TOTAL_SHOES_NUMBER-shoes_counter)
        url = f'https://api.runrepeat.com/api/product/reviews/{code}/{page_n}'
        shoe_data = make_request(url)
        if len(shoe_data)>0:
            data.extend(shoe_data)
        else:
            break
        page_n+=1
       
    return data


def populate_database():
    reviews={'name':[],'brand':[],'users_score':[],'score':[],'product_id':[],'brand_id':[],'SRP':[]}
    N=150
    MAX_API_RESULTS=30
    for i in range(1,N,MAX_API_RESULTS):
        if i+MAX_API_RESULTS<N:
            res=requests.get(f'https://api.runrepeat.com/get-documents?from={i}&size={i+30}&filter[]=1&filter[]=6214&f_id=2&c_id=2&orderBy=recommended_rank')
            time.sleep(3)
            for i in range(0,N):
                try:
                    reviews['users_score'].append(res.json()['products'][i]['users_score'])
                    reviews['score'].append( res.json()['products'][i]['score'])
                    reviews['name'].append( res.json()['products'][i]['slug'])
                    reviews['brand'].append( res.json()['products'][i]['brand']['name'])
                    reviews['product_id'].append( int(res.json()['products'][i]['product_id']))
                    reviews['brand_id'].append( int(res.json()['products'][i]['brand']['id']))
                    reviews['SRP'].append( res.json()['products'][i]['msrp_formatted'])
                except:
                    continue
    return reviews

def remove_text_inside_brackets(text, brackets="()[]"):
    count = [0] * (len(brackets) // 2) # count open/close brackets
    saved_chars = []
    for character in text:
        for i, b in enumerate(brackets):
            if character == b: # found bracket
                kind, is_close = divmod(i, 2)
                count[kind] += (-1)**is_close # `+1`: open, `-1`: close
                if count[kind] < 0: # unbalanced bracket
                    count[kind] = 0  # keep it
                else:  # found bracket to remove
                    break
        else: # character is not a [balanced] bracket
            if not any(count): # outside brackets
                saved_chars.append(character)
    return ''.join(saved_chars)

    # return the wordnet object value corresponding to the POS tag

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def text_clean(text):
    if text==None:
        return ''
    text = text.lower()
    text = html2text.html2text(text)
    text = ' '.join(text.split())#remove multiple spaces
    text = remove_text_inside_brackets(text)
    text = text.replace('\n',' ')
    text = text.replace('\'','')
    text = text.replace('/',' ')
    text = text.replace('#','')
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    #stop = stopwords.words('english')
    #text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    #text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return text
    
def identity(f):
  return f

def get_review_info_from_json_review(json_review, cleaning_func=identity):
  review_info = {
            "id" : json_review["id"],
            "score": json_review["score"],
            "type": json_review["type"],
            "expert_id": json_review["expert_id"],
            "content": cleaning_func(json_review["content"])
           }
  return review_info

def get_reviews_info_list(json_page, review_type, cleaning_func=identity):
    review_list=list(map(lambda json_review : get_review_info_from_json_review(json_review, cleaning_func), json_page[review_type]))
    review_list=list(filter(lambda review_info: len(review_info['content'])>0,review_list))
    return review_list

        
def extractShoeData(page):
    review_list = []
    review_types = ["nativeReviews","nonNativeReviews"]
    try:
        list_of_list = map(lambda review_type: get_reviews_info_list('page', review_type), review_types)
        review_list = [el for sublist in list_of_list for el in sublist]
    except:
        pass
    return list(filter(lambda review_info : len(review_info)>0, review_list))

def create_shoe_df(product_id,df,i):
    #returns a dataframe with the reviews for the prduct 'product_id'
    reviews_list=df[df['product_id']==product_id]['reviews'][i]
    
    shoe_dic={
        'brand':[],
        'name':[],
        'product_id':[],
        'expert_id': [],
        'id': [],
        'score': [],
        'type': [],
        'content':[]
    }
    for el in reviews_list:
        shoe_dic['brand'].append(df['brand'])
        shoe_dic['name'].append(df['name'])
        shoe_dic['product_id'].append(product_id)
        shoe_dic['expert_id'].append(el['expert_id'])
        shoe_dic['id'].append(el['id'])
        shoe_dic['score'].append(el['score'])
        shoe_dic['type'].append(el['type'])
        shoe_dic['content'].append(el['content'])
    shoe_df=pd.DataFrame(shoe_dic)
    return shoe_df

def get_sentiment_analysis(review):
    sentiment=SentimentIntensityAnalyzer().polarity_scores(review)
    return sentiment

def main_sent(sentiment):
    return sentiment['compound']

def shoe_df_processing(shoe_df, get_sentiment_analysis, main_sent):
    shoe_df['sentiment'] = shoe_df['content'].apply(get_sentiment_analysis)
    shoe_df['main_sentiment'] = shoe_df['sentiment'].apply(main_sent)
    return shoe_df

# range bins
def assign_label(x):
    if x>90:
        return'top'
    if x>60 and x<=90:
        return 'avarage'
    if x<=60:
        return 'low'
    
def precision(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) 
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1))) 
    precision = true_positives / (predicted_positives + K.epsilon()) 
    return precision

def recall(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) 
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1))) 
    recall = true_positives / (possible_positives + K.epsilon()) 
    return recall 

def encode_labels(x):
    label={'top':0,'avarage':1, 'low':2}
    return label[x]    
