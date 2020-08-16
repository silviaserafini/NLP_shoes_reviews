
import json
from bs4 import BeautifulSoup
import pandas as pd
import pickle
import requests
import time
from utils import identity

class Reviews_scraper:
    def __init__(self,path):
        self.path=path
        self.shoes_counter=0
        self.reviews_df=pd.DataFrame()
        self.api_url='https://api.runrepeat.com'

    def populate_database(self):
        print('populating database...')
        reviews={'name':[],'brand':[],'users_score':[],'score':[],'product_id':[],'brand_id':[],'SRP':[]}
        N=150
        MAX_API_RESULTS=30
        for i in range(1,N,MAX_API_RESULTS):
            if i+MAX_API_RESULTS<N:
                res=requests.get(f'{self.api_url}/get-documents?from={i}&size={i+ MAX_API_RESULTS}&filter[]=1&filter[]=6214&f_id=2&c_id=2&orderBy=recommended_rank')
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

    def get_shoe_reviews(self,shoe_code):
        print(f'extracting shoe ({shoe_code}) reviews...')
        MAX_N_PAGES = 20
        TOTAL_SHOES_NUMBER=2250
        page_n = 1
        shoe_reviews=[]
        self.shoes_counter+=1
        while page_n < MAX_N_PAGES :
            print('code=',shoe_code,'page_n=',page_n,'missing:',TOTAL_SHOES_NUMBER-self.shoes_counter)
            url = f'{self.api_url}/api/product/reviews/{shoe_code}/{page_n}'
            shoe_data = self.make_request(url)
            if len(shoe_data)>0:
                shoe_reviews.extend(shoe_data)
            else:
                break
            page_n+=1
        return shoe_reviews

    def make_request(self,url):
        print('processing', url)
        json_response=requests.get(url).json()
        time.sleep(1)
        return self.extractShoeData(json_response)
    
    def extractShoeData(self, page):
        review_list = []
        review_types = ["nativeReviews","nonNativeReviews"]
        try:
            list_of_list = map(lambda review_type: self.get_reviews_info_list('page', review_type), review_types)
            review_list = [el for sublist in list_of_list for el in sublist]
        except:
            pass
        return list(filter(lambda review_info : len(review_info)>0, review_list))

    def get_reviews_info_list(self, json_page, review_type, cleaning_func = identity):
        review_list=list(map(lambda json_review : self.get_review_info_from_json_review(json_review, cleaning_func), json_page[review_type]))
        review_list=list(filter(lambda review_info: len(review_info['content'])>0,review_list))
        return review_list

    def get_review_info_from_json_review(self,json_review, cleaning_func=identity):
        review_info = {
                    "id" : json_review["id"],
                    "score": json_review["score"],
                    "type": json_review["type"],
                    "expert_id": json_review["expert_id"],
                    "content": cleaning_func(json_review["content"])
                }
        return review_info

    def scrape(self):
        print('scraping...')
        reviews = self.populate_database()
        self.reviews_df = pd.DataFrame(reviews)
        #extration of the reviews text from json
        self.reviews_df['reviews']=self.reviews_df['product_id'].apply(self.get_shoe_reviews)
        self.save()
        print('done! :-)')

    def load(self):
        print('loading...')
        self.reviews_df=pd.read_pickle(self.path)

    def save(self):
        print('saving...')
        self.reviews_df.to_pickle(self.path) 


