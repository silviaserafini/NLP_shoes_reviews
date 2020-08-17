from utils import get_sentiment_analysis, main_sent, init_nltk
import pandas as pd
import pickle

class Reviews_processor:
    def __init__(self,load_path,save_path):
        self.load_path=load_path
        self.save_path=save_path
        self.reviews_df=pd.read_pickle(self.load_path)
        self.total_shoes_df = pd.DataFrame(columns=['brand','name','product_id','expert_id','id','score','type','content','sentiment','main_sentiment'])

    def process(self):
        init_nltk()
        print('extraction of the contents...')
        for i, product_id in enumerate(self.reviews_df['product_id']):
            print(f'processing product_id{product_id}...')
            shoe = self.create_shoe_df(product_id,i)
            shoe = self.shoe_df_processing(shoe)
            self.total_shoes_df = pd.concat([self.total_shoes_df,shoe])
            self.save()
            print('processing done!:-)')

    def create_shoe_df(self,product_id,i):
        #returns a dataframe with the reviews for the prduct 'product_id'
        reviews_list=self.reviews_df[self.reviews_df['product_id'] == product_id]['reviews'][i]#da sistemare:la lista si trova in i-esima pos
        
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
            shoe_dic['brand'].append(self.reviews_df['brand'])
            shoe_dic['name'].append(self.reviews_df['name'])
            shoe_dic['product_id'].append(product_id)
            shoe_dic['expert_id'].append(el['expert_id'])
            shoe_dic['id'].append(el['id'])
            shoe_dic['score'].append(el['score'])
            shoe_dic['type'].append(el['type'])
            shoe_dic['content'].append(el['content'])
            
        shoe_df=pd.DataFrame(shoe_dic)
        return shoe_df

    def shoe_df_processing(self, shoe_df):
        shoe_df['sentiment'] = shoe_df['content'].apply(get_sentiment_analysis)
        shoe_df['main_sentiment'] = shoe_df['sentiment'].apply(main_sent)
        return shoe_df

    def save(self):
        self.total_shoes_df.to_pickle(self.save_path) 
