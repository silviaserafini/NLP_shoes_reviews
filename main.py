from reviews_scraper import Reviews_scraper
from reviews_processor import Reviews_processor
from glove_trainer import Glove_trainer
from TSHub_model_trainer import TSH_trainer
import os 
import sys
from argparse import ArgumentParser


def main():
    
    parser = ArgumentParser(description = "specify the arguments -s or -y if you want to scrape or train the model, otherwise type just insert the review text you want to predict the class")
    #optional flags arguments
    review=list(input('insert the review you want to predict the class:'))
    #parser.add_argument("-r","--review", help = "insert the review text you want to predict")
    parser.add_argument("-s","--scrape", help = "flag: scrape the shoes reviews from runrepeat.com", action = "store_true")
    parser.add_argument("-t","--train", help = "flag: train the model", action = "store_true")
    parser.add_argument("-m","--model", help = "specify the model: t=TSHubDFF, g=GloVeLSTM", default = "t")
    
    args = parser.parse_args()

    if args.scrape:
        reviews_scraper = Reviews_scraper("raw_reviews_with_content2_df.pkl")
        reviews_scraper.scrape()
        
    if args.train:
        reviews_processor = Reviews_processor("raw_reviews_with_content1_df.pkl", "raw_total_shoes_df1.pkl")
        reviews_processor.process()
        if args.model=='g':
            glove_trainer = Glove_trainer( "raw_total_shoes_df.pkl", 'Glove_model.h5', 'glove.6B/glove.6B.50d.txt')
            glove_trainer.train()
        else:
            tsh_trainer = TSH_trainer("raw_total_shoes_df.pkl", 'THUB_model.h5','review_classifier_TFH')
            tsh_trainer.train()
    
    if review:
        if args.model=='g':
            glove_trainer = Glove_trainer( "raw_total_shoes_df.pkl", 'Glove_model.h5', 'glove.6B/glove.6B.50d.txt')
            glove_trainer.predict(review)
        else:
            tsh_trainer = TSH_trainer("raw_total_shoes_df.pkl", 'THUB_model.h5','review_classifier_TFH')
            tsh_trainer.predict(review)

if __name__== "__main__":
    main()