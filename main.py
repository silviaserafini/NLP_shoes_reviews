from reviews_scraper import Reviews_scraper
from reviews_processor import Reviews_processor
from glove_trainer import Glove_trainer
#import pandas as pd
def main():
    #reviews_scraper = Reviews_scraper("raw_reviews_with_content2_df.pkl")
    #reviews_scraper.scrape()

    #reviews_processor = Reviews_processor("raw_reviews_with_content1_df.pkl", "raw_total_shoes_df1.pkl")
    #reviews_processor.process()

    glove_trainer = Glove_trainer( "raw_total_shoes_df.pkl", 'Glove_model.json', 'glove.6B/glove.6B.50d.txt')
    glove_trainer.train()
   
if __name__== "__main__":
    main()