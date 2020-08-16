from reviews_scraper import Reviews_scraper

def main():
    reviews_scraper = Reviews_scraper("raw_reviews_with_content1_df.pkl")
    reviews_scraper.scrape()
    

if __name__== "__main__":
    main()