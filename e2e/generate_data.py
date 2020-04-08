import glob

import pandas as pd

if __name__ == "__main__":
    corpus = []
    for article in glob.glob("nyt-comments/Articles*"):
        print("Processing:", article)
        dat_articles = pd.read_csv(article)
        corpus += dat_articles["snippet"].str.cat().split()
    words = list(set(corpus))
    dat = pd.DataFrame()
    dat["words"] = words
    dat.to_csv("words.csv", index=None)
