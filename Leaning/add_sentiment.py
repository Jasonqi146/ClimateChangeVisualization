import sys

import pandas as pd
from os import path
import utils
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def add_sentiment_to_comments(pickle_startdate_enddate):
    # Opening file
    print("add_sentiment.py: Adding sentiment column to comments file with date: {}".format(pickle_startdate_enddate))
    comments = utils.get_comments_data(pickle_startdate_enddate)
    pd.set_option("max_columns", None)
    print(comments)

    # Add sentiment column
    analyzer = SentimentIntensityAnalyzer()
    comments["compound_sentiment"] = comments["body"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    print("add_sentiment.py: Finished adding sentiment column to comments file with date: {}".format(pickle_startdate_enddate))

    # Closing file
    base_path = path.dirname(__file__)
    data_filepath = path.abspath(path.join(base_path, "../data"))
    pickle_filepath = path.abspath(path.join(data_filepath, "comments", "with_sentiment", "{}_all_comments.pickle".format(pickle_startdate_enddate)))
    comments.to_pickle(pickle_filepath)
    print("add_sentiment.py: {} comments (now with sentiment column) successfully pickled!".format(pickle_startdate_enddate))
    print(comments)
    return comments


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Correct format: python add_sentiment.py 20201129_20210201")
    else:
        pickle_startdate_enddate = sys.argv[1]
        add_sentiment_to_comments(pickle_startdate_enddate=pickle_startdate_enddate)