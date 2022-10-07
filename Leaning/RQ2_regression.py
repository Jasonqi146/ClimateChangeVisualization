import pickle
import sys
from os import path, listdir
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

id_and_subreddit = [
    'id',
    'subreddit',
    'parent_id',
    'link_id'
]
self_values = [
    'comment_len',
    'aggregate_civility',
    'compound_sentiment',
    'civility',
    'vulgarity',
    'namecalling',
    'stereotype',
    'demeaning'
]
parent_values = [
    'parent_aggregate_civility',
    'parent_sentiment',
    'parent_civility',
    'parent_vulgarity',
    'parent_namecalling',
    'parent_stereotype',
    'parent_demeaning',
    'parent_post_divisiveness'
]
previous_comment_values = [
    'average_aggregate_civility_of_previous_comments',
    'average_sentiment_of_previous_comments',
    'average_civility_of_previous_comments',
    'average_vulgarity_of_previous_comments',
    'average_namecalling_of_previous_comments',
    'average_stereotype_of_previous_comments',
    'average_demeaning_of_previous_comments',
    'number_of_previous_comments'
]
child_comment_values = [
    'average_aggregate_civility_of_child_comments',
    'average_sentiment_of_child_comments',
    'average_civility_of_child_comments',
    'average_vulgarity_of_child_comments',
    'average_namecalling_of_child_comments',
    'average_stereotype_of_child_comments',
    'average_demeaning_of_child_comments',
    'number_of_direct_child_comments'
]
political_leanings = [
    'percentage_of_liberal_authors_in_previous_comments',
    'percentage_of_conservative_authors_in_previous_comments',
    'percentage_of_liberal_authors_in_thread',
    'percentage_of_conservative_authors_in_thread',
    'subreddit_political_leaning'
]
liwc_columns = [
    'we', 'ipron', 'they', 'home', 'quant', 'article', 'you', 'see', 'death', 'motion', 'achiev', 'discrep',
    'family', 'percept', 'conj', 'health', 'sexual', 'nonflu', 'present', 'feel', 'inhib', 'bio', 'social',
    'leisure', 'friends', 'future', 'prep', 'tentat', 'negemo', 'insight', 'excl', 'negate', 'filler',
    'pronoun', 'relig', 'posemo', 'time', 'hear', 'money', 'assent', 'anx', 'affect', 'sad', 'human',
    'space', 'vbs', 'incl', 'certain', 'auxvb', 'funct', 'cause', 'ppron', 'work', 'adverbs', 'shehe', 'i',
    'relativ', 'anger', 'numbers', 'body', 'swear', 'cogmech', 'ingest', 'past'
]


def calc_vif(X):
    # Calculating VIF
    print(f"Calculating VIF for IVs: {X.columns}")
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif


def new_regression(comments):
    independent_variables = [
        'average_sentiment_of_previous_comments',
        'average_civility_of_previous_comments',
        # 'average_vulgarity_of_previous_comments',
        # 'average_namecalling_of_previous_comments',
        # 'average_stereotype_of_previous_comments',
        # 'average_demeaning_of_previous_comments',

        'compound_sentiment',
        'civility',
        # 'vulgarity',
        # 'namecalling',
        # 'stereotype',
        # 'demeaning',

        'percentage_of_liberal_authors_in_thread',
        'percentage_of_conservative_authors_in_thread',
        'percentage_of_liberal_authors_in_previous_comments',
        'percentage_of_conservative_authors_in_previous_comments',

        'number_of_previous_comments',
        'parent_post_divisiveness'
    ]

    # Add LIWC vars
    interested_liwc_vars = ["social", "posemo", "negemo", "anger", "swear", "sad", "we", "negate", "you", "i"]
    independent_variables += interested_liwc_vars

    dependent_variables = [
        'average_sentiment_of_child_comments',
        'average_civility_of_child_comments',
        # 'average_vulgarity_of_child_comments',
        # 'average_namecalling_of_child_comments',
        # 'average_stereotype_of_child_comments',
        # 'average_demeaning_of_child_comments',

        'number_of_direct_child_comments'
    ]

    print("Regression: Normalizing data")
    # Need to normalize LIWC vars
    for cat in interested_liwc_vars:
        comments[cat] = comments[cat]/comments['comment_len']

    # Need to normalize number of previous comments and number of child comments
    # total_number_of_comments_per_thread =

    # Need to normalize other columns
    for column in independent_variables + dependent_variables:
        print(f"Normalizing {column}")
        comments[column] = (comments[column] - comments[column].min()) / (
                comments[column].max() - comments[column].min())

    print("=======> Train/test split")
    train, test = train_test_split(comments, test_size=0.2)
    print("Train length: {}, test length: {}".format(len(train), len(test)))

    print(calc_vif(train[independent_variables]))

    X_train = train[independent_variables]
    X_test = test[independent_variables]

    sig1_strs = []
    sig2_strs = []
    sig3_strs = []

    print(f"Regression with IVs {independent_variables}")
    print(f"Regression with DVs {dependent_variables}")
    for dv in dependent_variables:
        print(f"Fitting {dv}")
        X = sm.add_constant(X_train)
        y_train = train[dv]
        y_test = test[dv]
        model = sm.OLS(y_train, X).fit()
        print(f"=======> P-values and coefs for DV {dv}")
        for i in range(len(independent_variables)):
            print("==> IV {} has a p value of {}, and coef of {} with DV {}".format(independent_variables[i],
                                                                                    model.pvalues[i], model.params[i],
                                                                                    dv))
            if model.pvalues[i] < 0.001:
                sig1_strs.append(f"*** Statistically significant p-value < 0.001!: IV: {independent_variables[i]}, DV: {dv}, coef: {model.params[i]}, p-val: {model.pvalues[i]}")
            elif 0.001 <= model.pvalues[i] < 0.01:
                sig2_strs.append(f"** Statistically significant p-value < 0.01!: IV: {independent_variables[i]}, DV: {dv}, coef: {model.params[i]}, p-val: {model.pvalues[i]}")
            elif 0.01 <= model.pvalues[i] < 0.05:
                sig3_strs.append(f"* Statistically significant p-value < 0.05!: IV: {independent_variables[i]}, DV: {dv}, coef: {model.params[i]}, p-val: {model.pvalues[i]}")
    print(":::> Sig 1 strs")
    for sig1 in sig1_strs:
        print(sig1)
    print(":::> Sig 2 strs")
    for sig2 in sig2_strs:
        print(sig2)
    print("::: Sig 3 strs")
    for sig3 in sig3_strs:
        print(sig3)

def regression_model_final(comments):
    """
    - Independent Variables
        - Incivility level of the previous one comment
        - Average incivility of all previous comments
        - Whether personal attack is present in previous comment
        - Divisiveness of the topic of the original post
    - Control Variables
        - Subreddit that the conversation is in
        - Percentage of liberals/conservatives in the thread
    - Dependent Variables
        - Level of incivility of the subsequent conversation (?)
            - Whether we want to use the next message or the aggregate value of the entire conversation afterwards
        - Strong negative/positive sentiments of the subsequent conversation (LIWC, VADER)
    """
    print("=======> Overall comments")
    pd.set_option("max_columns", None)
    print(comments)

    print("=======> Statistics")
    print(comments.describe())

    # old_regression(train, test)
    new_regression(comments)


def get_relevant_columns(comments, filepath):
    print("Narrowing columns")
    pd.set_option("max_columns", None)
    print(comments)

    print("Regression: NaN in number_of_direct_child_comments and number_of_previous_comments means 0")
    comments["number_of_direct_child_comments"] = comments["number_of_direct_child_comments"].fillna(0)
    comments["number_of_previous_comments"] = comments["number_of_previous_comments"].fillna(0)

    # interested_columns = id_and_subreddit + self_values + parent_values + previous_comment_values + child_comment_values + political_leanings + liwc_columns
    interested_columns = id_and_subreddit + self_values + parent_values + previous_comment_values + child_comment_values + political_leanings + ["social", "posemo", "negemo", "anger", "swear", "sad", "we", "negate", "you", "i"]
    # Uncomment these lines if we need to write the files
    # pickle_narrowed_columns_with_liwc_filepath = path.abspath(path.join(data_filepath, "comments", "more_narrowed_columns_new", filepath))
    # comments[interested_columns].to_pickle(pickle_narrowed_columns_with_liwc_filepath)
    return comments[interested_columns]


if __name__ == '__main__':
    # Read every file in directory
    # TODO: This isn't actually right, it reads just the "comments/final" directory rather than user input
    if len(sys.argv) != 1:
        print("Correct usage: python RQ2_regression.py filepath/to/directory/of/pickles")
    else:
        comments_dataframes = []
        base_path = path.dirname(__file__)
        data_filepath = path.abspath(path.join(base_path, "../data"))
        comments_directory = path.abspath(path.join(data_filepath, "comments", "more_narrowed_columns_new"))
        for filepath in listdir(comments_directory):
            if filepath.endswith(".pickle"):
                # 'final' directory just has the columns below that are used in regression
                # 'all_columns' directory has all of the columns
                with open(path.abspath(path.join(data_filepath, "comments", "more_narrowed_columns_new", filepath)),
                          "rb") as input_file:
                    print("=======> Regression: Retrieving {} comments...".format(filepath))
                    comments = pickle.load(input_file)
                    print(len(comments.columns))
                    print("comment_len in comments:", 'comment_len' in comments)
                    """
                    Each comments file should have:
                    Index(['created_utc', 'parent_id', 'link_id', 'id', 'author', 'body',
                           'subreddit', 'link', 'score', 'author_flair_text', 'vulgarity',
                           'civility', 'namecalling', 'stereotype', 'demeaning', 'comment_len',
                           'compound_sentiment', 'parent_civility', 'parent_vulgarity',
                           'parent_demeaning', 'parent_stereotype', 'parent_namecalling',
                           'parent_author', 'path_id', 'path_civility', 'path_vulgarity',
                           'path_namecalling', 'path_stereotype', 'path_demeaning',
                           'average_civility_of_previous_comments',
                           'average_vulgarity_of_previous_comments',
                           'average_namecalling_of_previous_comments',
                           'average_stereotype_of_previous_comments',
                           'average_demeaning_of_previous_comments', 'number_of_previous_comments',
                           'author_political_leaning', 'path_author_political_leaning',
                           'percentage_of_liberal_authors_in_previous_comments',
                           'percentage_of_conservative_authors_in_previous_comments',
                           'percentage_of_liberal_authors_in_thread',
                           'percentage_of_conservative_authors_in_thread',
                           'average_sentiment_of_child_comments',
                           'average_civility_of_child_comments',
                           'average_vulgarity_of_child_comments',
                           'average_namecalling_of_child_comments',
                           'average_stereotype_of_child_comments',
                           'average_demeaning_of_child_comments', 'aggregate_civility',
                           'parent_aggregate_civility', 'path_sentiment',
                           'average_sentiment_of_previous_comments',
                           'average_aggregate_civility_of_previous_comments',
                           'average_aggregate_civility_of_child_comments', 'parent_sentiment', 
                           'parent_post_divisiveness'],
                          dtype='object')
                    """
                    comments = get_relevant_columns(comments, filepath)
                    comments_dataframes.append(comments.reset_index(drop=True))
        print("Regression: Concatenating comments dataframes...")
        overall_comments = pd.concat(comments_dataframes, axis=0, ignore_index=True)

        print("Regression: {} number of comments before dropping nan rows".format(len(overall_comments)))
        overall_comments = overall_comments[~overall_comments.isnull().any(axis=1)]
        print("Regression: {} number of comments after dropping nan rows".format(len(overall_comments)))

        print("Regression: Finished concatenating comments dataframes, of size: {} rows x {} columns".format(
            len(overall_comments), len(overall_comments.columns)))
        # basic_plots(overall_comments)
        regression_model_final(overall_comments)
