from os import path
import pickle
import pandas as pd


def get_comments_data(pickle_startdate_enddate):
    # 20201129_20210201
    base_path = path.dirname(__file__)
    data_filepath = path.abspath(path.join(base_path, "../data"))
    comments_filepath = path.abspath(path.join(data_filepath, "comments", "{}_all_comments.pickle".format(pickle_startdate_enddate)))
    with open(comments_filepath, "rb") as input_file:
        print("Utils: Retrieving {} comments...".format(pickle_startdate_enddate))
        comments = pickle.load(input_file)

        # TODO: Am I allowed to drop comments where author is AutoModerator?
        comments = comments[comments["author"] != "AutoModerator"]
        print("Utils: Retrieved {} {} comments".format(pickle_startdate_enddate, len(comments)))
        #
        # Cleaning comments
        # comments = comments[(comments.author != "[deleted]") & (comments.author != "[removed]") & (
        #         comments.body != "[deleted]") & (comments.body != "[removed]")]
        # print("After cleaning, we have {} comments".format(len(comments)))
        return comments


def get_comments_data_at_level(pickle_startdate_enddate, num):
    # 20201129_20210201
    base_path = path.dirname(__file__)
    data_filepath = path.abspath(path.join(base_path, "../data"))
    comments_filepath = path.abspath(
        path.join(data_filepath, "comments", "{}_all_comments_RQ2_{}.pickle".format(pickle_startdate_enddate, num)))
    print("Utils: Retrieving {} comments at level {}...".format(pickle_startdate_enddate, num))
    comments = pd.read_pickle(comments_filepath)
    print("Utils: Retrieved {} {} comments at level {}".format(pickle_startdate_enddate, len(comments), num))
    return comments


def get_posts_data(pickle_startdate_enddate):
    # 20201129_20210201
    base_path = path.dirname(__file__)
    data_filepath = path.abspath(path.join(base_path, "../data"))
    posts_filepath = path.abspath(path.join(data_filepath, "posts", "{}_all_posts.pickle".format(pickle_startdate_enddate)))
    with open(posts_filepath, "rb") as input_file:
        print("Utils: Retrieving {} posts...".format(pickle_startdate_enddate))
        posts = pickle.load(input_file)
        print("Utils: Retrieved {} {} posts".format(pickle_startdate_enddate, len(posts)))
        #
        # Cleaning posts
        # posts = posts[(posts.author != "[deleted]") & (posts.author != "[removed]") & (posts.title != "[deleted]") & (
        #         posts.title != "[removed]")]
        # print("After cleaning, we have {} posts".format(len(posts)))
        return posts


def get_posts_divisiveness_labels():
    base_path = path.dirname(__file__)
    data_filepath = path.abspath(path.join(base_path, "../data"))
    posts_filepath = path.abspath(path.join(data_filepath, "posts", "divisiveness", "all_posts.pickle"))
    with open(posts_filepath, "rb") as input_file:
        print("Utils: Retrieving {} posts...".format("divisiveness"))
        posts = pickle.load(input_file)
        print("Utils: Retrieved {} {} posts".format("divisiveness", len(posts)))
        return posts



def get_left_and_right_users(comments):
    """
    :param comments: comments_data from utils
    :return: left users, right users
    """
    # Get user comment counts
    all_users = comments.groupby('author')['id'].count().reset_index()
    all_users.columns = ['author', 'total_count']

    # Get comments in left and right subreddits, grouped by author
    red, blue = get_political_subreddits("red"), get_political_subreddits("blue")
    right_comment_groups = comments[comments.subreddit.isin(red)].groupby('author')
    left_comment_groups = comments[comments.subreddit.isin(blue)].groupby('author')

    # Get comment counts for each user in left and right subreddits,
    # mapping usernames (author) to comment counts (id)
    # dict(left_comment_groups["id"].count())
    # e.g. {'---gabers---': 4, '---stargazer---': 1, '--0IIIIIII0--': 1, '--Antitheist--': 1, '--ElonMusk': 3}
    all_users['left_count'] = all_users['author'].map(dict(left_comment_groups['id'].count())).fillna(0)
    all_users['right_count'] = all_users['author'].map(dict(right_comment_groups['id'].count())).fillna(0)

    # Get average score for each user's comments in left and right subreddits,
    # mapping usernames (author) to comment scores (score)
    # dict(left_comment_groups["score"].mean())
    # e.g. {'---gabers---': -0.5, '---stargazer---': 1.0, '--0IIIIIII0--': 1.0, '--Antitheist--': 1.0, '--ElonMusk': 1.6666666666666667}
    all_users['left_score'] = all_users['author'].map(dict(left_comment_groups['score'].mean())).fillna(0)
    all_users['right_score'] = all_users['author'].map(dict(right_comment_groups['score'].mean())).fillna(0)

    # Group users into left or right, based on:
    # if their left/right comment count is greater than their right/left comment count
    # if their left/right score is greater than 1
    # if their left/right score is greater than their right/left score
    left_users = all_users[(all_users['left_count'] > all_users['right_count']) & (all_users['left_score'] > 1) & (
            all_users['left_score'] > all_users['right_score'])]
    right_users = all_users[(all_users['left_count'] < all_users['right_count']) & (all_users['right_score'] > 1) & (
            all_users['left_score'] < all_users['right_score'])]

    """
    leftusers
                       author  total_count  left_count  right_count  left_score  right_score
    53             --ElonMusk           32         3.0          0.0    1.666667          0.0
    58      --GrinAndBearIt--           17         7.0          0.0    1.571429          0.0
    108           --_-_o_-_--            8         3.0          1.0    4.333333        -12.0
    143      --theriverstyx--            3         3.0          0.0    1.333333          0.0
    148                 -0-O-           27         9.0          0.0    1.777778          0.0
    ...                   ...          ...         ...          ...         ...          ...
    618369              zyygh            2         2.0          0.0  110.000000          0.0
    618393     zzcheeseballzz            3         1.0          0.0    4.000000          0.0
    618439          zzzaacchh           46         1.0          0.0    5.000000          0.0
    618442            zzzeoww            1         1.0          0.0   15.000000          0.0
    618450           zzztoken          105       105.0          0.0    2.295238          0.0
    
    [13143 rows x 6 columns]
    """
    return left_users, right_users


def get_political_subreddits(leaning):
    # leaning = "red" or "blue"
    # red = {'AskThe_Donald', 'Conservative', 'ConservativesOnly', 'donaldtrump'}
    #
    # blue = {'BlueMidterm2018', 'hillaryclinton', 'JoeBiden', 'OurPresident', 'progressive',
    #         'SandersForPresident', 'The_Mueller', 'VoteBlue', 'VoteDEM'}
    #
    # all_subreddits = {'AskThe_Donald', 'Ask_Politics', 'BlueMidterm2018',
    #                   'Conservative', 'ConservativesOnly', 'donaldtrump',
    #                   'hillaryclinton', 'JoeBiden', 'Libertarian',
    #                   'NeutralPolitics', 'OurPresident', 'PoliticalDiscussion',
    #                   'Political_Revolution', 'politics', 'progressive',
    #                   'SandersForPresident', 'The_Mueller', 'uspolitics',
    #                   'VoteBlue', 'VoteDEM'}

    # 12/5: From Yujia
    # blue = {'BlueMidterm2018', 'politics', 'JoeBiden', 'Libertarian', 'OurPresident', 'PoliticalDiscussion', 'Political_Revolution', 'SandersForPresident', 'VoteBlue', 'hillaryclinton', 'politics', 'progressive'}
    # red = {'AskThe_Donald', 'Conservative', 'ConservativesOnly', 'The_Mueller', 'donaldtrump'}

    # 12/22: My adjustments
    red = {'AskThe_Donald', 'Conservative', 'ConservativesOnly', 'donaldtrump'}
    blue = {'BlueMidterm2018', 'politics', 'JoeBiden', 'Libertarian', 'OurPresident', 'PoliticalDiscussion',
            'Political_Revolution', 'SandersForPresident', 'VoteBlue', 'hillaryclinton', 'politics', 'progressive',
            'The_Mueller'}

    if leaning == "red":
        return red
    else:
        return blue


def get_authors_leaning():
    base_path = path.dirname(__file__)
    data_filepath = path.abspath(path.join(base_path, "../data"))
    comments_filepath = path.abspath(path.join(data_filepath, "authors_with_leaning.pickle"))
    with open(comments_filepath, "rb") as input_file:
        comments = pickle.load(input_file)
        return comments