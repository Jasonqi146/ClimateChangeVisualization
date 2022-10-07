import sys

import utils
import pandas as pd
import numpy as np
from os import path
from add_liwc import get_liwc_categories

"""
RQ2: How does the emergence of incivility affect the subsequent conversation? Does it incite more incivility?

- Look at local context: full conversations under a thread
- For BERT model, add the context into the classification and see if a significant boost can be made (if no, stick with the model without context)
- Regression Model
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
- Test hypothesis
    - Hypothesis 1: When discussions are civil, participants are more likely to subsequently exhibit civil behaviors and indicate that they are willing to continue their participation in the discussion. (Han and Brazeal, 2015)
    - Hypothesis 2: When discussions are uncivil, participants are more likely to have strong sentiments like anger (Gervais, 2015)
"""


def regression_model_1(comments, posts, pickle_startdate_enddate):
    """
    How does the emergence of incivility affect the subsequent conversation? Does it incite more incivility?
    :param comments: comments_data from utils
    :return:
    """

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
    print("RQ2: 0) Pre-processing")
    base_path = path.dirname(__file__)
    data_filepath = path.abspath(path.join(base_path, "../data"))

    # Prefix all comment ids with t1_ to make mapping between parent ids and comment ids easier
    comments["id"] = "t1_" + comments["id"].astype(str)
    posts["id"] = "t3_" + posts["id"].astype(str)

    # Create aggregate column
    comments["aggregate_civility"] = comments["vulgarity"] + comments["demeaning"] + comments["stereotype"] + comments[
        "namecalling"]

    print(
        "RQ2: 1) Incivility level of the previous one comment: 'parent_civility' column (or vulgarity, demeaning, namecalling, stereotype)")

    # Map comment ids to civility value {"t1_glit0la" : 0.997876}
    comments_id_sentiment_dict = comments.set_index("id").compound_sentiment.to_dict()
    comments_id_civility_dict = comments.set_index("id").civility.to_dict()
    comments_id_vulgarity_dict = comments.set_index("id").vulgarity.to_dict()
    comments_id_demeaning_dict = comments.set_index("id").demeaning.to_dict()
    comments_id_stereotype_dict = comments.set_index("id").stereotype.to_dict()
    comments_id_namecalling_dict = comments.set_index("id").namecalling.to_dict()
    # Map comment ids to author {"t1_glit0la" : "user1"}
    comments_id_author_dict = comments.set_index("id").author.to_dict()
    # Map post ids to author {"t3_glit0la" : "user2"}
    posts_id_author_dict = posts.set_index("id").author.to_dict()

    # Get parent comment civility via the parent_ids mapped to civility
    comments["parent_sentiment"] = comments["parent_id"].map(comments_id_sentiment_dict)
    comments["parent_civility"] = comments["parent_id"].map(comments_id_civility_dict)
    comments["parent_vulgarity"] = comments["parent_id"].map(comments_id_vulgarity_dict)
    comments["parent_demeaning"] = comments["parent_id"].map(comments_id_demeaning_dict)
    comments["parent_stereotype"] = comments["parent_id"].map(comments_id_stereotype_dict)
    comments["parent_namecalling"] = comments["parent_id"].map(comments_id_namecalling_dict)
    comments["parent_aggregate_civility"] = comments["parent_vulgarity"] + comments["parent_demeaning"] + comments[
        "parent_stereotype"] + comments["parent_namecalling"]
    # Get parent comment/post author via the parent_ids mapped to authors
    comments["parent_author"] = comments["parent_id"].map(comments_id_author_dict)
    comments["parent_author"] = comments["parent_author"].fillna(comments["parent_id"].map(posts_id_author_dict))

    pickle_RQ2_1_filepath = path.abspath(
        path.join(data_filepath, "comments", "{}_all_comments_RQ2_1.pickle".format(pickle_startdate_enddate)))
    comments.to_pickle(pickle_RQ2_1_filepath)
    pd.set_option("max_columns", None)
    print(comments)
    print("RQ2_1 successfully pickled!")
    return comments


def regression_model_2(comments, pickle_startdate_enddate):
    print("RQ2: 2) Average incivility of all previous comments")
    base_path = path.dirname(__file__)
    data_filepath = path.abspath(path.join(base_path, "../data"))

    # We need to map comment id to parent id to parent id to parent id...
    # until the parent id is t3_ (a post, not a comment)

    # https://stackoverflow.com/questions/58821528/get-node-ancestors-in-a-pandas-dataframe

    comments_id_parent_id_dict = comments.set_index("id").parent_id.to_dict()
    # Map comment ids to civility value {"t1_glit0la" : 0.997876}
    comments_id_sentiment_dict = comments.set_index("id").compound_sentiment.to_dict()
    comments_id_civility_dict = comments.set_index("id").civility.to_dict()
    comments_id_vulgarity_dict = comments.set_index("id").vulgarity.to_dict()
    comments_id_namecalling_dict = comments.set_index("id").namecalling.to_dict()
    comments_id_stereotype_dict = comments.set_index("id").stereotype.to_dict()
    comments_id_demeaning_dict = comments.set_index("id").demeaning.to_dict()

    def get_parent_id(anc):
        anc = [anc] if not isinstance(anc, list) else anc
        if anc[-1].startswith("t3_"):
            return anc
        else:
            if anc[-1] in comments_id_parent_id_dict:
                parent = get_parent_id([comments_id_parent_id_dict[anc[-1]]])
                anc += parent
            return anc

    print(
        "RQ2: 2) a) Getting list of parent IDs for each comment, will include itself and parent post ('path_id' column)")
    comments["path_id"] = comments.id.apply(get_parent_id)  # includes language id

    print("RQ2: 2) b) Getting list of parent civilities for each comment ('path_civility' column)")
    comments["path_sentiment"] = comments.apply(lambda x: [comments_id_sentiment_dict[id_] for id_ in x.path_id
                                                           if (not (id_ == x.id or id_.startswith("t3_")) and (
                id_ in comments_id_civility_dict))], axis=1)
    comments["path_civility"] = comments.apply(lambda x: [comments_id_civility_dict[id_] for id_ in x.path_id
                                                          if (not (id_ == x.id or id_.startswith("t3_")) and (
                id_ in comments_id_civility_dict))], axis=1)
    comments["path_vulgarity"] = comments.apply(lambda x: [comments_id_vulgarity_dict[id_] for id_ in x.path_id
                                                           if (not (id_ == x.id or id_.startswith("t3_")) and (
                id_ in comments_id_civility_dict))], axis=1)
    comments["path_namecalling"] = comments.apply(lambda x: [comments_id_namecalling_dict[id_] for id_ in x.path_id
                                                             if (not (id_ == x.id or id_.startswith("t3_")) and (
                id_ in comments_id_civility_dict))], axis=1)
    comments["path_stereotype"] = comments.apply(lambda x: [comments_id_stereotype_dict[id_] for id_ in x.path_id
                                                            if (not (id_ == x.id or id_.startswith("t3_")) and (
                id_ in comments_id_civility_dict))], axis=1)
    comments["path_demeaning"] = comments.apply(lambda x: [comments_id_demeaning_dict[id_] for id_ in x.path_id
                                                           if (not (id_ == x.id or id_.startswith("t3_")) and (
                id_ in comments_id_civility_dict))], axis=1)

    print(
        "RQ2: 2) c) Getting average civility of parent civilities for each comment, includes only previous comments and not parent post ('average_civility_of_previous_comments' column)")
    comments["average_sentiment_of_previous_comments"] = comments["path_sentiment"].map(lambda x: np.array(x).mean())
    comments["average_civility_of_previous_comments"] = comments["path_civility"].map(lambda x: np.array(x).mean())
    comments["average_vulgarity_of_previous_comments"] = comments["path_vulgarity"].map(lambda x: np.array(x).mean())
    comments["average_namecalling_of_previous_comments"] = comments["path_namecalling"].map(
        lambda x: np.array(x).mean())
    comments["average_stereotype_of_previous_comments"] = comments["path_stereotype"].map(lambda x: np.array(x).mean())
    comments["average_demeaning_of_previous_comments"] = comments["path_demeaning"].map(lambda x: np.array(x).mean())

    comments["average_aggregate_civility_of_previous_comments"] = comments["average_vulgarity_of_previous_comments"] + \
                                                                  comments["average_namecalling_of_previous_comments"] + \
                                                                  comments["average_stereotype_of_previous_comments"] + \
                                                                  comments["average_demeaning_of_previous_comments"]

    print("RQ2: 2) d) Getting number of previous comments for each comment ('number_of_previous_comments' column)")
    comments["number_of_previous_comments"] = comments["path_civility"].map(lambda x: len(x))

    pickle_RQ2_2_filepath = path.abspath(
        path.join(data_filepath, "comments", "{}_all_comments_RQ2_2.pickle".format(pickle_startdate_enddate)))
    comments.to_pickle(pickle_RQ2_2_filepath)
    pd.set_option("max_columns", None)
    print(comments)
    print("RQ2_2 successfully pickled!")
    return comments


def regression_model_5(comments, pickle_startdate_enddate):
    print("RQ2: 5) Percentage of liberals/conservatives in the thread")
    base_path = path.dirname(__file__)
    data_filepath = path.abspath(path.join(base_path, "../data"))

    # Let's build out a new dataframe and then dictionary with percentage of liberals/conservatives in each thread
    left_users, right_users = utils.get_left_and_right_users(comments)
    # {"username1": "left", "username2": "right"}
    author_political_leaning_dict = {**dict(zip(left_users.author, ["left"] * len(left_users.author))),
                                     **dict(zip(right_users.author, ["right"] * len(right_users.author)))}

    # TODO: DELETE THIS DROPPING, JUST RE-ADJUSTING
    comments = comments.drop('author_political_leaning', 1)
    comments = comments.drop('path_author_political_leaning', 1)
    comments = comments.drop('percentage_of_liberal_authors_in_previous_comments', 1)
    comments = comments.drop('percentage_of_conservative_authors_in_previous_comments', 1)
    comments = comments.drop('percentage_of_liberal_authors_in_thread', 1)
    comments = comments.drop('percentage_of_conservative_authors_in_thread', 1)

    comments["author_political_leaning"] = comments["author"].map(author_political_leaning_dict)

    comments_id_author_political_leaning_dict = comments.set_index("id").author_political_leaning.to_dict()

    # Political leaning of authors in previous comments
    # TODO: These will never take into account the political leaning of the author of the post, because we are only working with the comments -- is this an issue?
    # TODO: This has 'nan' values in the list of leanings, for when an author's leaning hasn't been inferred I guess?
    print(
        "RQ2: 5) a) Getting the political leaning of the authors of previous comments ('path_author_political_leaning' column)")
    comments["path_author_political_leaning"] = comments.apply(
        lambda x: [comments_id_author_political_leaning_dict[id_] for id_ in x.path_id
                   if (not (id_ == x.id or id_.startswith("t3_")) and (
                    id_ in comments_id_author_political_leaning_dict))], axis=1)

    print(
        "RQ2: 5) b) Getting the percentage of liberal authors (not unique) in previous comments ('percentage_of_liberal_authors_in_previous_comments' column)")

    # This column could be NaN because it could be a comment at the highest level, meaning its responding directly to the post and has no parent comments, so we don't want to label it as 0 in that case
    # TODO: This is accurately getting the 'percentage of liberal authors in previous comments,' but it does not directly imply that 1-this_value is the percentage of conservative authors due to the potential presence of nan values in path_author_political_leaning
    comments["percentage_of_liberal_authors_in_previous_comments"] = comments["path_author_political_leaning"].map(
        lambda x: (x.count("left") / (len(x) if len(x) != 0 else 1)) if len(x) != 0 else np.nan)
    comments["percentage_of_conservative_authors_in_previous_comments"] = comments["path_author_political_leaning"].map(
        lambda x: (x.count("right") / (len(x) if len(x) != 0 else 1)) if len(x) != 0 else np.nan)

    one_thread_one_comment_per_author = comments.drop_duplicates(subset=["link_id", "author"], keep="last").reset_index(
        drop=True)
    number_of_unique_authors_per_thread = one_thread_one_comment_per_author.groupby("link_id").size()
    number_of_left_authors_per_thread = one_thread_one_comment_per_author.loc[
        one_thread_one_comment_per_author["author_political_leaning"] == "left"].groupby("link_id").size()
    number_of_right_authors_per_thread = one_thread_one_comment_per_author.loc[
        one_thread_one_comment_per_author["author_political_leaning"] == "right"].groupby("link_id").size()

    link_id_number_of_unique_authors = number_of_unique_authors_per_thread.to_dict()
    link_id_number_of_left_authors = number_of_left_authors_per_thread.to_dict()
    link_id_number_of_right_authors = number_of_right_authors_per_thread.to_dict()

    print(
        "RQ2: 5) c) Getting the percentage of liberal comment authors (unique) in the whole post ('percentage_of_liberal_authors_in_thread' column)")

    # This column should never be NaN, since the post will have at least one comment
    comments["percentage_of_liberal_authors_in_thread"] = comments["link_id"].map(
        lambda x: link_id_number_of_left_authors.get(x, 0) / link_id_number_of_unique_authors.get(x, 1))
    comments["percentage_of_conservative_authors_in_thread"] = comments["link_id"].map(
        lambda x: link_id_number_of_right_authors.get(x, 0) / link_id_number_of_unique_authors.get(x, 1))

    pickle_RQ2_5_filepath = path.abspath(
        path.join(data_filepath, "comments", "{}_all_comments_RQ2_5.pickle".format(pickle_startdate_enddate)))
    comments.to_pickle(pickle_RQ2_5_filepath)
    pd.set_option("max_columns", None)
    print(comments)
    print("RQ2_5 successfully pickled!")
    return comments


def regression_model_6(comments, pickle_startdate_enddate):
    print("RQ2: 6) Level of incivility and sentiment of subsequent conversation")
    base_path = path.dirname(__file__)
    data_filepath = path.abspath(path.join(base_path, "../data"))

    pd.set_option("max_columns", None)
    print(comments)

    # comment_id_sentiment_dict = comments.set_index("id").compound_sentiment.to_dict()
    # {"xyz": 0.62}

    # We need [0.42, -0.67] in overall column, then average this for the average column

    # Let's group rows by parent_id, and get their average sentiments: this would be the average sentiment of child comments
    print(
        "RQ2: 6) Grouping rows by parent_id and getting their average compound_sentiment, civility, vulgarity, namecalling, stereotype, demeaning")
    average_sentiment_of_children_comments = comments.groupby("parent_id").compound_sentiment.mean()
    average_civility_of_children_comments = comments.groupby("parent_id").civility.mean()
    average_vulgarity_of_children_comments = comments.groupby("parent_id").vulgarity.mean()
    average_namecalling_of_children_comments = comments.groupby("parent_id").namecalling.mean()
    average_stereotype_of_children_comments = comments.groupby("parent_id").stereotype.mean()
    average_demeaning_of_children_comments = comments.groupby("parent_id").demeaning.mean()

    print(
        "RQ2: 6) Creating dictionaries of parent_id: compound_sentiment, civility, vulgarity, namecalling, stereotype, demeaning")
    parent_id_average_sentiment_of_children_comments_dict = average_sentiment_of_children_comments.to_dict()
    parent_id_average_civility_of_children_comments_dict = average_civility_of_children_comments.to_dict()
    parent_id_average_vulgarity_of_children_comments_dict = average_vulgarity_of_children_comments.to_dict()
    parent_id_average_namecalling_of_children_comments_dict = average_namecalling_of_children_comments.to_dict()
    parent_id_average_stereotype_of_children_comments_dict = average_stereotype_of_children_comments.to_dict()
    parent_id_average_demeaning_of_children_comments_dict = average_demeaning_of_children_comments.to_dict()

    print("RQ2: 6) Mapping comment ids to average sentiment, civility, vulgarity, namecalling, stereotype, demeaning")
    comments["average_sentiment_of_child_comments"] = comments["id"].map(
        lambda x: parent_id_average_sentiment_of_children_comments_dict.get(x, np.nan))
    comments["average_civility_of_child_comments"] = comments["id"].map(
        lambda x: parent_id_average_civility_of_children_comments_dict.get(x, np.nan))
    comments["average_vulgarity_of_child_comments"] = comments["id"].map(
        lambda x: parent_id_average_vulgarity_of_children_comments_dict.get(x, np.nan))
    comments["average_namecalling_of_child_comments"] = comments["id"].map(
        lambda x: parent_id_average_namecalling_of_children_comments_dict.get(x, np.nan))
    comments["average_stereotype_of_child_comments"] = comments["id"].map(
        lambda x: parent_id_average_stereotype_of_children_comments_dict.get(x, np.nan))
    comments["average_demeaning_of_child_comments"] = comments["id"].map(
        lambda x: parent_id_average_demeaning_of_children_comments_dict.get(x, np.nan))

    comments["average_aggregate_civility_of_child_comments"] = comments["average_vulgarity_of_child_comments"] + \
                                                               comments["average_namecalling_of_child_comments"] + \
                                                               comments["average_stereotype_of_child_comments"] + \
                                                               comments["average_demeaning_of_child_comments"]

    print("RQ2: 6) Pickling")
    print(comments)
    pickle_RQ2_6_filepath = path.abspath(
        path.join(data_filepath, "comments", "{}_all_comments_RQ2_6.pickle".format(pickle_startdate_enddate)))
    comments.to_pickle(pickle_RQ2_6_filepath)
    pd.set_option("max_columns", None)
    print(comments)
    print("RQ2_6 successfully pickled!")
    return comments


def test_get_percentage_of_each_political_leaning():
    data = []
    for i in range(0, 30):
        data.append(["link_id_{}".format(i), "author_{}".format(i), "id_{}".format(i),
                     "{}".format("left" if (i % 2 == 0) else "right")])
        data.append(["link_id_{}".format(i), "author_{}".format(i + 10), "id_{}".format(i + 30),
                     "{}".format("left" if (i % 2 == 1) else "right")])
    data.append(["link_id_{}".format(29), "author_{}".format(29), "id_{}".format(60), "{}".format("right")])
    data.append(["link_id_{}".format(29), "author_{}".format(30), "id_{}".format(61), "{}".format("right")])
    data.append(["link_id_{}".format(29), "author_{}".format(31), "id_{}".format(62), "{}".format("right")])
    df = pd.DataFrame(data, columns=["link_id", "author", "id", "author_political_leaning"])
    print(df)

    # # Get percentage of left/right in each thread
    # total_comments_per_thread = df.groupby("link_id").size()
    # print(total_comments_per_thread) # TOTAL POSTS PER THREAD
    #
    # print()
    #
    one_thread_one_comment_per_author = df.drop_duplicates(subset=["link_id", "author"], keep="last").reset_index(
        drop=True)
    number_of_unique_authors_per_thread = one_thread_one_comment_per_author.groupby("link_id").size()
    number_of_left_authors_per_thread = one_thread_one_comment_per_author.loc[
        one_thread_one_comment_per_author["author_political_leaning"] == "left"].groupby("link_id").size()
    link_id_number_of_unique_authors = number_of_unique_authors_per_thread.to_dict()
    link_id_number_of_left_authors = number_of_left_authors_per_thread.to_dict()

    df["percentage_of_liberal_authors_in_thread"] = df["link_id"].map(
        lambda x: link_id_number_of_left_authors.get(x, 0) / link_id_number_of_unique_authors.get(x, 1))

    pd.set_option("max_columns", None)

    print(df)


def regression_model_7(comments, pickle_startdate_enddate):
    print("RQ2: 7) Adding divisiveness of parent post to each comment")

    pd.set_option("max_columns", None)
    print(comments)

    posts = utils.get_posts_divisiveness_labels()
    posts["id"] = "t3_" + posts["id"].astype(str)
    print(posts)
    post_id_divisiveness_dict = posts.set_index("id").divisiveness.to_dict()

    comments["parent_post_divisiveness"] = comments["link_id"].map(lambda x: post_id_divisiveness_dict.get(x, np.nan))

    print("RQ2: 7) Added divisiveness of parent post to each comment")

    print(comments)

    base_path = path.dirname(__file__)
    data_filepath = path.abspath(path.join(base_path, "../data"))
    pickle_RQ2_divisiveness_filepath = path.abspath(
        path.join(data_filepath, "comments", "{}_all_comments_RQ2_7.pickle".format(pickle_startdate_enddate)))
    comments.to_pickle(pickle_RQ2_divisiveness_filepath)
    return comments


def regression_model_8(comments, pickle_startdate_enddate):
    print("RQ2: 8) Getting how many child comments this comment has (number of comments with parent_id as the same)")

    pd.set_option("max_columns", None)
    number_of_comments_with_parent_id = comments.groupby(["parent_id"]).size().reset_index(name='counts')

    print("RQ2: 8) a) Getting number of comments per parent_id")
    comments_parent_id_count_dict = number_of_comments_with_parent_id.set_index("parent_id").counts.to_dict()

    # {"t1_67189": 5}

    print("RQ2: 8) b) Mapping number of comments per parent_id to comment id")
    comments["number_of_direct_child_comments"] = comments["id"].map(
        lambda x: comments_parent_id_count_dict.get(x, 0))

    print(comments)

    base_path = path.dirname(__file__)
    data_filepath = path.abspath(path.join(base_path, "../data"))
    pickle_RQ2_8_filepath = path.abspath(
        path.join(data_filepath, "comments", "{}_all_comments_RQ2_8.pickle".format(pickle_startdate_enddate)))
    comments.to_pickle(pickle_RQ2_8_filepath)
    return comments


def regression_model_9(comments, pickle_startdate_enddate):
    print("RQ2: 9) Add LIWC categories, dropping duplicate 'body' column created from LIWC, and getting length of body")
    pd.set_option("max_columns", None)
    print(comments)

    # Add LIWC
    comments = get_liwc_categories(comments)

    # Have to drop duplicate columns (body is the name of an LIWC category and it messed up having duplicate columns)
    print("testingA", 'comment_len' in comments.columns)
    print("testingB", comments.columns.duplicated())
    comments = comments.loc[:, ~comments.columns.duplicated()]
    print("testingC", comments.columns.duplicated())
    comments['comment_len'] = comments['body'].str.count(' ') + 1
    print("testingD", comments.columns.duplicated())
    print("RQ2: 9) After adding comment_len")
    print(comments)

    base_path = path.dirname(__file__)
    data_filepath = path.abspath(path.join(base_path, "../data"))
    pickle_RQ2_9_filepath = path.abspath(
        path.join(data_filepath, "comments", "{}_all_comments_RQ2_9.pickle".format(pickle_startdate_enddate)))
    comments.to_pickle(pickle_RQ2_9_filepath)
    return comments


def regression_model_10(comments, pickle_startdate_enddate):
    print("RQ2: 10) Add political leaning of subreddit")
    pd.set_option("max_columns", None)
    print(comments)
    red, blue = utils.get_political_subreddits("red"), utils.get_political_subreddits("blue")
    political_leaning_dict = {}
    for subreddit in red:
        political_leaning_dict[subreddit] = 0
    for subreddit in blue:
        political_leaning_dict[subreddit] = 1
    comments["subreddit_political_leaning"] = comments["subreddit"].map(lambda x: political_leaning_dict.get(x, np.nan))

    base_path = path.dirname(__file__)
    data_filepath = path.abspath(path.join(base_path, "../data"))
    pickle_RQ2_10_filepath = path.abspath(
        path.join(data_filepath, "comments", "{}_all_comments_RQ2_10.pickle".format(pickle_startdate_enddate)))
    comments.to_pickle(pickle_RQ2_10_filepath)
    return comments


if __name__ == '__main__':
    # test_get_percentage_of_each_political_leaning()
    if len(sys.argv) < 3:
        print(
            "1) Correct format: python RQ2.py augment [20201129_20210201]OR[all_dates] [all_columns]OR[1,2,5,6,7,8,9,10]")
        print("2) Correct format: python RQ2.py regression [20201129_20210201]OR[all_dates]")
    else:
        dates = ["20200201_20200528",
                 "20200528_20200713",
                 "20200713_20200813",
                 "20200813_20201025",
                 "20201025_20201128",
                 "20201129_20210201"]
        augment_or_regression = sys.argv[1]
        if augment_or_regression == "augment":
            # TODO: I think a potential concern of treating each post/comment file pair as a single entity separate
            #  from the rest is that comments in other files could have parent_ids/references to previous
            #  posts/comments that wouldn't be taken into account
            date_or_all_dates = sys.argv[2]
            comments_dataframes = []
            posts_dataframes = []
            print("Getting selected dates posts/comments...")
            if date_or_all_dates == "all_dates":
                # Either add all of them
                for date in dates:
                    comments_data = utils.get_comments_data(date)
                    comments_dataframes.append(comments_data)
                    posts_data = utils.get_posts_data(date)
                    posts_dataframes.append(posts_data)
            else:
                # Or just add one date
                comments_data = utils.get_comments_data(date_or_all_dates)
                comments_dataframes.append(comments_data)
                posts_data = utils.get_posts_data(date_or_all_dates)
                posts_dataframes.append(posts_data)

            print("Finished getting selected dates posts/comments.")

            # print("Concatenating posts/comments dataframes...")
            # comments = pd.concat(comments_dataframes, axis=1)
            # posts = pd.concat(posts_dataframes, axis=1)
            # print("Concatenation of posts/comments dataframes complete.")

            all_columns_or_numbers = sys.argv[3]
            columns_to_augment = []
            if all_columns_or_numbers == "all_columns":
                columns_to_augment = [1, 2, 5, 6, 7, 8, 9, 10]
            else:
                columns_to_augment = [int(numeric_string) for numeric_string in all_columns_or_numbers.split(",")]

            print("Columns to augment:", columns_to_augment)
            # TODO: Could concatenate all of the posts/comments and save as one file, but this would be massive so I decided to break it apart
            for column in columns_to_augment:
                for i in range(len(comments_dataframes)):
                    print("Augmenting {} with {}...".format(i, column))
                    pickle_startdate_enddate = date_or_all_dates if date_or_all_dates != "all_dates" else dates[i]
                    if column == 1:
                        comments_dataframes[i] = regression_model_1(comments=comments_dataframes[i],
                                                                    posts=posts_dataframes[i],
                                                                    pickle_startdate_enddate=pickle_startdate_enddate)
                    elif column == 2:
                        comments_dataframes[i] = regression_model_2(comments=comments_dataframes[i],
                                                                    pickle_startdate_enddate=pickle_startdate_enddate)
                    elif column == 5:
                        comments_dataframes[i] = regression_model_5(comments=comments_dataframes[i],
                                                                    pickle_startdate_enddate=pickle_startdate_enddate)
                    elif column == 6:
                        comments_dataframes[i] = regression_model_6(comments=comments_dataframes[i],
                                                                    pickle_startdate_enddate=pickle_startdate_enddate)
                    elif column == 7:
                        comments_dataframes[i] = regression_model_7(comments=comments_dataframes[i],
                                                                    pickle_startdate_enddate=pickle_startdate_enddate)
                    elif column == 8:
                        comments_dataframes[i] = regression_model_8(comments=comments_dataframes[i],
                                                                    pickle_startdate_enddate=pickle_startdate_enddate)
                    elif column == 9:
                        comments_dataframes[i] = regression_model_9(comments=comments_dataframes[i],
                                                                    pickle_startdate_enddate=pickle_startdate_enddate)
                    elif column == 10:
                        comments_dataframes[i] = regression_model_10(comments=comments_dataframes[i],
                                                                     pickle_startdate_enddate=pickle_startdate_enddate)

                    print("Finished augmenting {} with {}.".format(i, column))
            print("Finished augmenting columns:", columns_to_augment)
        elif augment_or_regression == "regression":
            print("Use RQ2_regression.py")
