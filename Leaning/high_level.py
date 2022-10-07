import utils
import numpy as np
import pandas as pd
import datetime as dt
from matplotlib import pyplot as plt


def incivility_distribution_across_subreddits(comments, analyze_all_subreddits):
    """
    Generates plot of x-axis: subreddits/subreddits grouped by political-leaning;
    y-axis: value of civility, vulgarity, etc.
    :param comments: comments_data from utils
    :param analyze_all_subreddits:
        if True, will just analyze the civility of all the subreddits of interest
        if False, will group subreddits into their political leanings ("red", or "blue")
    :return:
    """
    columns_of_interest = ["vulgarity", "civility", "namecalling", "stereotype", "demeaning"]
    # columns_of_interest = ["civility"]
    averages = []
    if analyze_all_subreddits:
        for column in columns_of_interest:
            average_column = comments.groupby("subreddit")[column].mean()
            averages.append(average_column)
    else:
        red, blue = utils.get_political_subreddits("red"), utils.get_political_subreddits("blue")
        comments.loc[comments["subreddit"].isin(red), "subreddit_political_leaning"] = "red"
        comments.loc[comments["subreddit"].isin(blue), "subreddit_political_leaning"] = "blue"
        # Get rid of rows with NaN values (i.e., where the "subreddit" column is neither red nor blue)
        comments = comments[comments["subreddit_political_leaning"].notna()]
        for column in columns_of_interest:
            average_column = comments.groupby("subreddit_political_leaning")[column].mean()
            averages.append(average_column)
    overall_df = pd.concat(averages, axis=1)
    print(overall_df)
    """
                              vulgarity  civility  namecalling  stereotype  demeaning
    subreddit                                                                    
    AskThe_Donald          0.298058  0.571077     0.433482    0.056477   0.082399
    Ask_Politics           0.226542  0.549265     0.370305    0.049766   0.084284
    BlueMidterm2018        0.286271  0.605874     0.356998    0.047597   0.064743
    Conservative           0.353930  0.691938     0.488511    0.060970   0.100101
    
                                     vulgarity  civility  namecalling  stereotype  demeaning
    subreddit_political_leaning                                                         
    blue                          0.337316  0.578931     0.417054    0.056388   0.081515
    red                           0.353703  0.688221     0.486396    0.060454   0.102767
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    overall_df.plot.bar(ax=ax)
    if analyze_all_subreddits:
        ax.set_xlabel("Subreddit")
    else:
        ax.set_xlabel("Political-leaning subreddits")
        ax.set_xticklabels(["Left-leaning", "Right-leaning"], rotation=0)

    ax.set_title("Linguistic features across subreddits")
    ax.set_ylabel("Value")

    plt.show()



def incivility_wrt_time(comments, analyze_all_subreddits, time_period, analysis_type):
    """
    comments: comments_data from utils
    analyze_all_subreddits:
        if True, will just analyze the civility over time of all the subreddits of interest
        if False, will group subreddits into their political leanings ("red", or "blue")
    time_period: plot averages of either day, week, or month
        if "day", will average civility across days
        if "week", will average civility across weeks
        if "month", will average civility across months
    analysis_type: "civility", "vulgarity", "namecalling", "stereotype", "demeaning"
    """

    # Graph mean of incivility for each subreddit for each day
    comments["date"] = pd.to_datetime(comments["created_utc"], unit="s")
    comments["day"] = comments["date"].dt.to_period("D")
    comments["week"] = comments["date"].dt.to_period("W")
    comments["month"] = comments["date"].dt.to_period("M")

    # Add user political leaning column
    # comments["political_leaning"] =

    print(comments)
    dataframe_list = []

    legend = []
    if analyze_all_subreddits:
        # We want to look at all subreddits
        legend = comments["subreddit"].unique()
        for subreddit in legend:
            subreddit_comments = comments[comments["subreddit"] == subreddit]
            civility = subreddit_comments.groupby(time_period)[analysis_type].mean()
            dataframe_list.append(civility)
    else:
        # We want to look at red vs. blue subreddits
        # Add subreddit political leaning column
        red, blue = utils.get_political_subreddits("red"), utils.get_political_subreddits("blue")
        comments.loc[comments["subreddit"].isin(red), "subreddit_political_leaning"] = "red"
        comments.loc[comments["subreddit"].isin(blue), "subreddit_political_leaning"] = "blue"

        # Get rid of rows with NaN values (i.e., where the "subreddit" column is neither red nor blue)
        comments = comments[comments["subreddit_political_leaning"].notna()]
        legend = comments["subreddit_political_leaning"].unique()
        print(legend)
        for leaning in legend:
            leaning_comments = comments[comments["subreddit_political_leaning"] == leaning]
            civility = leaning_comments.groupby(time_period)[analysis_type].mean()
            dataframe_list.append(civility)
        legend = ["Right-leaning subreddits", "Left-leaning subreddits"]

    overall_df = pd.concat(dataframe_list, axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid()

    if analyze_all_subreddits:
        # We are looking at all subreddits, color doesn't matter
        ax.plot_date(overall_df.index, overall_df.values, linestyle="solid", marker="None")
    else:
        # We are looking at political subreddits, let's change the lines to red and blue
        ax.plot_date(overall_df.index, overall_df.iloc[:, 0], color="red", linestyle="solid", marker="None")
        ax.plot_date(overall_df.index, overall_df.iloc[:, 1], color="blue", linestyle="solid", marker="None")

    ax.legend(legend)
    ax.set_title("Incivility of political subreddits over time")
    if analysis_type == "civility":
        # Must adjust title since the attribute in the dataset is "civility" and actually represents incivility
        analysis_type = "Incivility"
    ax.set_ylabel(analysis_type.capitalize())
    print(overall_df)

    # Adding vertical time period lines
    # x = dt.datetime(2020, 11, 1)
    # ax.axvline(x=x, ls='--', color="gray")
    # ax.text(x=x, y=0.9, s="U.S. election", fontsize=20)
    #
    # x = dt.datetime(2020, 3, 16)
    # ax.axvline(x=x, ls='--', color="gray")
    # ax.text(x=x, y=0.9, s="COVID-19 quarantine", fontsize=20)

    x = dt.datetime(2021, 1, 6)
    ax.axvline(x=x, ls='--', color="gray")
    ax.text(x=x, y=0.62, s="U.S. Capitol riot")

    plt.show()



def incivility_wrt_users_political_leanings(comments):
    """
    :param comments: comments_data from utils
    """
    left_users, right_users = utils.get_left_and_right_users(comments=comments)
    # all_users = utils.get_left_and_right_users(comments=comments)
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

    # Let us map authors to average incivility, as well as understanding if they are incivil
    # in subreddits with political-leanings opposite to their own

    # Get comments in left and right subreddits, grouped by author
    red, blue = utils.get_political_subreddits("red"), utils.get_political_subreddits("blue")
    overall_comment_groups = comments.groupby("author")
    right_comment_groups = comments[comments.subreddit.isin(red)].groupby('author')
    left_comment_groups = comments[comments.subreddit.isin(blue)].groupby('author')

    # Get individual users' averages of their civility in all subreddits, left subreddits, and right subreddits
    left_users["civility_overall"] = left_users["author"].map(
        dict(overall_comment_groups["civility"].mean()))  # .fillna(0)
    left_users["civility_in_left_subreddits"] = left_users["author"].map(
        dict(left_comment_groups["civility"].mean()))  # .fillna(0)
    left_users["civility_in_right_subreddits"] = left_users["author"].map(
        dict(right_comment_groups["civility"].mean()))  # .fillna(0)

    right_users["civility_overall"] = right_users["author"].map(
        dict(overall_comment_groups["civility"].mean()))  # .fillna(0)
    right_users["civility_in_left_subreddits"] = right_users["author"].map(
        dict(left_comment_groups["civility"].mean()))  # .fillna(0)
    right_users["civility_in_right_subreddits"] = right_users["author"].map(
        dict(right_comment_groups["civility"].mean()))  # .fillna(0)

    # We have the count of each user's comments in left, right, and all subreddits (i.e., left_count, right_count, total_count)
    # print(left_users)
    """
                       author  total_count  left_count  right_count  left_score  right_score  civility_overall  civility_in_left_subreddits  civility_in_right_subreddits
    53             --ElonMusk           32         3.0          0.0    1.666667          0.0          0.603192                     0.693118                           NaN
    58      --GrinAndBearIt--           17         7.0          0.0    1.571429          0.0          0.694647                     0.733383                           NaN
    108           --_-_o_-_--            8         3.0          1.0    4.333333        -12.0          0.727455                     0.622006                      0.981326
    143      --theriverstyx--            3         3.0          0.0    1.333333          0.0          0.812017                     0.812017                           NaN
    148                 -0-O-           27         9.0          0.0    1.777778          0.0          0.696358                     0.716932                           NaN
    ...                   ...          ...         ...          ...         ...          ...               ...                          ...                           ...
    618369              zyygh            2         2.0          0.0  110.000000          0.0          0.450855                     0.450855                           NaN
    618393     zzcheeseballzz            3         1.0          0.0    4.000000          0.0          0.981861                     0.983972                           NaN
    618439          zzzaacchh           46         1.0          0.0    5.000000          0.0          0.685752                     0.646661                           NaN
    618442            zzzeoww            1         1.0          0.0   15.000000          0.0          0.346513                     0.346513                           NaN
    618450           zzztoken          105       105.0          0.0    2.295238          0.0          0.589026                     0.589026                           NaN
    
    [13143 rows x 9 columns]
    """

    # Are left or right users more likely to engage non-civily in the opposite political leaning subreddit?
    # Get average of all left/right users' average civilities in all subreddits, left subreddits, and right subreddits
    left_stats = left_users[["civility_overall", "civility_in_left_subreddits", "civility_in_right_subreddits"]].describe()
    right_stats = right_users[["civility_overall", "civility_in_left_subreddits", "civility_in_right_subreddits"]].describe()
    print("STATS")
    print(left_stats)
    print(right_stats)

    labels = ['All subreddits', 'Left-leaning subreddits', 'Right-leaning subreddits']
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, left_stats.loc["mean"], width, label='Left-leaning users', color="blue")
    rects2 = ax.bar(x + width / 2, right_stats.loc["mean"], width, label='Right-leaning users', color="red")
    error1 = ax.errorbar(x-width/2, left_stats.loc["mean"], left_stats.loc["std"], ls="none", color="darkgrey")
    error2 = ax.errorbar(x+width/2, right_stats.loc["mean"], right_stats.loc["std"], ls="none", color="darkgrey")

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for k in range(len(rects)):
            rect = rects[k]
            height = rect.get_height()
            bar_label = str(height)[0:5] if height > 0 else str(height)[0:6]
            ax.annotate('{}'.format(bar_label),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        # ha='center', va='bottom', rotation=0, fontsize=14)
                        ha='center', va='bottom', rotation=0)

    autolabel(rects1)
    autolabel(rects2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylim(0, 1)
    ax.set_ylabel('Incivility')
    ax.set_title('Incivility in subreddits versus user political-leanings')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()



def user_participation(comments):
    # Histogram of subreddit counts for all users
    
    comment_groups = comments.groupby("author")
    num_subreddits = comment_groups.agg({"subreddit": "nunique"})
    num_subreddits = num_subreddits.reset_index()
    hist = plt.hist(num_subreddits["subreddit"])
    plt.yscale('log')
    plt.xlabel('Number of sub-reddits')
    plt.ylabel('Frequency')
    plt.title('Participation of users in subreddits')
    plt.show()
    

    # Histograms of subreddit counts for each user political leaning
    left_users, right_users = utils.get_left_and_right_users(comments)
    left_user_num_subreddits = num_subreddits[num_subreddits.author.isin(left_users["author"].values)]
    right_user_num_subreddits = num_subreddits[num_subreddits.author.isin(right_users["author"].values)]
    print("USER POLITICAL LEANING STATS")
    print(left_user_num_subreddits["subreddit"].describe())
    print(right_user_num_subreddits["subreddit"].describe())

    plt.hist(left_user_num_subreddits["subreddit"])
    plt.xlabel('Number of sub-reddits')
    plt.ylabel('Number of users')
    plt.title('Participation of left-leaning users in subreddits')
    plt.yscale('log')
    plt.show()
    plt.hist(right_user_num_subreddits["subreddit"], color="red")
    plt.xlabel('Number of sub-reddits')
    plt.ylabel('Number of users')
    plt.title('Participation of right-leaning users in subreddits')
    plt.yscale('log')
    plt.show()
    

    # User's political leaning vs commenting on red/blue subreddits
    
    comment_groups = comments.groupby("author")
    num_subreddits = comment_groups.agg({"subreddit": "unique"})
    num_subreddits = num_subreddits.reset_index()
    left_users, right_users = utils.get_left_and_right_users(comments)
    left_user_num_subreddits = num_subreddits[num_subreddits.author.isin(left_users["author"].values)]
    right_user_num_subreddits = num_subreddits[num_subreddits.author.isin(right_users["author"].values)]

    red, blue = utils.get_political_subreddits("red"), utils.get_political_subreddits("blue")
    left_user_num_subreddits = left_user_num_subreddits.assign(blue=[sum(x in blue for x in row) for row in left_user_num_subreddits.subreddit])
    left_user_num_subreddits = left_user_num_subreddits.assign(red=[sum(x in red for x in row) for row in left_user_num_subreddits.subreddit])
    left_user_num_subreddits = left_user_num_subreddits.assign(neutral=[sum(x not in red and x not in blue for x in row) for row in left_user_num_subreddits.subreddit])
    right_user_num_subreddits = right_user_num_subreddits.assign(blue=[sum(x in blue for x in row) for row in right_user_num_subreddits.subreddit])
    right_user_num_subreddits = right_user_num_subreddits.assign(red=[sum(x in red for x in row) for row in right_user_num_subreddits.subreddit])
    right_user_num_subreddits = right_user_num_subreddits.assign(neutral=[sum(x not in red and x not in blue for x in row) for row in right_user_num_subreddits.subreddit])
    
    # Stats 
    print("LEFT STATS")
    print(left_user_num_subreddits["blue"].describe())
    print(left_user_num_subreddits["red"].describe())
    print(left_user_num_subreddits["neutral"].describe())
    print("RIGHT STATS")
    print(right_user_num_subreddits["blue"].describe())
    print(right_user_num_subreddits["red"].describe())
    print(right_user_num_subreddits["neutral"].describe())

    bins = np.linspace(0, 15, 16)
    # Histogram for left-leaning users
    plt.hist(left_user_num_subreddits["blue"], color="blue", bins=bins, alpha=0.5, label="Liberal")
    plt.hist(left_user_num_subreddits["red"], color="red", bins=bins, alpha=0.7, label="Conservative")
    plt.hist(left_user_num_subreddits["neutral"], color="gray", bins=bins, alpha=0.6, label="Neutral")
    
    '''plt.hist((left_user_num_subreddits["blue"], left_user_num_subreddits["red"], left_user_num_subreddits["neutral"]), 
                bins=bins,
                density=True,
                histtype='bar',
                #colors=["blue", "red", "gray"],
                #labels=["Liberal", "Conservative", "Neutral"]
            )
    '''
    plt.legend(loc='upper right')
    plt.yscale('log')
    plt.xlabel('Number of sub-reddits')
    plt.ylabel('Number of users')
    plt.title('Participation of left-leaning users \n in Democratic and Republican subreddits')
    plt.show()
    
    # Histogram for right-leaning users
    plt.hist(right_user_num_subreddits["blue"], color="blue", bins=bins, alpha=0.7, label="Liberal")
    plt.hist(right_user_num_subreddits["red"], color="red", bins=bins, alpha=0.5, label="Conservative")
    plt.hist(right_user_num_subreddits["neutral"], color="gray", bins=bins, alpha=0.6, label="Neutral")
    plt.legend(loc='upper right')
    plt.yscale('log')
    plt.xlabel('Number of sub-reddits')
    plt.ylabel('Number of users')
    plt.title('Participation of right-leaning users \n in Democratic and Republican subreddits')
    plt.show()


def participation(comments):
    stereotype_new = np.load('stereotypes_new_dict.npy', allow_pickle=True)
    stereotype_new = stereotype_new.item()
    comments_data["stereotype"] = comments_data["id"].map(stereotype_new)

    # Prefix all comment ids with t1_ to make mapping between parent ids and comment ids easier
    comments["id"] = "t1_" + comments["id"].astype(str)

    # Map comment ids to civility value {"t1_glit0la" : 0.997876}
    comments_id_civility_dict = dict(zip(comments.id, comments.civility))
    # Map comment ids to author {"t1_glit0la" : "mattmattmatt"}
    comments_id_author_dict = dict(zip(comments.id, comments.author))
    comments_id_vulgarity_dict = dict(zip(comments.id, comments.vulgarity))
    comments_id_demeaning_dict = dict(zip(comments.id, comments.demeaning))
    comments_id_stereotype_dict = dict(zip(comments.id, comments.stereotype))
    comments_id_namecalling_dict = dict(zip(comments.id, comments.namecalling))

    # We would use the parent post's civility in addition to parent comments, but there is no civility for posts :(
    # So let's drop the rows for comments where parent_civility is null
    comments["parent_civility"] = comments["parent_id"].map(comments_id_civility_dict)
    comments["parent_author"] = comments["parent_id"].map(comments_id_author_dict)
    comments["parent_vulgarity"] = comments["parent_id"].map(comments_id_vulgarity_dict)
    comments["parent_demeaning"] = comments["parent_id"].map(comments_id_demeaning_dict)
    comments["parent_stereotype"] = comments["parent_id"].map(comments_id_stereotype_dict)
    comments["parent_namecalling"] = comments["parent_id"].map(comments_id_namecalling_dict)
    #comments = comments[comments["parent_civility"].notna()]

    leanings = utils.get_authors_leaning()
    leanings.index = leanings['author']
    
    comments = comments[comments["author"].isin(leanings["author"].values) & comments["parent_author"].isin(leanings["author"].values)]
    comments = comments.assign(user_leaning=["left" if leanings["is_left"].loc[author] == 1 else "right" for author in comments["author"]])
    comments = comments.assign(parent_leaning=["left" if leanings["is_left"].loc[author] == 1 else "right" for author in comments["parent_author"]])
    print(comments.columns)
    print(comments.head(5))

    left_user_left_parent = comments[(comments["user_leaning"] == "left") & (comments["parent_leaning"] == "left")]
    left_user_right_parent = comments[(comments["user_leaning"] == "left") & (comments["parent_leaning"] == "right")]
    right_user_left_parent = comments[(comments["user_leaning"] == "right") & (comments["parent_leaning"] == "left")] 
    right_user_right_parent = comments[(comments["user_leaning"] == "right") & (comments["parent_leaning"] == "right")]

    # LEFT USER LEFT PARENT
    print(left_user_left_parent["vulgarity"].describe())
    print(left_user_left_parent["demeaning"].describe())
    print(left_user_left_parent["stereotype"].describe())
    print(left_user_left_parent["namecalling"].describe())

    # LEFT USER RIGHT PARENT
    print(left_user_right_parent["vulgarity"].describe())
    print(left_user_right_parent["demeaning"].describe())
    print(left_user_right_parent["stereotype"].describe())
    print(left_user_right_parent["namecalling"].describe())

    # RIGHT USER LEFT PARENT
    print(right_user_left_parent["vulgarity"].describe())
    print(right_user_left_parent["demeaning"].describe())
    print(right_user_left_parent["stereotype"].describe())
    print(right_user_left_parent["namecalling"].describe())

    # RIGHT USER RIGHT PARENT
    print(right_user_right_parent["vulgarity"].describe())
    print(right_user_right_parent["demeaning"].describe())
    print(right_user_right_parent["stereotype"].describe())
    print(right_user_right_parent["namecalling"].describe())


    data_to_plot = [left_user_left_parent["vulgarity"], left_user_left_parent["demeaning"], left_user_left_parent["stereotype"], left_user_left_parent["namecalling"],
                left_user_right_parent["vulgarity"], left_user_right_parent["demeaning"], left_user_right_parent["stereotype"], left_user_right_parent["namecalling"],
                right_user_left_parent["vulgarity"], right_user_left_parent["demeaning"], right_user_left_parent["stereotype"], right_user_left_parent["namecalling"],
                right_user_right_parent["vulgarity"], right_user_right_parent["demeaning"], right_user_right_parent["stereotype"], right_user_right_parent["namecalling"]]

    plt.figure(figsize=(12, 10))
    plt.boxplot(data_to_plot,
                positions=[1, 1.6, 2.2, 2.8, 3.7, 4.3, 4.9, 5.5, 6.4, 7.0, 7.6, 8.2, 9.1, 9.7, 10.3, 10.9],
                labels=['vulgarity','demeaning','stereotype','namecalling',
                        'vulgarity','demeaning','stereotype','namecalling',
                        'vulgarity','demeaning','stereotype','namecalling',
                        'vulgarity','demeaning','stereotype','namecalling'
                        ])
    plt.title('Incivility label scores for users interacting with other users')
    plt.ylabel('Score')
    plt.xlabel('Incivility Label')
    plt.xticks(rotation=40)
    plt.text(1.9, -.03, "Left user, left parent", horizontalalignment="center")
    plt.text(4.6, -.03, "Left user, right parent", horizontalalignment="center")
    plt.text(7.3, -.03, "Right user, left parent", horizontalalignment="center")
    plt.text(10, -.03, "Right user, right parent", horizontalalignment="center")
    plt.show()


    left_user_left_parent["civility"] = left_user_left_parent["vulgarity"] + left_user_left_parent["demeaning"] + left_user_left_parent["stereotype"] + left_user_left_parent["namecalling"]
    left_user_left_parent["civility"] = (left_user_left_parent["civility"] - left_user_left_parent["civility"].min()) / (left_user_left_parent["civility"].max() - left_user_left_parent["civility"].min())
    left_user_left_parent["parent_civility"] = left_user_left_parent["parent_vulgarity"] + left_user_left_parent["parent_demeaning"] + left_user_left_parent["parent_stereotype"] + left_user_left_parent["parent_namecalling"]
    left_user_left_parent["parent_civility"] = (left_user_left_parent["parent_civility"] - left_user_left_parent["parent_civility"].min()) / (left_user_left_parent["parent_civility"].max() - left_user_left_parent["parent_civility"].min())    

    left_user_right_parent["civility"] = left_user_right_parent["vulgarity"] + left_user_right_parent["demeaning"] + left_user_right_parent["stereotype"] + left_user_right_parent["namecalling"]
    left_user_right_parent["civility"] = (left_user_right_parent["civility"] - left_user_right_parent["civility"].min()) / (left_user_right_parent["civility"].max() - left_user_right_parent["civility"].min())
    left_user_right_parent["parent_civility"] = left_user_right_parent["parent_vulgarity"] + left_user_right_parent["parent_demeaning"] + left_user_right_parent["parent_stereotype"] + left_user_right_parent["parent_namecalling"]
    left_user_right_parent["parent_civility"] = (left_user_right_parent["parent_civility"] - left_user_right_parent["parent_civility"].min()) / (left_user_right_parent["parent_civility"].max() - left_user_right_parent["parent_civility"].min())      

    right_user_left_parent["civility"] = right_user_left_parent["vulgarity"] + right_user_left_parent["demeaning"] + right_user_left_parent["stereotype"] + right_user_left_parent["namecalling"]
    right_user_left_parent["civility"] = (right_user_left_parent["civility"] - right_user_left_parent["civility"].min()) / (right_user_left_parent["civility"].max() - right_user_left_parent["civility"].min())
    right_user_left_parent["parent_civility"] = right_user_left_parent["parent_vulgarity"] + right_user_left_parent["parent_demeaning"] + right_user_left_parent["parent_stereotype"] + right_user_left_parent["parent_namecalling"]
    right_user_left_parent["parent_civility"] = (right_user_left_parent["parent_civility"] - right_user_left_parent["parent_civility"].min()) / (right_user_left_parent["parent_civility"].max() - right_user_left_parent["parent_civility"].min())      

    right_user_right_parent["civility"] = right_user_right_parent["vulgarity"] + right_user_right_parent["demeaning"] + right_user_right_parent["stereotype"] + right_user_right_parent["namecalling"]
    right_user_right_parent["civility"] = (right_user_right_parent["civility"] - right_user_right_parent["civility"].min()) / (right_user_right_parent["civility"].max() - right_user_right_parent["civility"].min())
    right_user_right_parent["parent_civility"] = right_user_right_parent["parent_vulgarity"] + right_user_right_parent["parent_demeaning"] + right_user_right_parent["parent_stereotype"] + right_user_right_parent["parent_namecalling"]
    right_user_right_parent["parent_civility"] = (right_user_right_parent["parent_civility"] - right_user_right_parent["parent_civility"].min()) / (right_user_right_parent["parent_civility"].max() - right_user_right_parent["parent_civility"].min())    
    

    
    # LEFT USER LEFT PARENT
    print(left_user_left_parent["civility"].describe())
    print(left_user_left_parent["parent_civility"].describe())
    left_user_left_parent["civility_difference"] = left_user_left_parent["civility"] - left_user_left_parent["parent_civility"]
    print(left_user_left_parent["civility_difference"].describe())
    print(' ')

    # LEFT USER RIGHT PARENT
    print(left_user_right_parent["civility"].describe())
    print(left_user_right_parent["parent_civility"].describe())
    left_user_right_parent["civility_difference"] = left_user_right_parent["civility"] - left_user_right_parent["parent_civility"]
    print(left_user_right_parent["civility_difference"].describe())
    print(' ')

    # RIGHT USER LEFT PARENT
    print(right_user_left_parent["civility"].describe())
    print(right_user_left_parent["parent_civility"].describe())
    right_user_left_parent["civility_difference"] = right_user_left_parent["civility"] - right_user_left_parent["parent_civility"]
    print(right_user_left_parent["civility_difference"].describe())
    print(' ')

    # RIGHT USER RIGHT PARENT
    print(right_user_right_parent["civility"].describe())
    print(right_user_right_parent["parent_civility"].describe())
    right_user_right_parent["civility_difference"] = right_user_right_parent["civility"] - right_user_right_parent["parent_civility"]
    print(right_user_right_parent["civility_difference"].describe())  
    

    
    data_to_plot = [left_user_left_parent["civility"], left_user_left_parent["parent_civility"],
                left_user_right_parent["civility"], left_user_right_parent["parent_civility"],
                right_user_left_parent["civility"], right_user_left_parent["parent_civility"],
                right_user_right_parent["civility"], right_user_right_parent["parent_civility"]]
    plt.figure(figsize=(10, 7))
    plt.boxplot(data_to_plot,
                positions=[1, 1.6, 2.5, 3.1, 4, 4.6, 5.5, 6.1],
                labels=['Left \n user','Left \n parent','Left \n user','Right \n parent','Right \n user','Left \n parent','Right \n user','Right \n parent'])
    plt.title('Incivility of user comments and their parent comments')
    plt.ylabel('Incivility')
    plt.xlabel('\n User-parent interaction pair')
    plt.show()
    
    


    

if __name__ == '__main__':
    """
    Posts: ['created_utc', 'id', 'author', 'title', 'selftext', 'subreddit', 'link',
           'score', 'upvote_ratio', 'author_flair_text', 'num_comments',
           'removed_by_category', 'url']
    """
    # Get all posts
    # posts_data = utils.get_posts_data()

    """
    Comments: ['created_utc', 'parent_id', 'link_id', 'id', 'author', 'body',
           'subreddit', 'link', 'score', 'author_flair_text', 'vulgarity',
           'civility', 'namecalling', 'stereotype', 'demeaning']
    """
    # Get all comments
    
    date = "20201129_20210201"
    comments_data = utils.get_comments_data(date)

    for date in ["20200201_20200528", "20200528_20200713", "20200713_20200813", "20200813_20201025", "20201025_20201128"]:
        comments = utils.get_comments_data(date)
        comments_data = comments_data.append(comments)
    participation(comments=comments_data)

    #user_participation(comments=comments_data)
    #incivility_wrt_users_political_leanings(comments=comments_data)

    # Overall data
    # incivility_distribution_across_subreddits(comments_data, analyze_all_subreddits=False)
    # incivility_distribution_across_subreddits(comments_data, analyze_all_subreddits=True)

    #incivility_wrt_users_political_leanings(comments=comments_data)

    # Temporal data
    # incivility_wrt_time(comments=comments_data, analyze_all_subreddits=False, time_period="day",
    #                     analysis_type="civility")
    # incivility_wrt_time(comments=comments_data, analyze_all_subreddits=False, time_period="week",
    #                     analysis_type="civility")
    # incivility_wrt_time(comments=comments_data, analyze_all_subreddits=False, time_period="month",
    #                     analysis_type="civility")
    #
    # incivility_wrt_time(comments=comments_data, analyze_all_subreddits=True, time_period="day",
    #                     analysis_type="civility")
    # incivility_wrt_time(comments=comments_data, analyze_all_subreddits=True, time_period="week",
    #                     analysis_type="civility")
    # incivility_wrt_time(comments=comments_data, analyze_all_subreddits=True, time_period="month",
    #                     analysis_type="civility")
