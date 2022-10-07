import utils

"""
RQ3: How does the participation and exposure of conversations containing incivility shape the subsequent 
participation of the user in a community and opinions about a specific topic? 

RQ3.a: What are the subsequent participation patterns of users who were exposed to or were recipients of incivility 
in a political discussion? Will users who are exposed to incivility be less likely to engage in political 
discussions?
; 
Independent Variables:
- Whether user has been exposed to political discussions
    - User’s direct parent comment is uncivil

Control variables:
- User tenure
- User political leaning
- The number of threads initiated by user
- The percentage of upvotes received by user
- The proportions of uncivil comments sent / The proportions of civil comments sent
- Percentage of comments

Dependent variables:
- Engagement patterns
    - Length of participation
    - Number of comments sent in political subreddits
    - Number of comments sent in all subreddits
    - The length of comments and comments
"""


def participation(comments, posts):
    # We need to understand how we can group together users with their
    # We need to check if the user's direct parent comment was uncivil, and
    # Add the current comment's civility, and the parent comment's civility as columns
    """
             created_utc   parent_id    link_id       id            author  ... vulgarity  civility namecalling  stereotype demeaning
    1         1612153203   t3_l9gngx  t3_l9gngx  glk8hec           runrain  ...  0.536045  0.947221    0.904272    0.082785  0.033633
    2         1612152460   t3_l9t1ha  t3_l9t1ha  glk757a      SilverHerfer  ...  0.182507  0.905429    0.645800    0.086220  0.053927
    4         1612151108  t1_glit0la  t3_l9lonl  glk4jhg        Psychowitz  ...  0.501289  0.549072    0.186612    0.104418  0.108164
    6         1612150714   t3_l9t1ha  t3_l9t1ha  glk3rkp   Salmankhan42069  ...  0.313048  0.775689    0.128174    0.090085  0.076681
    9         1612150618   t3_l9t1ha  t3_l9t1ha  glk3kvz  Pile_of_Walthers  ...  0.256269  0.349456    0.201140    0.057572  0.124592
    ...              ...         ...        ...      ...               ...  ...       ...       ...         ...         ...       ...
    7953555   1606657109   t3_k37m33  t3_k37m33  ge13eq1  arthurpenhaligon  ...  0.281723  0.981340    0.826698    0.023992  0.044817
    7953561   1606653320   t3_k37m33  t3_k37m33  ge0v2ee         screen317  ...  0.061908  0.020246    0.098792    0.025778  0.075058
    7953562   1606653034   t3_k37m33  t3_k37m33  ge0uf7z         Jeffery_G  ...  0.306073  0.982551    0.917995    0.047826  0.040399
    7953565   1606651495   t3_k37m33  t3_k37m33  ge0r9un  acroporaguardian  ...  0.148657  0.728100    0.086850    0.029220  0.048263
    7953567   1606647603   t3_k37m33  t3_k37m33  ge0j1dn     AutoModerator  ...  0.074717  0.890290    0.112902    0.007533  0.142146
    [6892381 rows x 15 columns]
    """
    """
            created_utc      id         author  ... num_comments removed_by_category                                                url
    1        1612148904  l9ta02        Magro18  ...           69                 NaN                https://i.redd.it/xileojmv8se61.jpg
    2        1612148168  l9t1ha    Vthyarilops  ...           35                 NaN                https://i.redd.it/x9xf9c0j6se61.jpg
    3        1612135296  l9ow1p       porgborg  ...            4                 NaN  https://www.wsaz.com/2021/01/28/vice-president...
    5        1612126293  l9lonl   MapleSyrup04  ...           18                 NaN                       https://youtu.be/1-e8ldzy_j4
    6        1612115020  l9he4i      IronWolve  ...            2                 NaN  https://nypost.com/2021/01/30/nyc-covid-19-vac...
    ...             ...     ...            ...  ...          ...                 ...                                                ...
    178336   1606660694  k3adz9     Cat27Queen  ...            1                 NaN  https://www.ajc.com/politics/five-burning-ques...
    178337   1606660521  k3acfg     Cat27Queen  ...            1                 NaN  https://www.independent.co.uk/news/world/ameri...
    178338   1606652795  k38l3g     Cat27Queen  ...            0    automod_filtered  https://www.washingtonpost.com/politics/asian-...
    178339   1606647602  k37m33      BM2018Bot  ...          516                 NaN  https://www.reddit.com/r/VoteDEM/comments/k37m...
    178340   1606636916  k35oxo  kawhisasshole  ...            0              reddit  https://www.nytimes.com/2020/11/25/opinion/gav...
    
    [156405 rows x 13 columns]
    """
    """
     parent_id       id 
     t3_l9gngx  glk8hec
     t3_l9t1ha  glk757a
    t1_glit0la  glk4jhg
     t3_l9t1ha  glk3rkp
     t3_l9t1ha  glk3kvz
           ...      ...
     t3_k37m33  ge13eq1
     t3_k37m33  ge0v2ee
     t3_k37m33  ge0uf7z
     t3_k37m33  ge0r9un
     t3_k37m33  ge0j1dn
    """

    # parent_id can be either the original post or another comment
    # posts["id"] = "t3_" + posts["id"].astype(str)
    # Prefix all comment ids with t1_ to make mapping between parent ids and comment ids easier
    comments["id"] = "t1_" + comments["id"].astype(str)

    # Map comment ids to civility value {"t1_glit0la" : 0.997876}
    comments_id_civility_dict = dict(zip(comments.id, comments.civility))
    # Map comment ids to author {"t1_glit0la" : "mattmattmatt"}
    comments_id_author_dict = dict(zip(comments.id, comments.author))

    # We would use the parent post's civility in addition to parent comments, but there is no civility for posts :(
    # So let's drop the rows for comments where parent_civility is null
    comments["parent_civility"] = comments["parent_id"].map(comments_id_civility_dict)
    comments["parent_author"] = comments["parent_id"].map(comments_id_author_dict)
    comments = comments[comments["parent_civility"].notna()]
    print(comments)

    covariance_of_civility_vs_parent_civility = comments.civility.cov(comments.parent_civility)
    correlation_of_civility_vs_parent_civility = comments.civility.corr(comments.parent_civility)
    print(covariance_of_civility_vs_parent_civility)
    print(correlation_of_civility_vs_parent_civility)




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
    comments_data = utils.get_comments_data("20201129_20210201")
    posts_data = utils.get_posts_data("20201129_20210201")

    participation(comments=comments_data, posts=posts_data)
