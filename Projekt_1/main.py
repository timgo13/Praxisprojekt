import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy

movies = pd.read_csv('..\movie_archive\movies_metadata.csv')
ratings_small = pd.read_csv('..\movie_archive\\ratings_small.csv')
links_small = pd.read_csv('..\movie_archive\links_small.csv')

word_bag = CountVectorizer(analyzer='word', ngram_range=(1, 3), stop_words='english')
bag_matrix = word_bag.fit_transform(movies['title'].apply(str) + " " + movies['overview'].apply(str))
print("word_bag finished")

userid_list = ratings_small['userId'].unique()
movieid_list = movies['id'].to_list()
# changes movieId type to int and deletes all false entrys
movieid_list = [x for x in movieid_list if x.isdigit()]
movieid_list = list(map(int, movieid_list))


# get all movie vector rated from user
def get_user_rated_movies(ids):
    # movie_vector_list = [get_movie_vector(i) for i in ids]

    movie_vector_list = []
    for i in ids:
        tmdbId = links_small['tmdbId'].loc[links_small['movieId'] == i]
        try:
            index = movieid_list.index(int(tmdbId))
            movie_vector = bag_matrix[index]
            movie_vector_list.append(movie_vector)
        except ValueError:
            print("ValueError tmdbId: " + str(tmdbId))  # deleted/NaN movies

    movie_vectors = scipy.sparse.vstack(movie_vector_list)

    return movie_vectors


# compute all movie vector form specific user into one vector for each user
all_user_profiles = {}
for uid in userid_list:
    user_profile = ratings_small.loc[ratings_small['userId'] == uid]
    user_rated_movies_vector_list = get_user_rated_movies(user_profile['movieId'].to_list())
    # print(user_rated_movies_vector_list)
    try:
        user_vector = user_rated_movies_vector_list[0]
        for x in user_rated_movies_vector_list[1:]:
            user_vector = user_vector + x

        user_norm = sklearn.preprocessing.normalize(user_vector)
        # print(user_norm)
        all_user_profiles[uid] = user_norm
        print(str(uid) + " finished")
    except TypeError:
        print(str(uid) + " didnt work, coo matrix parse error")


recommendation_users = {}

for uid in userid_list:
    user_vec = all_user_profiles[uid]
    cosine_similarity_user_movies = cosine_similarity(user_vec, bag_matrix)
    top_similar = cosine_similarity_user_movies[0].argsort()[:100:-1]
    recommendation_users[uid] = [(cosine_similarity_user_movies[0][i], movies['id'][i]) for i in top_similar]
print("user_recommendation finished")

for uid in userid_list:
    user_profile = ratings_small.loc[ratings_small['userId'] == uid]
    user_recommendation = recommendation_users[uid]

    m_count = 0
    for movie_cosine, mid in user_recommendation[:len(user_profile)]:
        if mid in user_profile['movieId']:
            m_count += 1

    if m_count == len(user_profile):
        print(str(uid) + " : the first movies are already viewed")