import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy
from timeit import default_timer as timer
from datetime import timedelta

from concurrent.futures import ThreadPoolExecutor

movies = pd.read_csv('..\movie_archive\movies_metadata.csv')
ratings_small = pd.read_csv('..\movie_archive\\ratings_small.csv')
links_small = pd.read_csv('..\movie_archive\links_small.csv')

# compute all movie vector form specific user into one vector for each user
all_user_profiles = {}
ram = 12  # in GB
count_to_big = 0
count_false_int = 0
recommendation_users = {}


# get all movie vector rated from user
def get_user_rated_movies(ids):
    movie_vector_list = []
    for i in ids:
        tmdbId = links_small['tmdbId'].loc[links_small['movieId'] == i]
        try:
            index = movieid_list.index(int(tmdbId))
            movie_vector = tfidf_matrix[index]
            movie_vector_list.append(movie_vector)
        except ValueError:
            # print ("ValueError tmdbId: " + str(tmdbId))     # deleted/NaN movies
            count_false_int = count_false_int + 1

    movie_vectors = scipy.sparse.vstack(movie_vector_list)

    return movie_vectors


def create_user_vector_jit(user_rated_movies_vector_array):
    user_vector = user_rated_movies_vector_array.sum(axis=0)

    return user_vector


def create_user_vectors(uid):
    user_profile = ratings_small.loc[ratings_small['userId'] == uid]

    user_rated_movies_vector_list = get_user_rated_movies(user_profile['movieId'].to_list())
    n = user_rated_movies_vector_list.shape[0]

    # variable threshold (change ram)
    # thresh = (n * user_rated_movies_vector_list.shape[1] / 8) * 64
    # thresh_ram = ram * 1000000000

    # if thresh > thresh_ram:
    if n > 1000:
        count_to_big + 1

    else:

        user_vectors_array = scipy.sparse.csr_matrix.toarray(user_rated_movies_vector_list)
        user_vectors_array.reshape(user_rated_movies_vector_list.shape)

        user_vector = create_user_vector_jit(user_vectors_array)

        user_norm = sklearn.preprocessing.normalize(scipy.sparse.csr_matrix(user_vector))
        all_user_profiles[uid] = user_norm


def sort_user_vectors(u_id):
    if u_id in all_user_profiles.keys():
        user_vec = all_user_profiles[u_id]
        cosine_similarity_user_movies = cosine_similarity(user_vec, tfidf_matrix)
        top_similar = cosine_similarity_user_movies[0].argsort()[:100:-1]
        # top_similar = cosine_similarity_user_movies[0].sort(kind='timsort')[:100:-1]
        # top_similar = np.sort(cosine_similarity_user_movies[0], kind='stable')
        recommendation_users[u_id] = [(cosine_similarity_user_movies[0][i], movies['id'][i]) for i in top_similar]


#feature_size_samples = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 1500000, 2000000, 2500000]
feature_size_samples = [1000000]

for feature_size in feature_size_samples:
    start = timer()

    # tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), stop_words='english', max_features= feature_size)
    tfidf = CountVectorizer(analyzer='word', ngram_range=(1, 3), stop_words='english', max_features=feature_size)
    tfidf_matrix = tfidf.fit_transform(movies['title'].apply(str) + " " + movies['overview'].apply(str))
    tfidf_matrix

    end = timer()
    print("Feature Size: " + str(feature_size))
    print("Time Calculate Bag of Words Matrix HH:MM:SS: ", timedelta(seconds=end - start))

    all_user_profiles = {}
    ram = 12  # in GB
    count_to_big = 0
    count_false_int = 0
    recommendation_users = {}

    userid_list = ratings_small['userId'].unique()
    movieid_list = movies['id'].to_list()

    # changes movieId type to int and deletes all false entries
    movieid_list = [x for x in movieid_list if x.isdigit()]
    movieid_list = list(map(int, movieid_list))

    start = timer()

    with ThreadPoolExecutor(max_workers=8) as executer:
        executer.map(create_user_vectors, userid_list)

    end = timer()
    print("Time Calculate Vectors HH:MM:SS: ", timedelta(seconds=end - start))

    start = timer()

    #with ThreadPoolExecutor(max_workers=8) as executer:
    #    executer.map(sort_user_vectors, userid_list)


    for u in userid_list:
        sort_user_vectors(u)
    end = timer()

    print("Time Sorting HH:MM:SS: ", timedelta(seconds=end - start))

    start = timer()

    percentage_already_seen = []
    for uid in userid_list:
        if uid in all_user_profiles.keys():
            user_profile = ratings_small.loc[ratings_small['userId'] == uid]
            user_recommendation = recommendation_users[uid]

            user_profile_mid_list = user_profile['movieId'].to_list()
            user_tmdb_list = []
            for m in user_profile_mid_list:
                tmdbId = links_small['tmdbId'].loc[links_small['movieId'] == m]
                try:
                    user_tmdb_list.append(int(tmdbId))
                except ValueError:
                    # print("ValueError: " + str(tmdbId))
                    count_false_int = count_false_int + 1

            m_count = 0
            for movie_cosine, mid in user_recommendation[:len(user_profile)]:
                try:
                    if int(mid) in user_tmdb_list:
                        m_count += 1
                except ValueError:
                    print("ValueError: " + str(
                        mid) + " not in tmdbid list")  # Fehler weil, schon gesucht aber nicht gefunden // skipped because to big sample size

            # if m_count == len(user_profile):
            #    print(str(uid) + " : the first movies are already viewed")
            # else:
            #    print(str(m_count) + " of " + str(len(user_profile)) + " in first recommended already seen")

            p = m_count / len(user_profile)
            percentage_already_seen.append(p)

    average_percentage = sum(percentage_already_seen) / len(percentage_already_seen)
    print("Average Percentage of already seen movies: " + str(average_percentage))

    end = timer()
    print("Time Percentage HH:MM:SS: ", timedelta(seconds=end - start))

    print("\n")
