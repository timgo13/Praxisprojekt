import numpy as np
import pandas as pd
from timeit import default_timer as timer
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import remove_stopwords
import scipy

movies = pd.read_csv('movie_archive\movies_metadata.csv')
ratings_small = pd.read_csv('movie_archive\\ratings_small.csv')
links_small = pd.read_csv('movie_archive\links_small.csv')


import nltk
from nltk.stem import PorterStemmer
words = set(nltk.corpus.words.words())
porter = PorterStemmer()

def remove_non_english(sent):
    s = " ".join(w for w in nltk.wordpunct_tokenize(sent) if w.lower() in words or not w.isalpha())
    return s

df = movies.filter(['id','title','overview'], axis=1)

# Create new movies df, with only important columns
df = movies.filter(['id','title','overview'], axis=1)
print('Before: ' + str(len(df)))
df = df.dropna()
# check movies ids, convert to int, drop if not possible
# check if text in english
for id in df['id'].to_list():
    try:
        id = int(id[0])
    except ValueError:
        df = df[df['id'] != id]

for title in df['title'].to_list():
    try:
        title = str(title)
        if title == 'nan' or title == '' or title == ' ':
            df = df[df['title'] != title]
        elif len(remove_non_english(title)) < len(title)/4:
            df = df[df['title'] != title]
    except ValueError:
        df = df[df['title'] != title]

for overview in df['overview'].to_list():
    try:
        overview = str(overview)
        if overview == 'nan' or overview == '' or overview == ' ' or overview == '...':
            df = df[df['overview'] != overview]
        elif len(remove_non_english(overview)) < len(overview)/2:
            df = df[df['overview'] != overview]
    except ValueError:
        df = df[df['overview'] != overview]

df = df.reset_index(drop=True)

print('After: ' + str(len(df)))

movieid_list = df['id'].to_list()
movieid_list = list(map(int, movieid_list))

ratings = ratings_small
#for r in ratings['movieId']:
#    try:
#        mid = r
#        tmdbId = links_small['tmdbId'].loc[links_small['movieId'] == int(mid)]
#        index = movieid_list.index(int(tmdbId))
#    except ValueError:
#        ratings = ratings[ratings['movieId'] != mid]

#print(len(ratings))

movie_texts = df['overview'].apply(str).to_list()
movie_titles = df['title'].apply(str).to_list()
text_tokens = []
for i in range(len(movie_texts)):
    if movie_titles[i] != 'nan' and movie_texts[i] != 'nan':
        text = movie_titles[i] + " " + movie_texts[i]
        #text = porter.stem(text)
        text = remove_stopwords(text)
        text_tokens.append(gensim.utils.simple_preprocess(text))

tagged_text = [TaggedDocument(t, [i]) for i, t in enumerate(text_tokens)]


def model_inferred_vectors(x_vector_size, x_min_count, x_window, x_negative, x_epochs, x_hs):
    model = gensim.models.doc2vec.Doc2Vec(dm=0, dbow_words=1, min_count=x_min_count, negative=x_negative,
                    hs=x_hs, sample=1e-4, window=x_window, vector_size=100, workers=8, epochs=x_epochs)

    model.build_vocab(tagged_text)

    model.train(tagged_text, total_examples=model.corpus_count, epochs=model.epochs)

    inferr_doc_vecs = []
    for i in range(len(tagged_text)):
        inferred_vector = model.infer_vector(tagged_text[0].words, steps=30, alpha=0.025)
        inferr_doc_vecs.append(inferred_vector)

    return model, inferr_doc_vecs


#userid_list = ratings_train['userId'].unique()
userid_list = ratings['userId'].unique()
movieid_list = df['id'].to_list()
movieid_list = list(map(int, movieid_list))

# get all movie vector rated from user
def get_user_rated_movies(ids, inferr_doc_vecs):
        
    movie_vector_list = []
    for i in ids:
        tmdbId = links_small['tmdbId'].loc[links_small['movieId'] == i]
        try:
            index = movieid_list.index(int(tmdbId))
            #movie_vector = model.dv[index]
            movie_vector = inferr_doc_vecs[index]
            movie_vector_list.append(movie_vector)
        except ValueError:
            pass

    movie_vectors = movie_vector_list
    return movie_vectors

def compute_user_profiles(inferr_doc_vecs):
    # compute all movie vector form specific user into one vector for each user
    all_user_profiles = {}

    for uid in userid_list:
        #user_profile = ratings_train.loc[ratings_train['userId'] == uid]
        user_profile = ratings.loc[ratings['userId'] == uid]
        user_rated_movies_vector_list = get_user_rated_movies(user_profile['movieId'].to_list(), inferr_doc_vecs)
        user_rated_movies_vector_array = np.array(user_rated_movies_vector_list)
        #user_vector = user_rated_movies_vector_array.sum(axis=0)
        user_vector = np.mean(user_rated_movies_vector_array, axis=0)
        all_user_profiles[uid] = user_vector

    return all_user_profiles


def compute_recommendations(all_user_profiles, model):
    recommendation_users = {}
    for uid in userid_list:
        if uid in all_user_profiles.keys():
            sims = model.dv.most_similar([all_user_profiles[uid]], topn=len(model.dv))
            recommendation_users[uid] = sims
    return recommendation_users


def compute_recall(recommendation_users):
    percentage_already_seen = []
    for uid in userid_list:
        user_profile = ratings.loc[ratings['userId'] == uid]
        user_recommendation = recommendation_users[uid]

        user_profile_mid_list = user_profile['movieId'].to_list()
        user_m_index_list = []
        for m in user_profile_mid_list:
            tmdbId = links_small['tmdbId'].loc[links_small['movieId'] == m]
            try:
                index = movieid_list.index(int(tmdbId))
                user_m_index_list.append(index)
            except ValueError:
                #print("ValueError: " + str(m))
                pass
        
        m_count = 0
        for m_i, cos in user_recommendation[:len(user_profile)]:
            try:
                if int(m_i) in user_m_index_list:
                    m_count += 1
            except ValueError:
                #print("ValueError: " + str(m_i) + " not in tmdbid list") 
                pass  

        #if m_count == len(user_profile):
        #    print(str(uid) + " : the first movies are already viewed")
        #else:
        #    print(str(m_count) + " of " + str(len(user_profile)) + " in first recommended already seen")

        p = m_count/len(user_profile)
        percentage_already_seen.append(p)

    average_percentage = sum(percentage_already_seen) / len(percentage_already_seen)
    print("Average Percentage of already seen movies: " + str(average_percentage))#


def compute_avg_rank(recommendation_users):
    avg_rank = []
    for uid in userid_list:
        user_profile = ratings.loc[ratings['userId'] == uid]
        user_recommendation = recommendation_users[uid]

        user_profile_mid_list = user_profile['movieId'].to_list()
        user_m_index_list = []
        for m in user_profile_mid_list:
            tmdbId = links_small['tmdbId'].loc[links_small['movieId'] == m]
            try:
                index = movieid_list.index(int(tmdbId))
                user_m_index_list.append(index)
            except ValueError:
                #print("ValueError: " + str(m))
                pass
        
        rank_list = []
        for mid in user_m_index_list:
            rank = 0
            for m_i, cos in user_recommendation:
                try:
                    if int(m_i) == int(mid):
                        break
                    rank = rank + 1
                except ValueError:
                    pass
            
            rank_list.append(rank)   
        

        r = sum(rank_list)/len(rank_list)
        avg_rank.append(r)
    print("Max Rank = " + str(len(df)))
    average_rank = sum(rank_list) / len(rank_list)
    print("Average Rank of already seen movies: " + str(average_rank))


def test_user_one(model):
    # Toy Story, Spider-Man, The Avengers, Batman
    mids = [0,4460,15159,497]
    inferred_vector1 = model.infer_vector(tagged_text[0].words, steps=30, alpha=0.025)
    inferred_vector2 = model.infer_vector(tagged_text[4460].words, steps=30, alpha=0.025)
    inferred_vector3 = model.infer_vector(tagged_text[15159].words, steps=30, alpha=0.025)
    inferred_vector4 = model.infer_vector(tagged_text[497].words, steps=30, alpha=0.025)


    test_user_array = np.zeros((4,len(inferred_vector1)))
    test_user_array[0] = inferred_vector1
    test_user_array[1] = inferred_vector2
    test_user_array[2] = inferred_vector3
    test_user_array[3] = inferred_vector4

    #cos_sim = cosine_similarity(test_user_array, test_user_array)
    #print("Cos Sim: ")
    #print(cos_sim)

    test_user = np.mean(test_user_array, axis=0)

    sim = model.dv.most_similar([test_user], topn=len(model.dv))

    #print(sim)
    indexes =[i[0] for i in sim[:10]]
    for i in indexes:
        print(df['title'][i])

    #rank
    rank_list = []
    for mid in mids:
        rank = 0
        print("Mid: " + str(mid))
        for m_i, cos in sim:
            if m_i == mid:
                print("Rank: " + str(rank))
                break
            rank = rank + 1
            
        rank_list.append(rank)   

    print("Max Rank = " + str(len(df)))
    average_rank = sum(rank_list) / len(rank_list)
    print("Average Rank of already seen movies: " + str(average_rank))


def test_user_two(model):
    # The Godfather, The Bourne Identity, The Pianist, Saving Pravite Rayn
    mids = [694,4523,5018,1637]
    inferred_vector1 = model.infer_vector(tagged_text[694].words, steps=30, alpha=0.025)
    inferred_vector2 = model.infer_vector(tagged_text[4523].words, steps=30, alpha=0.025)
    inferred_vector3 = model.infer_vector(tagged_text[5018].words, steps=30, alpha=0.025)
    inferred_vector4 = model.infer_vector(tagged_text[1637].words, steps=30, alpha=0.025)


    test_user_array = np.zeros((4,len(inferred_vector1)))
    test_user_array[0] = inferred_vector1
    test_user_array[1] = inferred_vector2
    test_user_array[2] = inferred_vector3
    test_user_array[3] = inferred_vector4

    #cos_sim = cosine_similarity(test_user_array, test_user_array)
    #print("Cos Sim: ")
    #print(cos_sim)

    test_user = np.mean(test_user_array, axis=0)

    sim = model.dv.most_similar([test_user], topn=len(model.dv))

    print(sim[:10])
    indexes =[i[0] for i in sim[:10]]
    for i in indexes:
        print(df['title'][i])

    #rank
    rank_list = []
    for mid in mids:
        rank = 0
        print("Mid: " + str(mid))
        for m_i, cos in sim:
            if m_i == mid:
                print("Rank: " + str(rank))
                break
            rank = rank + 1
            
        rank_list.append(rank)   

    print("Max Rank = " + str(len(df)))
    average_rank = sum(rank_list) / len(rank_list)
    print("Average Rank of already seen movies: " + str(average_rank))




# x_vector_size, x_min_count, x_window, x_negative, x_epochs, x_hs
default_values = [100, 4, 10, 3, 10, 0]

#test_vec_size = [50,100,150,200,250,300,350,400,450,500]
#test_min_count = [1,2,3,4,5,7,10]
#test_window = [3,5,10,15,20] # NEEDS Bigger?!
#Test Stop
test_vec_size = []
test_min_count = []
test_window = [] 
test_negative = [0,1,2,3,5,10,20]
test_epochs = [10,15,20,30]
test_hs = [0,1]


def test_pass(x_vector_size, x_min_count, x_window, x_negative, x_epochs, x_hs):

    model, inferr_doc_vecs = model_inferred_vectors(x_vector_size, x_min_count, x_window, x_negative, x_epochs, x_hs)
    user_profiles = compute_user_profiles(inferr_doc_vecs)
    recommendation_users = compute_recommendations(user_profiles, model)

    compute_recall(recommendation_users)
    compute_avg_rank(recommendation_users)

    test_user_one(model)
    test_user_two(model)

print("-----------------------------------------------------------------------------")
print("Vector Size Test")
# Vector Size
for v in test_vec_size:
    print("Vector: " + str(v))
    test_pass(v,default_values[1],default_values[2],default_values[3],default_values[4],default_values[5])
    print("-----------------------------------------------------------------------------")

print("Min Count Test")
# Min Count
for c in test_min_count:
    print("Min Count: " + str(c))
    test_pass(default_values[0],c,default_values[2],default_values[3],default_values[4],default_values[5])
    print("-----------------------------------------------------------------------------")

print("Window Test")
# Window
for w in test_window:
    print("Window Test: " + str(w))
    test_pass(default_values[0],default_values[1],w,default_values[3],default_values[4],default_values[5])
    print("-----------------------------------------------------------------------------")

print("Negative Test")
# Negative
for n in test_negative:
    print("Negative: " + str(n))
    test_pass(default_values[0],default_values[1],default_values[2],n,default_values[4],default_values[5])
    print("-----------------------------------------------------------------------------")

print("Epochs Test")
# Epochs
for e in test_epochs:
    print("Epochs: " + str(e))
    test_pass(default_values[0],default_values[1],default_values[2],default_values[3],e,default_values[5])
    print("-----------------------------------------------------------------------------")

print("HS Test")
# HS
for hs in test_hs:
    print("HS: " + str(hs))
    test_pass(default_values[0],default_values[1],default_values[2],default_values[3],default_values[4],hs)
    print("-----------------------------------------------------------------------------")

