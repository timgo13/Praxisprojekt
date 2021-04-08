import numpy as np
import pandas as pd
import sklearn

movies = pd.read_csv('..\movie_archive\movies_metadata.csv')

print(movies.info())