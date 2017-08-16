import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

def floor(x):
    if math.isnan(x):
        return 0.0
    else:
        return math.floor(x)
df = pd.read_csv("lib/movie_corpus.csv")
df.plot.scatter(x="imdb_score",y="facenumber_in_poster")
print(df["imdb_score"].apply(floor))