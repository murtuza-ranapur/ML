import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
import re

class Filter:
    def __init__(self):
        self.dic={'label':[],'data':[]}
        self.stemmer=SnowballStemmer('english')
        stop_words = open('lib\stop_words.txt')
        sw = [word.rstrip('\n') for word in stop_words.readlines()]
        self.sw = set(sw)

    def getFiltered(self,reObject,file):
        for i in file.readlines():
            out=reObject.search(i)
            self.dic['label'].append(out.group(2))
            filter_1 = re.sub(r'\W', ' ', out.group(1))  # remove non alphabets
            filter_2 = re.sub(r'\b.{1}\b', ' ', filter_1)  # remove single character
            filter_3 = nltk.word_tokenize(filter_2)  # tokenize
            filter_4 = [words.lower() for words in filter_3]  # lowercase everthing
            filter_7 = [self.stemmer.stem(word) for word in filter_4] #Perform stemming
            filter_5 = [word for word in filter_7 if word not in self.sw] #remove Stop Words
            self.dic['data'].append(' '.join(filter_5))

        return pd.DataFrame(self.dic)
