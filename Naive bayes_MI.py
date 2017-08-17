import pandas as pd
import re
import nltk
import math
from pprint import pprint
import dill
from tqdm import tqdm
import pickle

def sortme(val):
    return midic[val]

def getFeatureCount(tag):
    one = df.data.apply(lambda x: tag in x)
    df1 = df[one]
    return df1.shape[0]

def getFeatureCountGivenClass(tag,cls):
    one = df.data.apply(lambda x: tag in x)
    df1 = df[one]
    return df1[df1.label == cls].shape[0]

def prob_calac_for(containsTerm,belogsToClass,word,case):
    if containsTerm:
        if belogsToClass:
            return getFeatureCountGivenClass(word,case)
        else:
            return getFeatureCount(word)-getFeatureCountGivenClass(word,case)
    else:
        if belogsToClass:
            return dic['label'].count(case)-getFeatureCountGivenClass(word,case)
        else:
            return len(dic['label'])-dic['label'].count(case)-(getFeatureCount(word)-getFeatureCountGivenClass(word,case))

def mi_approach_2(word,case):
    N11=prob_calac_for(containsTerm=True,belogsToClass=True,word=word,case=case)
    N10=prob_calac_for(containsTerm=True,belogsToClass=False,word=word,case=case)
    N01=prob_calac_for(containsTerm=False,belogsToClass=True,word=word,case=case)
    N00=prob_calac_for(containsTerm=False,belogsToClass=False,word=word,case=case)
    N=N11+N10+N01+N00
    N1_=N11+N10
    N_1=N01+N11
    N0_=N01+N00
    N_0=N00+N10
    # print("{} / {} * math.log({} * {} / {} / {})".format(N10,N,N,N10,N1_,N_0))
    info_gain=(N11+1) / (N+len(feature)) * math.log2(((N * N11)+1) / ((N1_ * N_1)+len(feature))) + \
              (N01+1) / (N+len(feature)) * math.log2(((N * N01)+1) / ((N0_ * N_1)+len(feature))) + \
              (N10+1) / (N+len(feature)) * math.log2(((N * N10)+1) / ((N1_ * N_0)+len(feature))) + \
              (N00+1) / (N+len(feature)) * math.log2(((N * N00)+1) / ((N0_ * N_0)+len(feature)))
    return info_gain

dic={'label':[],'data':[]}
testdata=open('lib\SMSSpamCollection')
stop_words=open('lib\stop_words.txt')
sw=[word.rstrip('\n') for word in stop_words.readlines()]
sw=set(sw)
feature=set()
for i in tqdm(testdata.readlines()):
    out=re.search('(spam|ham)\s(.*)',i)
    dic['label'].append(out.group(1))
    filter_1=re.sub(r'\W',' ',out.group(2))#remove non alphabets
    filter_2=re.sub(r'\b.{1}\b',' ',filter_1)#remove single character
    filter_3=nltk.word_tokenize(filter_2)#tokenize
    filter_4=[words.lower() for words in filter_3] #lowercase everthing
    filter_5=set(filter_4) #Convertig to set
    filter_6=filter_5.difference(sw)# removing stop words
    feature=feature.union(filter_6)
    dic['data'].append(list(filter_6))

df=pd.DataFrame(dic)
midic={}
# print(mi_approach_2('free','spam'))
for i in tqdm(feature):
    midic[i]=mi_approach_2(i,'spam')
Bow=sorted(midic ,key=sortme,reverse=True)[:1000]
file=open('trainedmodels/mi_bow.pkl','wb')
pickle.dump(Bow,file)
pprint(Bow)