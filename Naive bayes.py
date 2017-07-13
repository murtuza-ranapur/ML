import pandas as pd
import re
import nltk
import math

def sortme(val):
    return dic[val]

def getFeatureCount(tag):
    one = df.data.apply(lambda x: tag in x)
    df1 = df[one]
    print(df1.shape[0])
    return df1.shape[0]

def getFeatureCountGivenClass(tag,cls):
    one = df.data.apply(lambda x: tag in x)
    df1 = df[one]
    print(df1[df1.label == cls].shape[0])
    return df1[df1.label == cls].shape[0]

def pf(tag):
    return getFeatureCount(tag)/df.shape[0]

def pc(i):
    one = df.label.apply(lambda x: i in x)
    df1 = df[one]
    return df1.shape[0] / df.shape[0]

def p(feature,cls):
    return getFeatureCountGivenClass(feature,cls)/getFeatureCount(feature)

def mutual_information(feature,case):
    summ=0
    for i in case:
        a=p(feature,i)
        print(a)
        if a>0:
            summ+=a*math.log2(a/pf(feature)/pc(i))
    return summ

dic={'label':[],'data':[]}
testdata=open('lib\SMSSpamCollection')
stop_words=open('lib\stop_words.txt')
sw=[word.rstrip('\n') for word in stop_words.readlines()]
sw=set(sw)
feature=set()
for i in testdata.readlines():
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

dic={}
print(mutual_information('free',['spam','ham']))
# for i in feature:
#     dic[i]=mutual_information(i,['spam','ham'])
# print( sorted(dic ,key=sortme,reverse=True))

