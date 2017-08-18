import pandas as pd
import re
import nltk
import math
from nltk.stem.snowball import SnowballStemmer
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def tf(word,blob):
    return blob.words.count(word)/len(blob.words)

def doc_with_word(word,bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word,bloblist):
    return math.log(len(bloblist)/(1+doc_with_word(word,bloblist)))

def tfidf(word,blob,bloblist):
    return tf(word,blob)*idf(word,bloblist)

dic={'label':[],'data':[]}
# testdata=open('lib\SMSSpamCollection')
testdata=open('lib/20ng-train-no-short.txt')
stop_words=open('lib/stop_words.txt')
sw=[word.rstrip('\n') for word in stop_words.readlines()]
sw=set(sw)
feature=set()

stemmer=SnowballStemmer('english')
bloblist=list()

for i in testdata.readlines():
    # out=re.search('(spam|ham)\s(.*)',i)
    out = re.search('(.*)\t(.*)', i)
    dic['label'].append(out.group(1))
    filter_1=re.sub(r'\W',' ',out.group(2))#remove non alphabets
    filter_2=re.sub(r'\b.{1}\b',' ',filter_1)#remove single character
    filter_3=nltk.word_tokenize(filter_2)#tokenize
    filter_4=[words.lower() for words in filter_3] #lowercase everthing
    filter_7=[stemmer.stem(word) for word in filter_4]
    filter_5=[word for word in filter_7 if word not in sw]
    # feature=feature.union(filter_5)
    dic['data'].append(' '.join(filter_5))
    # bloblist.append(tb(' '.join(filter_5)))

msg=pd.DataFrame(dic)
# print(msg)
allowed=['alt.atheism','talk.religion.misc']
msg=msg.loc[msg['label'].isin(allowed)]
print(msg)
# msg_train, msg_test,label_train,label_test=train_test_split(msg['data'],msg['label'], test_size=0.2)

bag_of_words=CountVectorizer().fit(msg['data'])
# print(len(bag_of_words.vocabulary_))
# print(msg['data'][3])
# print(bag_of_words.transform([msg['data'][3]]))

msgs_bag_of_word=bag_of_words.transform(msg['data'])
# joblib.dump(bag_of_words,"trainedmodels/bow.pkl")
joblib.dump(bag_of_words,"trainedmodels/bow_20.pkl")
# tfidf_bag_of_words=TfidfTransformer().fit(msgs_bag_of_word)
# msgs_tfidf=tfidf_bag_of_words.transform(msgs_bag_of_word)
# # joblib.dump(tfidf_bag_of_words,"trainedmodels/tfidf.pkl")
# joblib.dump(tfidf_bag_of_words,"trainedmodels/tfidf_20.pkl")

detector=MultinomialNB().fit(msgs_bag_of_word,msg['label'])

# msgs_bag_of_word=bag_of_words.transform(msg_test)
# msgs_tfidf=tfidf_bag_of_words.transform(msgs_bag_of_word)

predictions=detector.predict(msgs_bag_of_word)
print("Accuarcy",accuracy_score(msg['label'],predictions))
# joblib.dump(detector,"trainedmodels/spamdetector.pkl")
joblib.dump(detector,"trainedmodels/newsgroup.pkl")



#20 Newsgroup dataset
