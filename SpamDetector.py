from sklearn.externals import joblib
import re
from filter import Filter
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_20newsgroups
bow=joblib.load('trainedmodels/bow_20.pkl')
tfidf=joblib.load('trainedmodels/tfidf_20.pkl')
detector=joblib.load('trainedmodels/newsgroup.pkl')

# file = open("lib/sms_for_test.txt")
file = open("lib/20ng-test-no-short.txt")
#reObject=re.compile(r'(.*),(spam|ham)$')
reObject=re.compile(r'(.*)\t(.*)')
filter=Filter()
msg = filter.getFiltered(reObject,file)
allowed=['alt.atheism','talk.religion.misc']
msg=msg.loc[msg['label'].isin(allowed)]

msgs_bag_of_word=bow.transform(msg['data'])
# msgs_tfidf=tfidf.transform(msgs_bag_of_word)
predictions=detector.predict(msgs_bag_of_word)
print("Accuarcy",accuracy_score(msg['label'],predictions))