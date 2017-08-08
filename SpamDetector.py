from sklearn.externals import joblib
import re
from filter import Filter
from sklearn.metrics import accuracy_score
bow=joblib.load('trainedmodels/bow.pkl')
tfidf=joblib.load('trainedmodels/tfidf.pkl')
detector=joblib.load('trainedmodels/spamdetector.pkl')


file = open("lib/sms_for_test.txt")
reObject=re.compile(r'(.*),(spam|ham)$')
filter=Filter()
msg = filter.getFiltered(reObject,file)
print(msg)
msgs_bag_of_word=bow.transform(msg['data'])
msgs_tfidf=tfidf.transform(msgs_bag_of_word)
predictions=detector.predict(msgs_tfidf)
print("Accuarcy",accuracy_score(msg['label'],predictions))