import pickle
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from filter import Filter
import re
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

file=open('trainedmodels/mi_bow.pkl','rb')
bow=pickle.load(file) #Load features

myfilter=Filter()
reObject=re.compile('(.*)\t(.*)')
df=myfilter.getFiltered(reObject,open('lib/SMSSpamCollection'))

for_burnolli=CountVectorizer(vocabulary=bow)
joblib.dump(for_burnolli,'trainedmodels/bow_mi.pkl')
vector=for_burnolli.transform(df['data'])

detector=BernoulliNB().fit(vector,df['label'])
joblib.dump(detector,'trainedmodels/bernoulli.pkl')
predictions=detector.predict(vector)
print("Accuracy :",accuracy_score(df['label'],predictions))

