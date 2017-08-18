from sklearn.externals import joblib
import re
from filter import Filter
from sklearn.metrics import accuracy_score

bow=joblib.load('trainedmodels/bow_mi.pkl')
bernoulli=joblib.load('trainedmodels/bernoulli.pkl')

file = open("lib/sms_for_test.txt")
reObject=re.compile(r'(.*),(spam|ham)$')

myfil=Filter()
msg = myfil.getFiltered(reObject,file)


vectors=bow.transform(msg['label'])
predictions=bernoulli.predict(vectors)
print("Accuracy :",accuracy_score(predictions,msg['data']))