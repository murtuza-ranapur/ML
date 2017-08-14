import re
import pandas as pd
from sklearn.metrics import accuracy_score
from collections import Counter
from pprint import pprint
import math

def trans(val):
    if val == '+':
        return "plus"
    else:
        return "minus"

def  length(val):  # 55%
    if len(re.sub(r'\W', '', val)) % 2 != 0:
        return "plus"
    else:
        return "minus"

def vowel(val): # 53%
    if len(re.sub(r'[^aeiou]','',val.lower())) % 2 != 0:
        return "plus"
    else:
        return "minus"

def vowellen(val): # 58%
    newval=re.sub(r'[^aeiou]','',val.lower())
    newval=set(list(newval))
    print(''.join(newval))
    if len(newval) % 2 == 0:
        return "plus"
    else:
        return "minus"

def nameToNumber(val): # 54%
    newval=re.sub(r'[.]', '', val)
    newval=newval.split()
    for i in range(len(newval)):
        newval[i]=len(newval[i])
    if newval[-1] %2 !=0:
        return "plus"
    else:
        return "minus"

def tochar(val):
    newval=re.sub(r'\W', '', val.lower())
    newval=list(newval)
    return newval

def checkVowel(val):
    newval=val[1]
    if newval=='a' or newval=='e'or newval=='i' or newval=='o'or newval=='u':
        return 'plus'
    else:
        return 'minus'

badges=open('lib/badges.txt')
dic={'label':[],'data':[]}
for i in badges.readlines():
    out=re.search(r'(\+|-)\s(.*)',i)
    dic['label'].append(trans(out.group(1)))
    dic['data'].append(out.group(2))

df=pd.DataFrame(dic)
temp=[]
for i in df['data']:
    temp.append(checkVowel(i))

print( "Accuracy" , accuracy_score(df['label'],temp))