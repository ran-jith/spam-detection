# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 14:27:25 2018

@author: Bhavya K
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 17:23:27 2018

@author: Bhavya K
"""

import io
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from  sklearn.feature_extraction.text import CountVectorizer
from  sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
#word_tokenize accepts a string as an input, not a file.
stop_words = set(stopwords.words('english'))
stopwordsrem=[]
label = []
featurecount1 = []
#appendFile = open('fortraining.txt','a')
call_c = 0
call_c1 = 0
call_c2 = 0
ring_c = 0
account_c = 0
claim_c = 0
won_c = 0
free_c = 0
feature = []
with open("SMSSpamCollection.txt") as f:
    for line in f:
        featurecount=[]
        stopwordsrem=[]
        linelength=len(line)
        templine=line[4:linelength]
        words = templine.split()
        for r in words:
            if not r in stop_words:
                stopwordsrem.append(r)
        my_lst_str = ' '.join(stopwordsrem)
        line_to_write=line[:4]+" "+my_lst_str+"\n"
        
        
        
        w_c=len(re.findall(r'\w+', my_lst_str))
        ##print (w_c)
        featurecount.append(w_c)
        ##print (featurecount)
        
        count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
        a_chars = count(my_lst_str, string.ascii_letters)
        a_punct = count(my_lst_str, string.punctuation)
        #print (a_chars)
        #print (a_punct)
        featurecount.append(a_chars)
        featurecount.append(a_punct)
        #print(featurecount)
        
        
        d_c=len(re.findall(r'\d+', my_lst_str))
        featurecount.append(d_c)
        ##print (featurecount)
        
        
        alpha_c=0
        for s in my_lst_str:
        
            m = re.match('([0-9A-Z]+)', s)
            if (m):
                alpha_c=alpha_c+1
        featurecount.append(alpha_c)
        ##print (featurecount)
        
        mob_c=len(re.findall('(?:(?:\\+|0{0,2})91(\\s*[\\-]\\s*)?|[0]?)?[789]\\d{9}', my_lst_str))
        ##print (mob_c)
        featurecount.append(mob_c)
        ##print (featurecount)
        
        
        caps_c=len(re.findall('[A-Z]+',my_lst_str))
        ##print(caps_c)
        featurecount.append(caps_c)
        #print(featurecount)
        
        urls_c = len(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', my_lst_str))
        #print (urls)
        #url_c=len(re.findall(r'(ftp|http)://.*\.(com|in|org)$', my_lst_str))
        #print(url_c)
        featurecount.append(urls_c)
        #print(featurecount)
        
        
        #quote_cn=len(re.findall('\', my_lst_str))
        #print (quote_cn)
        
        
        term = "call"
            
        if (term in line):
            call_c = call_c+1
        else:
            call_c = 0
        #print (call_c)
        featurecount.append(call_c)
        #print (featurecount)
        
        term1 = "Call"
        
        if (term1 in line):
            call_c1 = call_c1+1
        else:
            call_c1 = 0
        featurecount.append(call_c1)
        #print (featurecount)
        #alpha_c=len(re.findall('&v=[0-9]+', my_lst_str))
        #print(alpha_c)
        
        term2= "cash"
        if (term2 in line):
            call_c2 = call_c2+1
        else:
            call_c2=0
        featurecount.append(call_c2)
        #print(featurecount)
        
        term3= "FREE"
        if(term3 in line):
            ring_c = ring_c+1
        else:
            ring_c = 0
        featurecount.append(ring_c)
        
        term4= "FREE"
        if(term4 in line):
            free_c= free_c+1
        else:
            free_c = 0
        featurecount.append(free_c)
    
            
        term4 = "claim"
        if(term4 in line):
            claim_c = claim_c+1
        else:
            claim_c = 0
        featurecount.append(claim_c)
        #print (featurecount)
        
        #print (type(a)) 
        #print (a)
        feature.append(featurecount)
#print(len(feature))
        
        tline = line[0:4]
        
        w = tline.split()
        
        #print (w)
        #print (tline)
        for k in w:
            if (k == 'spam'):
                label.append(1)
            elif (k == 'ham'):
                label.append(0)
        
#print(label)
#print (len(label))
#print(label)
#print(feature)    
a=np.array(feature)  
b=np.array(label)
b1=np.reshape(b,(-1,1))
f.close()
#print(featurecount)



x_train,x_test,y_train,y_test=train_test_split(a,b1,test_size=0.05,random_state=2)
#print(x_train.shape)
#print(y_train.shape)
model= RandomForestClassifier()
model.fit(x_train,y_train )
predicted= model.predict(x_test)
print("Accuracy is",(accuracy_score(y_test, predicted)))
Accuracy = (accuracy_score(y_test, predicted)*100)
print("Accuracy =",Accuracy)
Avg_Precision= (average_precision_score(y_test, predicted)*100)
print ("Average Precision=",Avg_Precision)
Precision = (precision_score(y_test, predicted)*100)
print("Precision=", Precision)
Recall= ((recall_score(y_test, predicted))*100)
print ("Recall=", Recall)
f1_score= (f1_score(y_test, predicted)*100)
print("f1_score", f1_score)   

test_sentence = input("enter a message")        
	
w_c=len(re.findall(r'\w+', test_sentence))
        ##print (w_c)
featurecount1.append(w_c)
        ##print (featurecount)
        
count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
a_chars = count(test_sentence, string.ascii_letters)
a_punct = count(test_sentence, string.punctuation)
        #print (a_chars)
        #print (a_punct)
featurecount1.append(a_chars)
featurecount1.append(a_punct)
        #print(featurecount)
        
        
d_c=len(re.findall(r'\d+', test_sentence))
featurecount1.append(d_c)
        ##print (featurecount)
        
        
alpha_c=0
for s in test_sentence:
        
    m = re.match('([0-9A-Z]+)', s)
    if (m):
        alpha_c=alpha_c+1
featurecount1.append(alpha_c)
        ##print (featurecount)
        
mob_c=len(re.findall('(?:(?:\\+|0{0,2})91(\\s*[\\-]\\s*)?|[0]?)?[789]\\d{9}', test_sentence))
        ##print (mob_c)
featurecount1.append(mob_c)
        ##print (featurecount)
        
        
caps_c=len(re.findall('[A-Z]+',test_sentence))
        ##print(caps_c)
featurecount1.append(caps_c)
        #print(featurecount)
        
urls_c = len(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', test_sentence))
        #print (urls)
        #url_c=len(re.findall(r'(ftp|http)://.*\.(com|in|org)$', my_lst_str))
        #print(url_c)
featurecount1.append(urls_c)
        #print(featurecount)
        
        
        #quote_cn=len(re.findall('\', my_lst_str))
        #print (quote_cn)
        
        
term = "call"
if (term in test_sentence):
    call_c = call_c+1
else:
    call_c = 0
        #print (call_c)
featurecount1.append(call_c)
        #print (featurecount)
        
term1 = "Call"
if (term1 in test_sentence):
    call_c1 = call_c1+1
else:
    call_c1 = 0
featurecount1.append(call_c1)
        #print (featurecount)
        #alpha_c=len(re.findall('&v=[0-9]+', my_lst_str))
        #print(alpha_c)
        
term2= "cash"
if (term2 in test_sentence):
    call_c2 = call_c2+1
else:
    call_c2=0
featurecount1.append(call_c2)
        #print(featurecount)
        
term3= "FREE"
if(term3 in test_sentence):
    ring_c = ring_c+1
else:
    ring_c = 0
featurecount1.append(ring_c)
        
term4= "FREE"
if(term4 in test_sentence):
    free_c= free_c+1
else:
    free_c = 0
featurecount1.append(free_c)
term4 = "claim"
if(term4 in test_sentence):
    claim_c = claim_c+1
else:
    claim_c = 0
featurecount1.append(claim_c)
#print(featurecount1)
test_x = np.array(featurecount1)
test_x1=np.reshape(test_x,(1,-1))
#print(test_x1.shape)
#print(type(test_x1))
predicted= model.predict(test_x)
#print(predicted)
if (predicted == 1):
    print("Message is SPAM")
else:
    print("Message is HAM")


        
        
        

                
                

    
        
        
            
                
            
                
        
        
        
        
              
        
        
        
       
                
        
        
        
        
   
        
        

      
