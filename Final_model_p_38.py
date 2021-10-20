# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 20:07:15 2020

@author: SUCHARITA
"""

import pandas as pd
import numpy as np
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from math import log, sqrt
import re # for handling string
import string # for handling mathematical operations
import math
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
df= pd.read_csv("F:\\ExcelR\\P_38\\emails.csv")
df.shape #(48076, 5)
df.head()
df.info() # 48076, object(4)
df.describe()
df.isnull().sum() # no null values
df.columns
df['Class'].unique() # abusive, non abusive
df['Class'].value_counts() # abusive (3410), non-abusive (44666) imbalanced dataset
df['content'].unique()
df['content'].value_counts()
df.columns


# creating new dataframe using "content" and "class"
df1= df.iloc[:,3:5]
df1.head(5)

#################
#df1.loc[df1.Class=="Abusive","Class"] = 1
#df1.loc[df1.Class=="Non Abusive","Class"] = 0
###############

df1['Class'].value_counts()
duplicate= df[df1.duplicated()] 
df1= df1.drop_duplicates() 
df1.shape #(24656, 2)
df1['Class'].value_counts() # 0:23014, 1:1642 imbalanced dataset

# text cleaning
df1['cleaned']=df1['content'].apply(lambda x: x.lower()) # remove lower cases
df1['cleaned']=df1['cleaned'].apply(lambda x: re.sub('\w*\d\w*','', x)) # remove digits and words with digits
df1['cleaned']=df1['cleaned'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x)) # remove punctuation
df1['cleaned']=df1['cleaned'].apply(lambda x: re.sub(' +',' ',x)) # remove extra spaces
df1['cleaned']=df1['cleaned'].apply(lambda x: x.split('\n\n')[0])
df1['cleaned']=df1['cleaned'].apply(lambda x: x.split('\n')[0])
df1['cleaned'].head()

# tokenize one sentence from teh dataframe
sample= df1.iloc[100]
print(sample['cleaned'])
print (nltk.word_tokenize(sample['cleaned']))

# tokenise entire df
def identify_tokens(row):
    new = row['cleaned']
    tokens = nltk.word_tokenize(new)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

df1['cleaned'] = df1.apply(identify_tokens, axis=1)
df1['cleaned'].head()

#lemmatization
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in text]

df1['lemma'] = df1['cleaned'].apply(lemmatize_text)
df1['lemma'].head()

# remove stopwords
stop_words = []
with open("F:\\ExcelR\\P_38\\stop.txt") as f:
    stop_words = f.read()

# getting list of stop words
stop_words = stop_words.split("\n")               

def remove_stops(row):
    my_list = row['lemma']
    meaningful_words = [w for w in my_list if not w in stop_words]
    return (meaningful_words)

df1['lemma_meaningful'] = df1.apply(remove_stops, axis=1)
df1['lemma_meaningful'].tail()

# rejoin meaningful stem words in single string like a sentence
def rejoin_words(row):
    my_list = row['lemma_meaningful']
    joined_words = ( " ".join(my_list))
    return joined_words

df1['final'] = df1.apply(rejoin_words, axis=1)


# check the cleaned mails
for index,text in enumerate(df1['final'][40:55]):
  print('Mail %d:\n'%(index+1),text)  



# 1= abusive and 0= non abusive wordcloud

spam= ' '.join(list(df1[df1['Class'] == "Abusive"]['final']))
spam_cloud = WordCloud(width = 512, height = 512).generate(spam)
plt.figure(figsize = (10,8), facecolor = 'k')
plt.imshow(spam_cloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


ham= ' '.join(list(df1[df1['Class'] == "Non Abusive"]['final']))
ham_cloud = WordCloud(width = 512, height = 512).generate(ham)
plt.figure(figsize = (10,8), facecolor = 'k')
plt.imshow(ham_cloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# Preparing email texts into word count matrix format 
mail= df1.loc[:,['final','Class']]
# removing empty rows 
mail.shape #(24656,2)
mail['final'].replace('', np.nan, inplace=True)
mail.dropna(subset=['final'], inplace=True)
mail.shape #(20036,2)


def split_into_words(i):
    return (i.split(" "))

#create vectors from words
from sklearn.feature_extraction.text import CountVectorizer

# Preparing email texts into word count matrix format 
mail_vector = CountVectorizer(analyzer=split_into_words).fit(mail.final)
mail_vector # its is a cv
# vectorising all mails
all_emails_matrix = mail_vector.transform(mail['final'])
all_emails_matrix.shape # (20036, 11783)
type(all_emails_matrix)

# dtm
df_dtm = pd.DataFrame(all_emails_matrix.toarray(), columns=mail_vector.get_feature_names())
df_dtm.index=mail.index
df_dtm.head(3)


# handle imbalance
from imblearn.combine import SMOTEENN 
sme = SMOTEENN(random_state=42)
x_res, y_res = sme.fit_resample(all_emails_matrix, mail['Class'])
z=y_res.to_frame()
z.value_counts()
#Abusive        16896
#Non Abusive    10022
# splitting data into train and test data sets 
pd.DataFrame(x_res.todense(), columns=mail_vector.get_feature_names())
pd.DataFrame(x_res.todense()[y_res == 'Abusive'], columns= mail_vector.get_feature_names()).head(5)

from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x_res,y_res,test_size=0.3)


# For training messages
x_train.shape #(18842, 11783)
type(x_train)

# For testing messages
x_test.shape #(8076, 11783)
type(x_test)

####### Without TFIDF matrices ########################
# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(x_train, y_train)

train_pred_m = classifier_mb.predict(x_train)
accuracy_train_m = np.mean(train_pred_m==y_train) # 98%
confusion_matrix = confusion_matrix(train_pred_m,y_train)
print (confusion_matrix)
print(classification_report(train_pred_m,y_train))

test_pred_m = classifier_mb.predict(x_test)
accuracy_test_m = np.mean(test_pred_m==y_test) # 98%
print(classification_report(test_pred_m,y_test)) #97.47%

# Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(x_res)

# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(x_train)
train_tfidf.shape # (18842, 11783)

# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(x_test)
test_tfidf.shape #  (8076, 11783)

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf,y_train)
train_pred_m_tfidf = classifier_mb.predict(train_tfidf)
accuracy_train_m_tfidf = np.mean(train_pred_m_tfidf==y_train) # 98.46%

print(classification_report(train_pred_m_tfidf,y_train))

test_pred_m_tfidf = classifier_mb.predict(test_tfidf)
accuracy_test_m_tfidf = np.mean(test_pred_m_tfidf==y_test) # 98.26%
print(classification_report(test_pred_m_tfidf,y_test))




