####### Importing libraries
import pandas as pd
import nltk
import numpy as np
import csv
import re
import string
import nltk
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet') 
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

#importing the training dataset
train = pd.read_csv('/home/tanmay/Archana/paper2/data/Dataset_tripadvisor_reviews_agra.csv')

#################################################################### DATA PRE-PROCESSING #####################################################

############ cleaning dataset
def text_cleaning(text):
    text = [re.sub(r'@\S+', '', t) for t in text ]
    text = [re.sub(r'#', '', t) for t in text ]
    text = [re.sub(r"https?\S+", '', t) for t in text ]
    text = [re.sub(r"\d*", '', t) for t in text ]    
    text = [re.sub(r"[+|-|*|%]", '', t) for t in text ]  
    text = [re.sub(r"[^^(éèêùçà)\x20-\x7E]", '', t) for t in text]
    return text
    
train['Cleaned'] = text_cleaning(train['review'])

########### removing punctuation
def remove_punctuation(text):
    punctuationfree = "".join([i for i in str(text) if i not in string.punctuation])
    return punctuationfree
train['Cleaned']= train['Cleaned'].apply(lambda x:remove_punctuation(x))

############ Removing Stopwords
stop_words = stopwords.words('english')
train['Cleaned'] = train['Cleaned'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

############# Changing to lower case
train['Cleaned']= train['Cleaned'].apply(lambda x: x.lower())

############## Tokenisation
train['Tokenised'] = train.apply(lambda row: nltk.word_tokenize(row['Cleaned']), axis=1)

############## Stemming
porter_stemmer = PorterStemmer()
def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text

train['Stemmed_Text']=train['Tokenised'].apply(lambda x: stemming(x))

############## Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text
    
train['Lemmatized_Text']=train['Stemmed_Text'].apply(lambda x:lemmatizer(x))

print(train)

########################################################### Pre-train model (VADER) #########################################################

############### creating an object of sentiment intensity analyzer
sia= SentimentIntensityAnalyzer()

############## creating new column scores using polarity scores function
train['scores']=train['Cleaned'].apply(lambda body: sia.polarity_scores(str(body)))
train['compound']=train['scores'].apply(lambda score_dict:score_dict['compound'])
train['pos']=train['scores'].apply(lambda pos_dict:pos_dict['pos'])
train['neg']=train['scores'].apply(lambda neg_dict:neg_dict['neg'])

train['Type']=''
train.loc[train.compound>0,'Type']='Positive'
train.loc[train.compound==0,'Type']='Neutral'
train.loc[train.compound<0,'Type']='Negative'

############### different type of comments
len=train.shape
(rows,cols)=len
pos=0
neg=0
neutral=0
for i in range(0,rows):
  if train.loc[i][14]=="Positive":
      pos=pos+1
  if train.loc[i][14]=="Negative":
      neg=neg+1
  if train.loc[i][14]=="Neutral":
      neutral=neutral+1

print("Positive :"+str(pos) + " Negative :" + str(neg) + " Neutral :"+ str(neutral))

########################################################## Training the Model ################################################################

