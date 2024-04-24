#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Saturday Apr 07 2023 at 16:16:19

@author: Archana
"""
import csv,json,os,re,sys
import nltk
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import statistics as st
import torch
import wandb
#import torchvision
from torch.utils.data import DataLoader
from nltk.corpus import stopwords
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
import joblib
#from sklearn.externals import joblib
from sklearn import svm 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, train_test_split 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix 
#from gensim.models import LogEntropyModel
#from gensim.corpora import Dictionary
#from gensim.models.doc2vec import Doc2Vec,TaggedDocument 
from collections import Counter
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import LongformerTokenizer,LongformerForSequenceClassification
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification,Trainer, TrainingArguments

# from transformers import BertTokenizer, BertForSequenceClassification
# from transformers import LongformerTokenizer,LongformerForSequenceClassification
# from transformers import RobertaTokenizer, RobertaForSequenceClassification,Trainer, TrainingArguments 

#from transformers import RobertaTokenizerFast, RobertaForMaskedLM, RobertaForSequenceClassification
#from transformers import AutoTokenizer, AutoModelForMaskedLM
#from transformers import Trainer, TrainingArguments
from tqdm import tqdm



#torch.cuda.empty_cache()

import warnings
warnings.filterwarnings("ignore")

CUDA_LAUNCH_BLOCKING=1
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["WANDB_DISABLED"] = "true"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"


# Release memory of a list immediately after it's use
def release_memory(lt=[]):
   del lt[:]
   del lt

en_stopwords = ['a', 'about', 'above', 'across', 'after', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'among', 'an', 'and', 'another', 'any', 'anybody', 'anyone', 'anything', 'anywhere', 'are', 'area', 'areas', 'around', 'as', 'ask', 'asked', 'asking', 'asks', 'at', 'away', 'b', 'back', 'backed', 'backing', 'backs', 'be', 'became', 'because', 'become', 'becomes', 'been', 'before', 'began', 'behind', 'being', 'beings', 'best', 'better', 'between', 'big', 'both', 'but', 'by', 'c', 'came', 'can', 'cannot', 'case', 'cases', 'certain', 'certainly', 'clear', 'clearly', 'come', 'could', 'd', 'did', 'differ', 'different', 'differently', 'do', 'does', 'done', 'down', 'down', 'downed', 'downing', 'downs', 'during', 'e', 'each', 'early', 'either', 'end', 'ended', 'ending', 'ends', 'enough', 'even', 'evenly', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'f', 'face', 'faces', 'fact', 'facts', 'far', 'felt', 'few', 'find', 'finds', 'first', 'for', 'four', 'from', 'full', 'fully', 'further', 'furthered', 'furthering', 'furthers', 'g', 'gave', 'general', 'generally', 'get', 'gets', 'give', 'given', 'gives', 'go', 'going', 'good', 'goods', 'got', 'great', 'greater', 'greatest', 'group', 'grouped', 'grouping', 'groups', 'h', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'herself', 'high', 'high', 'high', 'higher', 'highest', 'him', 'himself', 'his', 'how', 'however', 'i', 'if', 'important', 'in', 'interest', 'interested', 'interesting', 'interests', 'into', 'is', 'it', 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kind', 'knew', 'know', 'known', 'knows', 'l', 'large', 'largely', 'last', 'later', 'latest', 'least', 'less', 'let', 'lets', 'like', 'likely', 'long', 'longer', 'longest', 'm', 'made', 'make', 'making', 'man', 'many', 'may', 'me', 'member', 'members', 'men', 'might', 'more', 'most', 'mostly', 'mr', 'mrs', 'much', 'must', 'my', 'myself', 'n', 'necessary', 'need', 'needed', 'needing', 'needs', 'never', 'new', 'new', 'newer', 'newest', 'next', 'no', 'nobody', 'non', 'noone', 'not', 'nothing', 'now', 'nowhere', 'number', 'numbers', 'o', 'of', 'off', 'often', 'old', 'older', 'oldest', 'on', 'once', 'one', 'only', 'open', 'opened', 'opening', 'opens', 'or', 'order', 'ordered', 'ordering', 'orders', 'other', 'others', 'our', 'out', 'over', 'p', 'part', 'parted', 'parting', 'parts', 'per', 'perhaps', 'place', 'places', 'point', 'pointed', 'pointing', 'points', 'possible', 'present', 'presented', 'presenting', 'presents', 'problem', 'problems', 'put', 'puts', 'q', 'quite', 'r', 'rather', 'really', 'right', 'right', 'room', 'rooms', 's', 'said', 'same', 'saw', 'say', 'says', 'second', 'seconds', 'see', 'seem', 'seemed', 'seeming', 'seems', 'sees', 'several', 'shall', 'she', 'should', 'show', 'showed', 'showing', 'shows', 'side', 'sides', 'since', 'small', 'smaller', 'smallest', 'so', 'some', 'somebody', 'someone', 'something', 'somewhere', 'state', 'states', 'still', 'still', 'such', 'sure', 't', 'take', 'taken', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'therefore', 'these', 'they', 'thing', 'things', 'think', 'thinks', 'this', 'those', 'though', 'thought', 'thoughts', 'three', 'through', 'thus', 'to', 'today', 'together', 'too', 'took', 'toward', 'turn', 'turned', 'turning', 'turns', 'two', 'u', 'under', 'until', 'up', 'upon', 'us', 'use', 'used', 'uses', 'v', 'very', 'w', 'want', 'wanted', 'wanting', 'wants', 'was', 'way', 'ways', 'we', 'well', 'wells', 'went', 'were', 'what', 'when', 'where', 'whether', 'which', 'while', 'who', 'whole', 'whose', 'why', 'will', 'with', 'within', 'without', 'work', 'worked', 'working', 'works', 'would', 'x', 'y', 'year', 'years', 'yet', 'you', 'young', 'younger', 'youngest', 'your', 'yours', 'z']
nltk_stopwords=list(set(stopwords.words('english')))
for word in nltk_stopwords:
    if word not in en_stopwords:
        en_stopwords.append(word)

# Transformer model encodings for whole data passed at a time
class get_torch_data_format(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

# class get_torch_data_format(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self,idx):
#         current_sample = self.encodings[idx :]
#         current_target = self.labels[idx]
#         return {
#             "x": current_sample,  #tensor.to(torch.float64)
#             "y": current_target}

# Transformer model encodings for whole data passed batch by batch
class get_validation_data_format(torch.utils.data.Dataset):
   def __init__(self, encodings):
       self.encodings = encodings
       self.device = torch.device('cuda')
   
   def __getitem__(self, idx):
       item = {k: v[idx].clone().detach().to(device=self.device) for k, v in self.encodings.items()}
       return item

   def __len__(self):
       return len(self.encodings['input_ids'])

# BERT model accuracy function
def compute_metrics(pred):
     labels = pred.label_ids
     preds = pred.predictions.argmax(-1)
     acc = accuracy_score(labels, preds)
     return {
         'accuracy': acc,
     }     

 # Transformer models    
def bert_training_model(model,model_source,model_path,trn_data,trn_cat,test_size=0.20,max_length=512):
   print('\n ***** Running BERT Model ***** \n')       
   torch.cuda.empty_cache() 
   
   labels=np.asarray(trn_cat)     # Class labels in nparray format      
   
   (train_texts, valid_texts, train_labels, valid_labels), class_names = train_test_split(trn_data, labels, stratify=trn_cat,test_size=test_size), trn_cat
# Model selection

   if model=='bert':   
       print('\n ***** Running BERT-basecase Model ***** \n')
       tokenizer = BertTokenizerFast.from_pretrained(model_source, do_lower_case=True) 
       model = BertForSequenceClassification.from_pretrained(model_source, num_labels=len(class_names)).to("cuda")
   elif model=='roberta':            
       print('\n ***** Running RoBERTa-base Model ***** \n')  
       tokenizer = RobertaTokenizerFast.from_pretrained(model_source, do_lower_case=True,ignore_mismatched_sizes=True)
       model = RobertaForSequenceClassification.from_pretrained(model_source, num_labels=len(class_names)).to("cuda")
   elif model=='longformer':
       print('\n ***** Running BERT-Longformer Model ***** \n')
       model = LongformerForSequenceClassification.from_pretrained(model_source, num_labels=len(class_names)).to("cuda")
       tokenizer = LongformerTokenizer.from_pretrained(model_source, do_lower_case=True)
   else:
      print('\n Error!!! Please select a valid transformer model \n')
      sys.exit(0)                
   
   train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
   valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length) 
   train_dataset = get_torch_data_format(train_encodings, train_labels)
   valid_dataset = get_torch_data_format(valid_encodings, valid_labels)
   
   print("############## Tokenisation DONE #################")
   training_args = TrainingArguments(
       output_dir='./results',          # output directoryEngineering Optimization: Theory and Practice
       num_train_epochs=100,            # total number of training epochs
       per_device_train_batch_size=16,   # batch size per device during training 
       per_device_eval_batch_size=20,   # batch size for evaluation
       warmup_steps=500,                # number of warmup steps for learning rate scheduler
       fp16 = True,
       weight_decay=0.01,               # strength of weight decay
       logging_dir='./logs',            # directory for storing logs
       dataloader_num_workers = 8,
       load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
       logging_steps=100,               # log & save weights each logging_steps
       evaluation_strategy="steps",     # evaluate each `logging_steps`
       max_steps= 100
       )    
   trainer = Trainer(
       model=model,                         # the instantiated Transformers model to be trained
       args=training_args,                  # training arguments, defined above
       train_dataset=train_dataset,         # training dataset
       eval_dataset=valid_dataset,          # evaluation dataset
       compute_metrics=compute_metrics,  # the callback that computes metrics of interest
       )
   
   print('\n Trainer done \n')
   trainer.train()
   print('\n Trainer train done \n')        
   trainer.evaluate()
   print('\n Save Model \n')
   model.save_pretrained(model_path)
   tokenizer.save_pretrained(model_path)
   torch.cuda.empty_cache()
   print("Hello !! Transformer has saved model and tokenizer")
   return model,tokenizer,class_names 

################## MODELS
model = 'bert'
model_source = 'bert-base-cased'
model_path  = './bert_model'   

# model='longformer'
# model_source='allenai/longformer-base-4096'
# model_path = './long_former'

# model='roberta'
# model_source='roberta-base'
# model_path = './roberta'

################# Reading dataframe
# valid_data = pd.read_csv('/home/guest/Archana/Sample data/val_df_sample.csv')
# tst_data = pd.read_csv('/home/guest/Archana/Sample data/test_df_sample.csv')

# valid_data = valid_data.dropna()
# tst_data = tst_data.dropna()

# ###### Label encoding for Sentiments
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# c = le.fit_transform(valid_data['label'])
# valid_data['label_1'] = c.tolist()

# c = le.fit_transform(tst_data['label'])
# tst_data['label_1'] = c.tolist()


# valid_texts = list(valid_data['text'])#.tolist()
# valid_labels = list(valid_data['label_1'])#.tolist()
# tst_texts = list(tst_data['text'])#.tolist()
# tst_labels = list(tst_data['label_1'])#.tolist()

# data = pd.read_csv('/home/guest/Archana/val_test.csv')
# data = data.dropna()
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# c = le.fit_transform(data['label'])
# data['label_1'] = c.tolist()
# texts = list(data['text'])#.tolist()
# labels = list(data['label_1'])#.tolist()

data = pd.read_csv('/home/guest/Archana/sheet2_val_test.csv', encoding= 'unicode_escape')
data = data.dropna()

########### Label encoding
# le = LabelEncoder()
# c = le.fit_transform(data['label'])
# data['label_1'] = c.tolist()
texts = list(data['Cleaned'])
labels = list(data['Label'])

############### Valid-test splitting
valid_texts,tst_texts,valid_labels,tst_labels = train_test_split(texts,labels, test_size=0.20,stratify=labels)

############ training bert model
trn_model,trn_tokenizer,class_names = bert_training_model(model,model_source,model_path,valid_texts,valid_labels)  

print('\n ***** Processing Test Documents ***** \n')
predicted=[]; predicted_probability=[]; 
for doc in tst_texts:
    tst_encodings = trn_tokenizer(doc, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")        
    tst_dataset = get_validation_data_format(tst_encodings)
    tst_loader = DataLoader(tst_dataset,batch_size=48)
    with torch.no_grad():
        for batch in tst_loader:
                outputs = trn_model(**batch)
                probs = outputs[0].softmax(1)
                cl=class_names[probs.argmax()]
                predicted.append(cl)
                predicted_probability.append(probs)  



        
