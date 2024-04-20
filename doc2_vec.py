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
from collections import Counter
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import LongformerTokenizer,LongformerForSequenceClassification
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification,Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertModel
from transformers import BertConfig
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
#import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
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

class dataset_maker(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        current_sample = self.encodings[idx, :]
        current_target = self.labels[idx]
        return {
            "x": current_sample,  #tensor.to(torch.float64)
            "y": current_target}

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

############################ Labeled data
data = pd.read_csv('/home/guest/Archana/val_test.csv')
data = data.dropna()

########### Label encoding
le = LabelEncoder()
c = le.fit_transform(data['label'])
data['label_1'] = c.tolist()
texts = list(data['text'])
labels = list(data['label_1'])

########## train-test-split on labeled data
valid_texts,tst_texts,valid_labels,tst_labels = train_test_split(texts,labels, test_size=0.20,stratify=labels)

################# Pre-train data
train_df_1 = pd.read_csv('/home/guest/Archana/Data/training_data.csv')
train_data_ = train_df_1.dropna()
train_data = list(train_data_['Cleaned'])

print("################### Creating Doc2Vec Enbeddings ################")
card_docs = [TaggedDocument(doc.split(' '), [i]) for i, doc in enumerate(train_data_.Cleaned)]
#model_doc = Doc2Vec(vector_size=768, min_count=1, epochs = 20)
#instantiate model
model_doc = Doc2Vec(vector_size=768, window=2, min_count=1, workers=8, epochs = 40)
#build vocab
model_doc.build_vocab(card_docs)
#train model
model_doc.train(card_docs, total_examples=model_doc.corpus_count, epochs=model_doc.epochs)

#generate vectors
classes = [[1,0,0],[0,1,0],[0,0,1]]
valid_label = torch.tensor([classes[i] for i in valid_labels])
tst_label = torch.tensor([classes[i] for i in tst_labels])

valid2vec_ = [model_doc.infer_vector(i.split(' ')) for i in valid_texts]
test2vec_ = [model_doc.infer_vector(i.split(' ')) for i in tst_texts]

valid2vec = torch.tensor(valid2vec_)
test2vec = torch.tensor(test2vec_)

train_dataset = dataset_maker(valid2vec, valid_label)
valid_dataset = dataset_maker(test2vec, tst_label)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

#########################################################################
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch

epochs = 1

model_bert = BertModel.from_pretrained("bert-base-uncased")

classifier = torch.nn.Sequential(
            nn.Linear(768, 50),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(50, 3)
        )

# for param in model_bert.parameters():
#     param.requires_grad = False
# classifier = torch.nn.Linear(768,3)

softmax = torch.nn.Softmax()

lr = 1e-4

optimizer1 = torch.optim.Adam(
    (p for p in classifier.parameters() if p.requires_grad), lr = lr)

optimizer2 = torch.optim.Adam(
    (p for p in model_bert.parameters() if p.requires_grad), lr = lr)

criterion = CrossEntropyLoss()

for epoch in tqdm(range(epochs)):
    print("Training BERT model")
    for batch in train_loader:
        label = batch['y'].float()
        inside = batch['x'].unsqueeze(1)
        output_ = model_bert(inputs_embeds = inside).pooler_output
        output1 = classifier(output_).squeeze().float()
        ########## normalisation
        #output1 = torch.nn.functional.normalize(output, p=2.0, dim = 0)
        output = softmax(output1)
        # print("Softmax output: ",output1)
        #output1 = softmax(output)
        loss = criterion(output,label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_bert.parameters(),0.5)
        torch.nn.utils.clip_grad_norm_(classifier.parameters(),0.5)
        optimizer1.step()
        optimizer2.step()

print("Training done")

########################################## Testing model

pred = []
label = []

for batch in valid_loader:
    label_ = batch['y']
    inside_ = batch['x'].unsqueeze(1)
    output = model_bert(inputs_embeds = inside_).last_hidden_state
    output1 = classifier(output).squeeze()
    output = softmax(output1)
    predictions = output.detach().numpy().tolist()
    labels = label_.detach().numpy().tolist()

    pred.append(np.array([i.index(max(i)) for i in predictions]))
    label.append(np.array([i.index(max(i)) for i in labels]))

prediction = np.concatenate(pred)
prediction_ = np.round(prediction)
labels = np.concatenate(label)

print("Testing Done")

from sklearn.metrics import classification_report
result = classification_report(labels, prediction)
print(result)

accuracy = accuracy_score(labels,prediction)
print("Accuracy: ",accuracy)
