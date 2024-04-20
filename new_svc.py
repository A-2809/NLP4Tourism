import sys
import pandas as pd
from sklearn import svm
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest,chi2,mutual_info_classif
from sklearn.metrics import classification_report

############ reading csv
val = pd.read_csv('/home/guest/Archana/Data/val - Sheet1.csv')
test = pd.read_csv('/home/guest/Archana/Data/test - Sheet1.csv')

######## dropping nan values
# val = val.dropna()
# test = test.dropna()

####### spiliting already done
X_train = list(val['review'])
Y_train = list(val['Label'])
X_test = list(test['review'])
Y_test = list(test['Label'])


no_of_selected_terms=500

def tfidf_classification_model(trn_data,trn_cat):
    print('\n ***** Building TF-IDF Based Training Model ***** \n')
    clf = svm.SVC(kernel='linear', class_weight='balanced')
    clf_parameters = {'clf__C':(0.5,0.9,1,1.5,2,3,5),
                      'clf__probability' : [True]
                     }

    print('No terms \t'+str(no_of_selected_terms))
    try:                                        # To use selected terms of the vocabulary
        pipeline = Pipeline([
                ('vect', CountVectorizer(token_pattern=r'\b\w+\b')),
                ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),
                ('feature_selection', SelectKBest(chi2, k=no_of_selected_terms)),    
                #('feature_selection', SelectKBest(mutual_info_classif, k=no_of_selected_terms)),                     
                ('clf', clf),])
    except:                                  # If the input is wrong
            print('Wrong Input. Enter the number of terms correctly. \n')
            sys.exit()
            
# Fix the values of the parameters using Grid Search and cross validation on the training samples
    # feature_parameters = {
    #  'vect__min_df': (2,3),
    #  'vect__ngram_range': (1,1),  # Unigrams, Bigrams or Trigrams
    #  }
    # parameters={**feature_parameters,**clf_parameters}

    parameters={**clf_parameters}
    grid = GridSearchCV(pipeline,parameters,scoring='f1_macro',cv=10)
    grid.fit(trn_data,trn_cat)
    clf= grid.best_estimator_
    return clf

clf = tfidf_classification_model(X_train,Y_train)
print(clf)
predicted = clf.predict(X_test)
print(classification_report(Y_test, predicted))

x_test_proba = clf.predict_proba(X_test)
# print("X test probabilities: ",x_test_proba)

x_train_proba = clf.predict_proba(X_train)
# print("X train probabilties: ",x_train_proba)

#saving the probabilities
np.save('test prob.npy',x_test_proba)
np.save('train prob.npy',x_train_proba)
