# NLP For International Tourist Arrival Prediction
A novel regression framework is developed in this work to estimate international tourist arrival in 38 Organization for Economic Co-operation and Development (OECD) countries by combining significant socio-economic-environment features and a natural language processing (NLP) developed social media index. The index is self-developed based on TripAdvisor reviews by using classical machine learning techniques. A pre-trained BERT model is fine-tuned with the reviews from different countries, collected from a tourism portal to generate country-wise feedback scores, which are used as an additional feature with the other features for tourism arrival estimation using an adaptive boosting technique. The outcomes demonstrate that the proposed framework performs reasonably well. The findings of this study can be utilized to make decisions that support the growth of sustainable tourism.

## Prerequisites
The following libraries have to be installed one by one before running the code, if they are not already installed. 
[NLTK](https://www.nltk.org/install.html), [NumPy](https://numpy.org/install/), [Python 3.7 or later version](https://www.python.org/downloads/), [Scikit-Learn](https://scikit-learn.org/0.16/install.html), [Torch](https://pypi.org/project/torch/), [Transformers](https://pypi.org/project/transformers/)

## How to run the framework?

Pass the path of the project e.g., `/home/xyz/nlp4tourism/` as a parameter of the main class in `tourist_prediction.py`. Create the following directories inside this path: 1) `training_data`, 2) `test_data`. Therefore keep the individual PDFs of training and test data in the respective directories. The list of keywords to build the training data should be stored as `keywords.txt` in the main project path. Create a directory, called, `output` in the main project path to store the outputs of individual test samples. 

Subsequently, run the following lines to get relevant sentences of geometric errors of radiotherapy for individual test documents. 

```
de=data_extraction('/home/xyz/nlp4tourism/',model='entropy',clf_opt='s',no_of_selected_terms=1500,threshold=0.5)  
de.build_training_data()       
de.tourist_prediction()
```

The following options of `model` are available and the `default` is `entropy`: 

        'bert' for BERT model

        'entropy' for Entropy based term weighting scheme

        'doc2vec' for Doc2Vec based embeddings 

        'tfidf' for TF-IDF based term weighting scheme 

The following options of 'clf_opt' are available and the `default` is `s`: 

        'lr' for Logistic Regression 

        'ls' for Linear SVC

        'n' for Multinomial Naive Bayes

        'r' for Random Forest

        's' for Support Vector Machine 

`model_source` is the path of BERT model from [Hugging Face](https://huggingface.co/models?search=biobert) or from the local drive. The default option is `monologg/biobert_v1.1_pubmed`. `vec_len` is the desired length of the feature vectors developed by the Doc2Vec model. The deafult option of `no_of_selected_terms` is `None`, otherwise desired number of terms should be mentioned. The default option of threshold (i.e., the sentence similarity threshold α) is 0.5. An example code to implement the whole model is uploaded as `testing_data_extraction.py`. 

### Note
The required portion of the code (in `tourist_prediction.py`) to run a given BERT model is commented, as in many standalone machine one may face difficulty in installing BERT. These comments has to be removed in order to run BERT. Otherwise, a separate code (`sentence_classification_bert.py`) is given to run BERT model for the given data on a standalone machine. 

## Contact

For any further query, comment or suggestion, you may reach out to archana19@iiserb.ac.in and welcometanmay@gmail.com

## Citation
```
@article{nlp4tourism,
  title={Modelling International Tourist Arrivals: A NLP Perspective},
  author={},
  journal={},
  volume={},
  number={},
  pages={},
  year={},
  publisher={}
}
```