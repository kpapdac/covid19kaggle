import json
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet          
from nltk.stem import WordNetLemmatizer, PorterStemmer
from gensim import corpora, models

#data location
path = r'C:\Users\Katerina\Documents\covid19_kaggle\data\CORD-19-research-challenge'

metadata = pd.read_csv(path+'\metadata.csv')

titles = metadata['title'].dropna()
titles = titles.reset_index(drop=True)#.str.split(' ')

# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(titles)
# print(vectorizer.get_feature_names())

#stem = nltk.PorterStemmer()

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.portstem = PorterStemmer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) 
                    if t not in stopwords.words('english') and t.isalpha()]
             #self.portstem.stem(t) for t in word_tokenize(doc) 
                            
class Pos:
    def __init__(self, features):
        self.features = features
    def __call__(self):
        #ironically coronavirus is not defined in wordnet
        return [(i[0], i[0].pos()) for i in [wordnet.synsets(f[0]) for f in 
                self.features if f[0] != 'coronavirus']]   

vectorizer = CountVectorizer(tokenizer=LemmaTokenizer())
X = vectorizer.fit_transform(titles)
#print(vectorizer.get_feature_names())

count_tf = np.sum(X, axis=0)
print('Most common title word is {}'.format(vectorizer.get_feature_names()[np.argmax(count_tf)]))

#most common words
common_words = sorted([(vectorizer.get_feature_names()[i[1]], count_tf[0,i[1]]) for i in 
                    np.argwhere(count_tf>1000)], key=lambda x:x[1], reverse=True)

#synsets
# wordnet.synsets('dog')
# wordnet.synset('dog.n.01')
# wordnet.synset('dog.n.01').definition()
pos_clas = Pos(common_words)
pos_clas()

#articles about transmission
feature_index = vectorizer.get_feature_names().index('transmission')
articles_tot = count_tf[0, feature_index]
feature_vect = [X[i,feature_index] for i in range(X.shape[0])]
articles_id = [i for i, x in enumerate(feature_vect) if x == 1]   

#looking at a specific article title on transmission
article_index = articles_id[0]
titles[article_index]
terms_arr = [X[article_index,i] for i in range(X.shape[1])]
present_terms = [i  for i,x in enumerate(terms_arr) if x==1]
title_tokens = [wordnet.synsets(vectorizer.get_feature_names()[i])[0].pos() for i in 
                present_terms if len(wordnet.synsets(vectorizer.get_feature_names()[i]))>0]

title_tokens = {}
for art_id in articles_id:
    terms_arr = [X[art_id,i] for i in range(X.shape[1])]
    present_terms = [i  for i,x in enumerate(terms_arr) if x==1]
    title_tokens[art_id] = [wordnet.synsets(vectorizer.get_feature_names()[i])[0] for i in 
                    present_terms if len(wordnet.synsets(vectorizer.get_feature_names()[i]))>0 
                    and wordnet.synsets(vectorizer.get_feature_names()[i])[0].pos()=='n']

#get lemma names per topic
nouns_doc = [[t.lemma_names()[0] for t in doc] for doc in list(title_tokens.values()) for t in doc]


# list_of_list_of_tokens = [["a","b","c"], ["d","e","f"]]
# ["a","b","c"] are the tokens of document 1, ["d","e","f"] are the tokens of document 2...
dictionary_LDA = corpora.Dictionary(nouns_doc)
dictionary_LDA.filter_extremes(no_below=3)
corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in list_of_list_of_tokens]

num_topics = 20
%time lda_model = models.LdaModel(corpus, num_topics=num_topics, \
                                  id2word=dictionary_LDA, \
                                  passes=4, alpha=[0.01]*num_topics, \
                                  eta=[0.01]*len(dictionary_LDA.keys()))






#get similarity between feature names




# pdfs = os.listdir(path+r'\biorxiv_medrxiv\biorxiv_medrxiv\pdf_json')
# #print(pdfs[0])
# titles = {}
# for doc in pdfs:
#     with open(path + r'\biorxiv_medrxiv\biorxiv_medrxiv\pdf_json\\' + doc, 'r') as data:
#         pdf_json = json.load(data)
#         data.close()
#     titles[pdf_json['paper_id']] = pdf_json['metadata']['title']


