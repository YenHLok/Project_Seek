# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 14:55:18 2019

@author: YeNz
"""

#####################################################################################################
## Set-up

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import json
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None) # to ensure console display all columns

import os
import sys
sys.path.append(os.path.abspath('..'))

import word2vec_functions as wvf


import pickle
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

df = load_obj('Q2_Data')  




#####################################################################################################
## Get Word Embedding

# Get all word counts
all_word_counts = {}
#i=0
for i in range(len(df)):
    if i%100==0:
        print(f'{i} in {len(df)-1}')
    row = df.loc[i]
    s = (row.Job_Responsibilities_Normalized+' '+row.Job_Description_Normalized).split()
    if len(s) > 1:
        for word in s:
            if word not in all_word_counts:
                all_word_counts[word] = 0
            all_word_counts[word] += 1


# Keep top 10000 words
vocab_size = 10000
all_word_counts = sorted(all_word_counts.items(), key=lambda x: x[1], reverse=True)
top_words = [w for w, count in all_word_counts[:min(vocab_size, len(all_word_counts))-1]] + ['<UNK>']
word2idx = {w:i for i, w in enumerate(top_words)}
unk = word2idx['<UNK>']


# Get sentences in index form to train word embedding model
sentences = []
for i in range(len(df)):
    if i%100==0:
        print(f'{i} in {len(df)-1}')
    row = df.loc[i]
    s = (row.Job_Responsibilities_Normalized+' '+row.Job_Description_Normalized).split()
    if len(s) > 1:
        sent = [word2idx[w] if w in word2idx else unk for w in s]
        sentences.append(sent)


  


# Configurations for word embedding training
window_size = 5
learning_rate = 0.025
final_learning_rate = 0.0001
num_negatives = 5 # number of negative samples to draw per input word
epochs = 20
D = 50 # word embedding size

# learning rate decay
learning_rate_delta = (learning_rate - final_learning_rate) / epochs

# params
W = np.random.randn(vocab_size, D) # input-to-hidden
V = np.random.randn(D, vocab_size) # hidden-to-output

# distribution for drawing negative samples
p_neg = wvf.get_negative_sampling_distribution(sentences, vocab_size)

# save the costs to plot them per iteration
costs = []

# number of total words in corpus
total_words = sum(len(sentence) for sentence in sentences)
print("total number of words in corpus:", total_words)

# for subsampling each sentence
threshold = 1e-5
p_drop = 1 - np.sqrt(threshold / p_neg)


# train the model
for epoch in range(epochs):
    # randomly order sentences so we don't always see sentences in the same order
    np.random.shuffle(sentences)

    # accumulate the cost
    cost = 0
    counter = 0
    t0 = datetime.now()
    for sentence in sentences:
        # keep only certain words based on p_neg
        sentence = [w for w in sentence if np.random.random() < (1 - p_drop[w])]
        if len(sentence) < 2: continue
    
        # randomly order words so we don't always see samples in the same order
        randomly_ordered_positions = np.random.choice(
            len(sentence),
            size=len(sentence),#np.random.randint(1, len(sentence) + 1),
            replace=False,
        )


        for pos in randomly_ordered_positions:
            # the middle word
            word = sentence[pos]

            # get the positive context words/negative samples
            context_words = wvf.get_context(pos, sentence, window_size)
            neg_word = np.random.choice(vocab_size, p=p_neg)
            targets = np.array(context_words)

            # do one iteration of stochastic gradient descent
            c = wvf.sgd(word, targets, 1, learning_rate, W, V)
            cost += c
            c = wvf.sgd(neg_word, targets, 0, learning_rate, W, V)
            cost += c

        counter += 1
        if counter % 100 == 0:
            sys.stdout.write("processed %s / %s\r" % (counter, len(sentences)))
            sys.stdout.flush()
            # break


    # print stuff so we don't stare at a blank screen
    dt = datetime.now() - t0
    print("epoch complete:", epoch, "cost:", cost, "dt:", dt)

    # save the cost
    costs.append(cost)

    # update the learning rate
    learning_rate -= learning_rate_delta


# plot the cost per iteration
plt.plot(costs)
plt.show()


with open('word2idx.json', 'w') as f:
    json.dump(word2idx, f)

np.savez('weights.npz', W, V)





#####################################################################################################
## Clustering Documents


from sklearn.manifold import TSNE

with open('word2idx.json') as f:
    word2idx = json.load(f)
unk = word2idx['<UNK>']    
npz = np.load('weights.npz')
W = npz['arr_0']
V = npz['arr_1']
We = (W + V.T) / 2


df = df.loc[(df.Job_Responsibilities_Normalized!='')|(df.Job_Description_Normalized!='')]
df.reset_index(drop=True,inplace=True)
document_vector = np.zeros((len(df),W.shape[1]))

#i=0
for i in range(len(df)):
    if i%100==0:
        print(f'{i} in {len(df)-1}')
    row = df.loc[i]
    s = (row.Job_Responsibilities_Normalized+' '+row.Job_Description_Normalized).split()
    sent = [word2idx[w] if w in word2idx else unk for w in s]
    document_vector[i] = We[sent].mean(axis=0)
    
   
    
    
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster     
dist_array = pdist(document_vector ,'cosine')

# calculate hierarchy
tree = linkage(dist_array, 'ward')
plt.title("Ward")
dendrogram(tree)
plt.show()



# Visualizing cluster using t-sne
tsne = TSNE()
Z = tsne.fit_transform(document_vector) 

classes = fcluster(tree, 9, criterion='distance') # 9 seems to be a good cutoff point
#classes = fcluster(tree, 19, criterion='distance') # 19 seems to be a good cutoff point

plt.scatter(Z[:,0], Z[:,1],c=classes)
plt.show()


# Order classes based on their size
df['classes'] = classes
old_classes = list(df.groupby('classes')['classes'].count().sort_values(ascending=False).index)
class_map = {}
for i in range(len(set(classes))):
    class_map.update({old_classes[i]: i+1})
df['classes'] = df['classes'].apply(lambda x: class_map[x])

save_obj(df, 'Q3_Data')



#####################################################################################################
## Intepreting the clusters

df = load_obj('Q3_Data')  

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

def normalizer_list(s):
    s = re.sub('[-:;,#\(\)]',' ',s) # remove special symbols
    s = re.sub("[^a-zA-Z]", " ",s)  # only letters
    s = re.sub("NA", " ",s)  # remove NA                
    s = re.sub('[ ]{2,}',' ',s).strip()
    tokens = nltk.word_tokenize(s)
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))  
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas

def ngrams(input_list):
    #onegrams = input_list
    bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]
    trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]
    #out = onegrams+bigrams+trigrams
    out = bigrams+trigrams
    return out

import collections
def count_words(input):
    cnt = collections.Counter()
    for row in input:
        for word in row:
            cnt[word] += 1
    return cnt

df['grams'] = df['Job Title'].apply(normalizer_list).apply(ngrams)
print(df[(df.classes == 1)][['grams']].apply(count_words)['grams'].most_common(5))
print(df[(df.classes == 2)][['grams']].apply(count_words)['grams'].most_common(5))
print(df[(df.classes == 3)][['grams']].apply(count_words)['grams'].most_common(5))
print(df[(df.classes == 4)][['grams']].apply(count_words)['grams'].most_common(5))






















