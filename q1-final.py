#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import nltk
import math
from sklearn.manifold import TSNE


# In[2]:


nltk.download('abc')


# In[3]:


def softmax(x):
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum(axis=0)

def get_weights(shape,low,high):
    w1 = np.random.rand(shape[0],shape[1])*(high-low)
    w2 = np.random.rand(shape[1],shape[0])*(high-low)
    w1 -= low
    w2 -= low
    return [w1,w2]

class Word_2_Vec():
    def __init__(self, window,embedding_dimension, corpus):
        self.dimension = embedding_dimension
        self.window = window
        self.eta = 0.01
        self.epochs = 50
        
        if len(corpus) < 1:
            return None
        my_word_count = {}
        all_words = []
        for row in corpus:
            all_words.extend(row)
            
        unique_elements, counts_elements = np.unique(np.asarray(all_words), return_counts=True)
        number_unique = unique_elements.shape[0]
        
        for index in range(unique_elements.shape[0]):
            my_word_count[unique_elements[index]] = counts_elements[index]
            
        word_counts = my_word_count
        
        self.v_count = number_unique
        self.words_list = unique_elements.tolist()
        self.word_index = {}
        self.index_word = {}
        for i in range(unique_elements.shape[0]):
            self.word_index[unique_elements[i]] = i
            self.index_word[i] = unique_elements[i]

        training_data = []
        for sentence in corpus:
            sent_len = len(sentence)
            complete_context = []
            for word in sentence:
                w = np.zeros(self.v_count).astype(int).tolist()
                w[self.word_index[word]] = 1
                complete_context.append(w)
                  
            for i in range(len(sentence)):
                w_target = complete_context[i]
                w_context = self.get_context(i,complete_context)
                training_data.append([w_target, w_context])
        print(training_data)
        self.training_data = np.array(training_data)
    
    def get_context(self,i,complete_context):
        context = []
        for j in range(i-self.window,i+self.window+1):
            if j==i: continue
            elif j >= len(complete_context): break
            elif j<0 : continue
            context.append(complete_context[j])
        return context
    
    
    def train(self, training_data):
        (self.w1,self.w2) = get_weights((self.v_count, self.dimension),-0.7,1)
        for epoch in range(self.epochs):
            self.loss = 0
            for w_t, w_c in training_data:               
                w1_T = self.w1.T
                w2_T = self.w2.T
                h = np.dot(w1_T,w_t)
                u = np.dot(w2_T, h)
                y_pred = softmax(u)
                edit_dist = np.subtract(y_pred, w_c[0])
                for word_ind in range(1,len(w_c)):
                    row_temp = np.subtract(y_pred, w_c[word_ind])
                    edit_dist += row_temp
                self.backprop(edit_dist, h, w_t)
#             if epoch%100 == 99:
#                 visualize(epoch)
        
    
    

    def word_sim(self, word, top_n):
        word_sim = {}
        for i in range(self.v_count):
            value_1 = np.dot(self.w1[self.word_index[word]], self.w1[i])
            p1 = self.w1[self.word_index[word]] ** 2
            p1 = np.sqrt(p1.sum())
            p2 = self.w1[i] ** 2
            p2 = np.sqrt(p2.sum())
            closeness = value_1/(p1 * p2)

            word_sim[self.index_word[i]] = closeness

        return sorted(word_sim.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)[:top_n]

    
    
    # BACKPROPAGATION
    def backprop(self, e, h, x):
        
        col_h = np.asarray([h]).T
        row_e = np.asarray([e])
        dl_dw2 = np.dot(col_h,row_e) * self.eta
        dot_row = np.asarray([np.dot(self.w2, e)])
        self.w2 -= dl_dw2
        x_col = np.asarray([x]).T
        dl_dw1 = np.dot(x_col, dot_row) * self.eta
        self.w1 -= dl_dw1
        
    
    



# In[4]:


# def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
#     plt.figure(figsize=(16, 9))
#     colors = cm.rainbow(np.linspace(0, 1, len(labels)))
#     for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
#         x = embeddings[:, 0]
#         y = embeddings[:, 1]
#         plt.scatter(x, y, c=color, alpha=a, label=label)
#         for i, word in enumerate(words):
#             plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
#                          textcoords='offset points', ha='right', va='bottom', size=8)
#     plt.title(title)
#     plt.grid(True)
#     if filename:
#         plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
#     plt.show()

    
# def visualize(i):
#     embedding_clusters = []
#     word_clusters = []
#     for row in corpus:
#         for word in row:
# #             print(word)
#             embeddings = []
#             words = []
#             for t in w.word_sim(word, 5):
#                 similar_word=t[0]
#                 words.append(similar_word)
# #                 print("similar_word:",similar_word)
#                 embeddings.append(w.word_vec(similar_word))
#             embedding_clusters.append(embeddings)
#             word_clusters.append(words)
#     embedding_clusters = np.array(embedding_clusters)
#     n, m, k = embedding_clusters.shape
#     tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
#     embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
#     name='visualizations/similar_words'+str(i)+'.png'
#     tsne_plot_similar_words('Similar words', corpus, embeddings_en_2d, word_clusters, 0.7,name)


# In[5]:


corpus = [['the','quick','brown','fox','jumped','over','the','lazy','dog'],["lazy","dog"]]
n = 5
w = Word_2_Vec(2,n,corpus)
# t = w.generate_training_data(corpus)
w.train(w.training_data)

"""sizes
w1 (8, 5)
h (5,)
e (8,)
eta 0.01
dldw1 (8, 5)
done"""


# In[6]:


w.word_sim("the",4)


# In[ ]:




