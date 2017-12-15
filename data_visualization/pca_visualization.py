import numpy as np
from os import listdir
from os.path import isfile, join
import re
import sys
from random import randint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
#from sklearn.decomposition import SparsePCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

num_words_per_message = 250

# define for string cleaning
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

# define for index finding
def findIndex(search_list, begin, end, key):
    mid = int((end - begin + 1)/2) + begin
    if end == begin:
        if search_list[mid] == key:
            return mid
        else:
            return -1
    if end == begin + 1:
        if search_list[begin] == key:
            return begin
        if search_list[end] == key:
            return end
        else:
            return -1
    if search_list[mid] < key:
        return findIndex(search_list, mid, end, key)
    return findIndex(search_list, begin, mid, key)

# test on test data (not adversarial or adversarial)
positiveFiles = ['../stanford_test/pos/' + f for f in listdir('../stanford_test/pos') if isfile(join('../stanford_test/pos/', f))]
negativeFiles = ['../stanford_test/neg/' + f for f in listdir('../stanford_test/neg') if isfile(join('../stanford_test/neg/', f))]
del positiveFiles[:-(len(positiveFiles)/5)]
del negativeFiles[:-(len(positiveFiles)/5)]

reviews = []
for pf in positiveFiles:
    with open(pf, "r") as f:
        #line = cleanSentences(f.readline()).split()
        line = f.readline()
        reviews.append(line)
for nf in negativeFiles:
    with open(nf, "r") as f:
        #line = cleanSentences(f.readline()).split()
        line = f.readline()
        reviews.append(line)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(reviews)
#num_k = 2
#model = KMeans(n_clusters=num_k, init='k-means++', max_iter=1000, n_init=1)
#model.fit(X)
#pca = SparsePCA(n_components=2).fit(X.toArray())
'''
pca = PCA(n_components=2).fit(X.toarray())
data2D = pca.transform(X.toarray())
plt.scatter(data2D[:len(data2D)/2,0], data2D[:len(data2D)/2,1], c='r')
plt.scatter(data2D[len(data2D)/2:,0], data2D[len(data2D)/2:,1], c='k')
#plt.scatter(data2D[:,0], data2D[:,1], c='k')
plt.show()
'''
ndims = 4
pca = PCA(n_components=ndims).fit(X.toarray())
dataND = pca.transform(X.toarray())
for i in range(ndims):
    for j in range(i+1, ndims):
        plt.scatter(dataND[:len(dataND)/2,i], dataND[:len(dataND)/2,j], c='r')
        plt.scatter(dataND[len(dataND)/2:,i], dataND[len(dataND)/2:,j], c='k')
        ##plt.scatter(data2D[:,0], data2D[:,1], c='k')
        plt.ylabel('PC %d' % j)
        plt.xlabel('PC %d' % i)
        plt.show()

#https://stackoverflow.com/questions/28160335/plot-a-document-tfidf-2d-graph/28205420#28205420
#https://stackoverflow.com/questions/27889873/clustering-text-documents-using-scikit-learn-kmeans-in-python
