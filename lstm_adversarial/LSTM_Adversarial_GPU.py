
# coding: utf-8

# In[3]:

import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import re
import sys
from random import randint
import datetime
from bisect import bisect_left
import pdb
import time 

UNKNOWN_WORD_VECTOR_IDX = 399999
nPFiles = 12500
nNFiles = 12500
ckptInterval = 10000
num_pos=5
end_pos=5 #250 default
dict_start=44000
INSERT_ADVERSARIAL = False
# As found using Mark's Naive Bayes analysis
advExsPos = ['edie', 'antwone', 'din', 'gunga', 'yokai']
advExsNeg = ['boll', '410', 'uwe', 'tashan', 'hobgoblins']

def posAdvWord():
    return advExsPos[randint(0,len(advExsPos)-1)]

def negAdvWord():
    return advExsNeg[randint(0,len(advExsNeg)-1)]

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
def cleanSentences(string):
    '''
    Cleans Sentences
    '''
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())
    
def getSentenceMatrix(sentence):
    arr = np.zeros([batchSize, maxSeqLength])
    sentenceMatrix = np.zeros([batchSize, maxSeqLength],dtype='int32')
    cleanSentence = cleanSentences(sentence)
    split = cleanSentence.split()
    for idxCtr, word in enumerate(split):
        if idxCtr>=250:
            break
        try:
            #sentenceMatrix[0, idxCtr] = binarySearchIndex(wordsList, word)
            sentenceMatrix[0, idxCtr] = binarySearchIndex(wordsList, word)
        except ValueError:
            sentenceMatrix[0, idxCtr] = UNKNOWN_WORD_VECTOR_IDX
    return sentenceMatrix

def generateAdversary(data, wordVectors, prediction, jacobian, origClass, sess):
    data_adv=data
    data_adv_wordEmbedded=tf.squeeze(tf.nn.embedding_lookup(wordVectors,data),axis=0)
    wordPos=end_pos-1

    newClass=origClass

    while newClass==origClass:
        curr_word=data_adv_wordEmbedded[wordPos,:]
        curr_word=tf.reshape(curr_word,[1,50])

        pdb.set_trace()
        dist_curr_word=tf.abs(tf.matmul(tf.sign(curr_word-wordVectors[dict_start:,:]),tf.sign(jacobian[wordPos-end_pos].T)))
        dist_matrix = sess.run(dist_curr_word)
        min_dist = np.minimum(dist_matrix)
        min_indices = np.where(dist_matrix == min_dist)
        location = np.random.randint(np.size(min_indices), size = 1)
        
        #data_adv_wordEmbedded[wordPos,:]=wordVectors[tf.argmin(dist_curr_word),:]
        #data_adv_wordEmbedded=tf.reshape(data_adv,[1,50])
        pdb.set_trace()

        # data_adv[0,wordPos]=sess.run(tf.argmin(dist_curr_word))+dict_start
        data_adv[0, wordPos] = min_indices[location] + dict_start
        data_adv=(data_adv).reshape((1,250))

        predictedSentimentNew=sess.run(prediction, feed_dict = {input_data: data_adv})[0]
        if predictedSentimentNew[0]>predictedSentimentNew[1]:
            newClass=0
        else:
            newClass=1

        wordPos=wordPos-1
        if wordPos==end_pos-1-num_pos:
            break
    
    return data_adv, newClass

    #print(y)

###############################################
############### HYPERPARAMETERS ###############
###############################################
numDimensions = 300
maxSeqLength = 250 # truncate reviews longer than this
batchSize = 1
lstmUnits = 64
numClasses = 2
iterations = 100000 #100K
###############################################

def binarySearchIndex(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError

# not the embeddings matrix, but the list
wordsList = np.load('wordsList-lexic-sorted.npy').tolist()
wordVectors = np.load('wordVectors-lexic-sorted.npy')

#pdb.set_trace()

nWordsInDict = len(wordsList)
print("wordsList (%d words) loaded." % nWordsInDict)
print("wordVectors loaded.")


tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

#data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)

data=tf.split(data,250,axis=1)
data = [tf.squeeze(data_elem, axis=1) for data_elem in data]
#pdb.set_trace()
lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
#lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
#value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
state_init=tf.constant(np.zeros([1,64],dtype=np.float32))
#pdb.set_trace()
value, state = tf.nn.static_rnn(lstmCell, data, dtype=tf.float32, initial_state=(state_init,state_init))


# In[ ]:


weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]), name='Variable_1')
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]), name='Variable_2')
#value = tf.transpose(value, [1, 2])

prediction = (tf.matmul(value[-1], weight) + bias)

#correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
#accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

predictionList=tf.split(prediction,2,axis=1)

#pdb.set_trace()
t0=time.time()
jacobian = {str(k):[tf.gradients(pred_comp, data_inp)[0] for data_inp in data[end_pos-num_pos:end_pos]] for k, pred_comp in enumerate(predictionList)}
t1=time.time()

pMatrix = np.load('pIDsMatrix-train.npy')
nMatrix = np.load('nIDsMatrix-train.npy')
print('Loaded pMatrix-train and nMatrix-train (index matrices)')

config=tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess=tf.InteractiveSession(config=config)
#sess = tf.InteractiveSession()
#with tf.Session() as sess:
with sess.as_default():
    saver = tf.train.Saver(tf.trainable_variables())

    # Restore the checkpointed model we trained on,
    # so we can run inference on it now.
    #pdb.set_trace()
    saver.restore(sess, tf.train.latest_checkpoint('models-100k'))
    
    #tf.global_variables_initializer()
    tf.set_random_seed(100)
    predictedSentiment=sess.run(prediction, feed_dict = {input_data: pMatrix[np.newaxis,2000]})[0]
    #import pdb; pdb.set_trace()
    print(pMatrix[np.newaxis,2000])

    if predictedSentiment[0]>predictedSentiment[1]:
        jac = sess.run([jac for jac in jacobian['0']], feed_dict = {input_data: pMatrix[np.newaxis,2000]})
        origClass=0
        print('Original Class= '+str(0))
    else:
        jac = sess.run([jac for jac in jacobian['1']], feed_dict = {input_data: pMatrix[np.newaxis,2000]})
        origClass=1
        print('Original Class= '+str(1))
    
    adv,newClass=generateAdversary(pMatrix[np.newaxis,2000],wordVectors,prediction,jac,origClass,sess)

    print(adv,newClass)
    print('New Class= '+str(np.argmax(sess.run(prediction, feed_dict = {input_data: pMatrix[np.newaxis,2000]})[0])))
    t2=time.time()

    print('time for single instance= '+str(t2-t1))
    print('time for creating jacobian= '+str(t1-t0))
#pdb.set_trace()


# pMatrix = np.load('pIDsMatrix-train.npy')
# nMatrix = np.load('nIDsMatrix-train.npy')
# print('Loaded pMatrix-train and nMatrix-train (index matrices)')



# ##### Begin the accuracy assessment #####

# # TODO go back and vectorize (currently only processes one review per session run)
# # TODO maybe dedup positive and negative inference
# posCorrect = negCorrect = 0
# inputMatrix = np.zeros([batchSize, maxSeqLength],dtype='int32')

# for idx, review in enumerate(pMatrix):
#     inputMatrix[0] = review
#     if INSERT_ADVERSARIAL:
#         # replace the first word with an adversarial word
#         inputMatrix[0][0] = binarySearchIndex(wordsList, negAdvWord())
#     predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
    
#     # classify
#     if predictedSentiment[0] > predictedSentiment[1]:
#         posCorrect = posCorrect + 1
    
#     print("Finished %d pos reviews; accuracy %f" % (idx, float(posCorrect)/(idx+1)))
# print("Finished classifying all positive reviews: %d out of %d correct.", (posCorrect, nPFiles))

# # TODO dedup with above^^^^
# for idx, review in enumerate(nMatrix):
#     inputMatrix[0] = review
#     if INSERT_ADVERSARIAL:
#         # replace the first word with an adversarial word
#         inputMatrix[0][0] = binarySearchIndex(wordsList, posAdvWord())
#     predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
    
#     # classify
#     if predictedSentiment[0] <= predictedSentiment[1]:
#         negCorrect = negCorrect + 1
    
#     print("Finished %d neg reviews; accuracy %f" % (idx, float(negCorrect)/(idx+1)))
# print("Finished classifying all positive reviews: %d out of %d correct.", (negCorrect, nNFiles))

# print("Done.")

# sys.exit(0)

# # Below is how you'd evaluate the sentiment of a single handcrafted sentence.

inputText = "That movie was great."
#inputText = "Simply terrible."
#inputText= "Movie was awesome and great!"
inputMatrix = getSentenceMatrix(inputText)

#

print(inputText)
print(inputMatrix)
#with tf.Session() as sess1:
with sess.as_default():
    predictedSentiment = sess.run(prediction, {input_data: inputMatrix[np.newaxis,0]})[0]
# predictedSentiment[0] represents output score for positive sentiment
# predictedSentiment[1] represents output score for negative sentiment
    if (predictedSentiment[0] > predictedSentiment[1]):
        print("Positive Sentiment")
        jac = sess.run([jac for jac in jacobian['0']], feed_dict = {input_data: inputMatrix[np.newaxis,0]})
        origClass=0
        print('Original Class= '+str(0))
    else:
        print("Negative Sentiment")
        jac = sess.run([jac for jac in jacobian['1']], feed_dict = {input_data: inputMatrix[np.newaxis,0]})
        origClass=1
        print('Original Class= '+str(1))
    
    print("Generating Adversary")
    
    adv,newClass=generateAdversary(inputMatrix[np.newaxis,0],wordVectors,prediction,jac,origClass,sess)

    #pdb.set_trace()

    print(adv)
    print('New Class= ', newClass)
    for i in range(10):
        print(wordsList[adv[0][i]])


# # In[14]:

# import pprint
# variable_list = tf.trainable_variables()


# # In[19]:

# variable_list[2].eval()


# # In[ ]:



