import numpy as np
from os import listdir
from os.path import isfile, join
import re
import sys

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

adversarial_input = False
if ('--adversarial' in sys.argv):
    adversarial_input = True

# read in trained model
nb_neg_weights = open("model_data/negative_weights.txt", "r")
nb_pos_weights = open("model_data/positive_weights.txt", "r")
nb_stats       = open("model_data/training_stats.txt", "r")
wordsList = []
positive_weights = []
negative_weights = []
for i in range (0, int(nb_stats.readline())):
    wordsList.append(nb_stats.readline().replace('\n', ''))
    positive_weights.append(float(nb_pos_weights.readline()))
    negative_weights.append(float(nb_neg_weights.readline()))
nb_stats.close()
nb_neg_weights.close()
nb_pos_weights.close()

# test on test data (not adversarial or adversarial)
if not adversarial_input:
    positiveFiles = ['../training_data/positiveReviews_test/' + f for f in listdir('../training_data/positiveReviews_test') if isfile(join('../training_data/positiveReviews_test/', f))]
    negativeFiles = ['../training_data/negativeReviews_test/' + f for f in listdir('../training_data/negativeReviews_test') if isfile(join('../training_data/negativeReviews_test/', f))]
else:
    positiveFiles = ['../adversarial_data/positiveReviews_test/' + f for f in listdir('../adversarial_data/positiveReviews_test') if isfile(join('../adversarial_data/positiveReviews_test/', f))]
    negativeFiles = ['../adversarial_data/negativeReviews_test/' + f for f in listdir('../adversarial_data/negativeReviews_test') if isfile(join('../adversarial_data/negativeReviews_test/', f))]

total_files = 0
incorrect   = 0
for pf in positiveFiles:
    total_files += 1
    positive_total = 0
    negative_total = 0
    with open(pf, "r") as f:
        line = cleanSentences(f.readline())
        for word in line.split():
            index = findIndex(wordsList, 0, len(wordsList)-1, word)
            if not index == -1:
                positive_total += np.log(positive_weights[index])
                negative_total += np.log(negative_weights[index])
        if positive_total <= negative_total:
            incorrect += 1
print ('Positive files finished')
for nf in negativeFiles:
    total_files += 1
    positive_total = 0
    negative_total = 0
    with open(nf, "r") as f:
        line = cleanSentences(f.readline())
        for word in line.split():
            index = findIndex(wordsList, 0, len(wordsList)-1, word)
            if not index == -1:
                positive_total += np.log(positive_weights[index])
                negative_total += np.log(negative_weights[index])
        if negative_total <= positive_total:
            incorrect += 1
print ('Negative files finished')

# print statistics
print 'Total files: ' + str(total_files)
print 'Incorrect classifications: ' + str(incorrect)
print 'Correctness: ' + str(100 - ((float(incorrect)/float(total_files))*100)) + '%'
