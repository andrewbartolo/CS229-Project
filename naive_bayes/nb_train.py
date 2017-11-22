import numpy as np
from os import listdir
from os.path import isfile, join
import re

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

# read in predefined word list (from LSTM demo)
wordsList = np.load('../training_data/wordsList.npy')
print ('Loaded the word list!')
#print wordsList
wordsList = wordsList.tolist()
wordsList.sort()

# get statistics on training data
positiveFiles = ['../stanford_train/pos/' + f for f in listdir('../stanford_train/pos') if isfile(join('../stanford_train/pos/', f))]
negativeFiles = ['../stanford_train/neg/' + f for f in listdir('../stanford_train/neg') if isfile(join('../stanford_train/neg/', f))]
numWords = []
for pf in positiveFiles:
    with open(pf, "r") as f:
        line = f.readline()
        counter = len(line.split())
        numWords.append(counter)
print ('Positive files finished')
for nf in negativeFiles:
    with open(nf, "r") as f:
        line = f.readline()
        counter = len(line.split())
        numWords.append(counter)
print ('Negative files finished')
numFiles = len(numWords)
print 'The total number of files is: ' + str(numFiles)
print 'The total number of words in the files is: ' + str(sum(numWords))
print 'The average number of words in the files is: ' + str(sum(numWords)/len(numWords))

# train on training files
positive_count = []
negative_count = []
unknown_words = 0
npositive = 0
npositive_unknown = 0
nnegative = 0
nnegative_unknown = 0
for i in range (0, len(wordsList)):
    positive_count.append(0)
    negative_count.append(0)
for pf in positiveFiles:
    with open(pf, "r") as f:
        line = cleanSentences(f.readline())
        for word in line.split():
            npositive += 1
            if npositive % 100000 == 0:
                print 'Words processed: ' + str(npositive)
                print 'Not found: ' + str(unknown_words)
            index = findIndex(wordsList, 0, len(wordsList)-1, word)
            if not index == -1:
                positive_count[index] += 1
            else:
                npositive_unknown += 1
                unknown_words += 1
print ('Positive files finished')
for nf in negativeFiles:
    with open(nf, "r") as f:
        line = cleanSentences(f.readline())
        for word in line.split():
            nnegative += 1
            if nnegative % 100000 == 0:
                print 'Words processed: ' + str(nnegative)
                print 'Not found: ' + str(unknown_words)
            index = findIndex(wordsList, 0, len(wordsList)-1, word)
            if not index == -1:
                negative_count[index] += 1
            else:
                nnegative_unknown += 1
                unknown_words += 1
print ('Negative files finished')
# Laplace smoothing
for i in range (0, len(wordsList)):
    positive_count[i] = float(positive_count[i] + 1)/float(npositive - npositive_unknown + len(wordsList))
    negative_count[i] = float(negative_count[i] + 1)/float(nnegative - nnegative_unknown + len(wordsList))

# save training data
nb_neg_weights = open("model_data/negative_weights.txt", "w")
nb_pos_weights = open("model_data/positive_weights.txt", "w")
nb_stats       = open("model_data/training_stats.txt", "w")
nb_stats.write(str(len(wordsList)) + '\n')
for i in range (0, len(wordsList)):
    nb_stats.write(wordsList[i] + '\n')
nb_stats.write('Total words: ' + str(npositive + nnegative) + '\n')
nb_stats.write('Number of unknown words: ' + str(unknown_words) + '\n')
nb_stats.write('Number of words in positive reviews: ' + str(npositive) + '\n')
nb_stats.write('Number of words in negative reviews: ' + str(nnegative) + '\n')
for i in range (0, len(wordsList)):
    nb_pos_weights.write(str(positive_count[i]) + '\n')
    nb_neg_weights.write(str(negative_count[i]) + '\n')
nb_stats.close()
nb_neg_weights.close()
nb_pos_weights.close()

print 'Done saving!'
print 'Total words: ' + str(npositive + nnegative)
print 'Number of unknown words: ' + str(unknown_words)
print 'positive sanity check (should equal 1): ' + str(sum(positive_count))
print 'negative sanity check (should equal 1): ' + str(sum(negative_count))
