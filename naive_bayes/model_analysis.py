import numpy as np
import sys

# read in predefined word list (from LSTM demo)
wordsList = np.load('../training_data/wordsList.npy')
print ('Loaded the word list!')
#print wordsList
wordsList = wordsList.tolist()
wordsList.sort()
#read in trained model
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
print 'Loaded in trained model!'

# process data and find maximum-ratio pos/neg or neg/pos keywords
positive_slant = []
negative_slant = []
positive_top = []
negative_top = []
top_numbers = 10
if '--number' in sys.argv:
    top_numbers = int(sys.argv[2])
for i in range (0, top_numbers):
    positive_top.append((0, 0))
    negative_top.append((0, 0))
for i in range (0, len(wordsList)):
    positive_slant.append(float(positive_weights[i])/float(negative_weights[i]))
    for j in range (0, top_numbers):
        if positive_slant[i] > positive_top[j][0]:
            for k in reversed(range (j + 1, top_numbers)):
                positive_top[k] = positive_top[k - 1]
            positive_top[j] = (positive_slant[i], i)
            break
    negative_slant.append(float(negative_weights[i])/float(positive_weights[i]))
    for j in range (0, top_numbers):
        if negative_slant[i] > negative_top[j][0]:
            for k in reversed(range (j + 1, top_numbers)):
                negative_top[k] = negative_top[k - 1]
            negative_top[j] = (negative_slant[i], i)
            break
print 'Positive Top'
#print positive_top
for i in range (0, len(positive_top)):
    print wordsList[positive_top[i][1]]
#print str(positive_weights[wordsList.index('edie')]/negative_weights[wordsList.index('edie')])
print ''
print 'Negative Top'
#print negative_top
for i in range (0, len(negative_top)):
    print wordsList[negative_top[i][1]]
#print str(negative_weights[wordsList.index('boll')]/positive_weights[wordsList.index('boll')])
