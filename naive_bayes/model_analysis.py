import numpy as np

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
positive_top = [(0,0), (0,0), (0,0), (0,0), (0,0)]
negative_top = [(0,0), (0,0), (0,0), (0,0), (0,0)]
for i in range (0, len(wordsList)):
    positive_slant.append(float(positive_weights[i])/float(negative_weights[i]))
    if positive_slant[i] > positive_top[0][0]:
        positive_top[4] = positive_top[3]
        positive_top[3] = positive_top[2]
        positive_top[2] = positive_top[1]
        positive_top[1] = positive_top[0]
        positive_top[0] = (positive_slant[i], i)
    elif positive_slant[i] > positive_top[1][0]:
        positive_top[4] = positive_top[3]
        positive_top[3] = positive_top[2]
        positive_top[2] = positive_top[1]
        positive_top[1] = (positive_slant[i], i)
    elif positive_slant[i] > positive_top[2][0]:
        positive_top[4] = positive_top[3]
        positive_top[3] = positive_top[2]
        positive_top[2] = (positive_slant[i], i)
    elif positive_slant[i] > positive_top[3][0]:
        positive_top[4] = positive_top[3]
        positive_top[3] = (positive_slant[i], i)
    elif positive_slant[i] > positive_top[4][0]:
        positive_top[4] = (positive_slant[i], i)
    negative_slant.append(float(negative_weights[i])/float(positive_weights[i]))
    if negative_slant[i] > negative_top[0][0]:
        negative_top[4] = negative_top[3]
        negative_top[3] = negative_top[2]
        negative_top[2] = negative_top[1]
        negative_top[1] = negative_top[0]
        negative_top[0] = (negative_slant[i], i)
    elif negative_slant[i] > negative_top[1][0]:
        negative_top[4] = negative_top[3]
        negative_top[3] = negative_top[2]
        negative_top[2] = negative_top[1]
        negative_top[1] = (negative_slant[i], i)
    elif negative_slant[i] > negative_top[2][0]:
        negative_top[4] = negative_top[3]
        negative_top[3] = negative_top[2]
        negative_top[2] = (negative_slant[i], i)
    elif negative_slant[i] > negative_top[3][0]:
        negative_top[4] = negative_top[3]
        negative_top[3] = (negative_slant[i], i)
    elif negative_slant[i] > negative_top[4][0]:
        negative_top[4] = (negative_slant[i], i)
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
