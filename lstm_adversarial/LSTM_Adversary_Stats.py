import numpy as np
from os import listdir
from os.path import isfile, join

advFiles = ['./adversary_info/' + f for f in listdir('./adversary_info') if isfile(join('./adversary_info/', f))]
#print advFiles

total = 0
avg_num_words_changed = 0
originally_neg = 0
originally_neg_classified_neg = 0
originally_neg_changed_pos = 0
originally_pos = 0
originally_pos_classified_pos = 0
originally_pos_changed_neg = 0

# not the embeddings matrix, but the list
wordsList = np.load('wordsList-lexic-sorted.npy').tolist()

for fname in advFiles:
    with open(fname, "r") as f:
        #print fname
        total += 1
        #print fname.split('/')[-1].split('.')[0]
        example_num = int(fname.split('/')[-1].split('.')[0])
        original_classification = int(f.readline().split(' ')[-1].replace('\n', ''))
        new_classification = int(f.readline().split(' ')[-1].replace('\n', ''))
        num_words_changed = int(f.readline().split(' ')[-1].replace('\n', ''))
        f.readline() # "WordPositions"
        f.readline() # WordPositions information
        f.readline() # "OldWordVectorPositions"
        old_word_vector = ""
        while ']' not in old_word_vector:
            old_word_vector += f.readline()
        #print old_word_vector
        if not '[' in old_word_vector or not ']' in old_word_vector:
            raise RuntimeError
        old_word_vector = old_word_vector.replace('[', '').replace(']', '').replace('\n', '').lstrip()
        while '  ' in old_word_vector:
            old_word_vector = old_word_vector.replace('  ', ' ')
        #print old_word_vector.split(' ')
        old_word_vector = [int(x) for x in old_word_vector.split(' ')]
        #print old_word_vector
        f.readline() # "NewWordVectorPositions"
        new_word_vector = ""
        while ']' not in new_word_vector:
            new_word_vector += f.readline()
        if not '[' in new_word_vector or not ']' in new_word_vector:
            raise RuntimeError
        new_word_vector = new_word_vector.replace('[', '').replace(']', '').replace('\n', '').lstrip()
        while '  ' in new_word_vector:
            new_word_vector = new_word_vector.replace('  ', ' ')
        new_word_vector = [int(x) for x in new_word_vector.split(' ')]
        #print new_word_vector
        old_review = ""
        for wd in old_word_vector:
            old_review += wordsList[wd] + " "
        new_review = ""
        for wd in new_word_vector:
            new_review += wordsList[wd] + " "
        if num_words_changed <= 3:
            print old_review
            print new_review
            print ""
            print ""
        if (example_num < 12500):
            originally_pos += 1
            if (original_classification == 0):
                originally_pos_classified_pos += 1
                if (new_classification == 1):
                    originally_pos_changed_neg += 1
                    avg_num_words_changed += num_words_changed
                    with open("../adversary_JSMA/pos/" + fname.split('/')[-1], "w") as out:
                        out.write(new_review.encode("UTF-8"))
        else:
            originally_neg += 1
            if (original_classification == 1):
                originally_neg_classified_neg += 1
                if (new_classification == 0):
                    originally_neg_changed_pos += 1
                    avg_num_words_changed += num_words_changed
                    with open("../adversary_JSMA/neg/" + fname.split('/')[-1], "w") as out:
                        out.write(new_review.encode("UTF-8"))
avg_num_words_changed /= float(total)

print "Total # examples: " + str(total)
print "# originally positive files attempted: " + str(originally_pos)
print "\t# originally positive files classified positive: " + str(originally_pos_classified_pos)
print "\t# originally positive files changed to negative: " + str(originally_pos_changed_neg)
print "# originally negative files attempted: " + str(originally_neg)
print "\t# originally negative files classified negative: " + str(originally_neg_classified_neg)
print "\t# originally negative files changed to positive: " + str(originally_neg_changed_pos)
print "Average # words changed: " + str(avg_num_words_changed)
print "Adversary's success: " + str((originally_pos_changed_neg + originally_neg_changed_pos)*100/float(
    originally_pos_classified_pos + originally_neg_classified_neg)) + "%"

'''
import os

#Change Directory for Reading Files
directory_in_str=str('/afs/cs.stanford.edu/u/aditir/scr/lstm/CS229-Project/lstm_adversarial')

directory = os.path.fsencode(directory_in_str)

for file in os.path.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".txt"): 
        print('.txt file found=>'+filename)

        f=open(filename,"r")

        if f.mode == 'r':
        	contents=f.read()
        	fl=f.readlines()
        	for x in fl:
        		print(x)
'''
'''
from pathlib import Path

#Change Directory for Reading Files
directory_in_str=str('/afs/cs.stanford.edu/u/aditir/scr/lstm/CS229-Project/lstm_adversarial/')

pathlist = Path(directory_in_str).glob('**/*.txt')
for path in pathlist:
    path_in_str = str(path)

 	f=open(path_in_str,"r")

 	if f.mode == 'r':
 		contents=f.read()
    	fl=f.readlines()
    	for x in fl:
    		print(x)
'''
