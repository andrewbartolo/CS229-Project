import numpy as np
from os import listdir
from os.path import isfile, join

advFiles = ['./adversary_info/' + f for f in listdir('./adversary_info') if isfile(join('./adversary_info/', f))]
#print advFiles

originally_neg = 0
originally_pos = 0

for fname in advFiles:
    with open(fname, "r") as f:
        print fname
        example_num = int(fname.split('/')[-1].split('.')[0])
        if (example_num < 12500):
            originally_neg += 1
        else:
            originally_pos += 1
        print f.readline()

print "# originally positive files attempted: " + str(originally_neg)
print "# originally negative files attempted: " + str(originally_pos)

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
