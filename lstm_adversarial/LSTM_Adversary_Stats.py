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