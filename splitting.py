import os
import random
import numpy as np


root = os.getcwd()
input_file = open(root + "\\data\\data_file.txt")
lines = input_file.readlines()
index_list = []
for i in range(100000000):
    r = random.randint(0, len(lines)/2)
    if r not in index_list:
        index_list.append(r)
    if len(index_list) > 0.2*len(lines)/2:
        break
training = []
testing = []
for i in range(len(lines)):
    if (i % 2 == 0 and int(i/2) in index_list) or (i % 2 != 0 and (i-1)/2 in index_list):
        testing.append(lines[i].strip())
    else:
        training.append(lines[i].strip())
input_file.close()
np.save(root + "\\data\\training", np.array(training))
np.save(root + "\\data\\testing", np.array(testing))
np.save(root + "\\data\\testing", np.array(testing))
