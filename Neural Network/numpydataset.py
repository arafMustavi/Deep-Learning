import random
from operator import add
import numpy as np
totalFeature = 0
nClass = 0
nSample = 0

#
# TODO:
# 1. Dataset Input
# Dot Product

# *****************************************************************************
# **************Data Reading from File + Essential Functions ******************
# *****************************************************************************

def dataset_profile(givenset):
    print("The dataset has total Rows: " + str(len(givenset)))
    dict = {}

    for frame in givenset:
        if frame[-1] not in dict.keys():
            dict[frame[-1]] = []
        dict[frame[-1]].append(frame[:-1])
    # print(dict)
    print("Class TotalEntry")
    for key in dict.keys():
        print(key, len(dict[key]))

    totalFeature = len(givenset[0]) -1
    nClass = len(dict)
    nSample = len(givenset)

    return


# totalFeature = int(input("Total Feature:"))
# nClass = int(input("Total Class:"))
# nSample = int(input("Total Sample"))



with open('testNN.txt', 'r') as data:
    filedata = data.readlines()
# header = filedata[0].split()
# totalFeature, nClass, nSample = (tuple(header))


dataset = []
for row in filedata[0:]:
    frame = row.split()
    frame = [float(i) for i in frame]
    frame[-1] = int(frame[-1]) - 1
    frame.insert(-1, 1)
    dataset.append(frame)

dataset_profile(dataset)
