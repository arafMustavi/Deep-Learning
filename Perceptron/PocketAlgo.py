import random
from operator import add

# In the Train File:The first row contains three integers:
# the no.of features(d),
# the no.of classes(m) and
# the total number of samples(n).
# TODO:
# 1. Dataset Input
# 2. Train

#
#
# traindata = open('trainLinearlySeparable.txt','r')
# line = traindata.readline()
# line = line.split()
# d, m, n = tuple(line)
#
# print(line)
# print(d)
# print(m)
# print(n)
# print(type(d))
# # traindata.close()
# # f = open(“input.txt’)
# # lines = traindata.readlines()
# # print(line)
# # print(lines)
# # print(lines[0])
# # traindata.close()
# #
# #
# # for x in traindata:
# #     print(x)


with open('trainLinearlyNonSeparable.txt', 'r') as data:
    filedata = data.readlines()
header = filedata[0].split()
totalFeature, nClass, nSample = (tuple(header))
totalFeature = int(totalFeature)
nClass = int(nClass)
nSample = int(nSample)

dataset_dict = {}
count = 0
dataset = []
for row in filedata[1:]:
    frame = row.split()
    frame = [float(i) for i in frame]
    frame[-1] = int(frame[-1]) - 1
    frame.insert(-1, 1)
    dataset.append(frame)


def dataset_profile(givenset):
    dict = {}
    for frame in givenset:
        if frame[-1] not in dict.keys():
            dict[frame[-1]] = []
        dict[frame[-1]].append(frame[:-1])
    print(dict)
    for key in dict.keys():
        print(key, len(dict[key]))


# dataset_profile(dataset)
# for x in dataset:
#     print(x)
#     print(x[-1])

# print(dataset)
######################
def dot_product(data, weight):
    sum = 0
    for i in range(totalFeature + 1):
        sum += weight[i] * float(data[i])
    return sum


def inWrongClass(data, weight):
    actual_class = int(data[-1])
    apparent_class = 0
    dot_value = dot_product(weight, data)
    if dot_value >= 0:
        apparent_class = 1
    else:
        apparent_class = 0
    if apparent_class != actual_class:
        return True
    else:
        return False

learning_rate = 0.7

# *************************************************
# **************BASIC PERCEPTRON ******************
# *************************************************
def update_batch_weight(weight, misclassified_dataset):
    for data in misclassified_dataset:
        if (data[-1] == 0):
            constant = -1
        else:
            constant = 1
        # weight = weight + learning_rate*data[:-1]*constant
        adder = [learning_rate * constant * x for x in data[:-1]]
        weight = list(map(add, weight, adder))
        # print(x)
    return weight


# hard weight = [-0.0248,0.5048456,-5.45665,8.25,1.2]
weight = []
weightPocket = []
accuracyPocket = 0

for x in range(totalFeature + 1):
    weight.append(random.randint(0, 100) / 100)
print("Total Feature :" + str(totalFeature) + " Initial Weight:")
print(weight)
# print(dataset[1])

# weight = [ 1,2,3,4,5]
# print(weight)
# print(dot_product(weight,dataset[1]))
print("===============================")
print("*******Training Started********")
print("===============================")
accuracy = 0
total_iteration = 1500
for i in range(total_iteration):
    print("At Stage " + str(i))
    misclassified_data = []
    for data in dataset:
        if inWrongClass(data, weight):
            misclassified_data.append(data)
        else:
            pass

    weight = update_batch_weight(weight, misclassified_data)
    accuracy = (1 - len(misclassified_data) / len(dataset)) * 100
    if (accuracy > accuracyPocket):
        accuracyPocket = accuracy
        weightPocket = weight
    print("Total Incorrect:" + str(len(misclassified_data)) + " Current Accuracy : " + str(accuracy) + " Updated Weight :")
    print(weight)
    print("Accuracy Pocket : " + str(accuracyPocket)+ " Weight Pocket :" )
    print(weightPocket)
    if accuracy == 100:
        break

print("===============================")
print("*******Training Ended********")
print("===============================")

# misclassified , acquired_accuracy = calc_aacuracy(dataset,weight)


with open('testLinearlyNonSeparable.txt', 'r') as data:
    filedata = data.readlines()

testdataset = []
for row in filedata[1:]:
    frame = row.split()
    frame = [float(i) for i in frame]
    frame[-1] = int(frame[-1]) - 1
    frame.insert(-1, 1)
    testdataset.append(frame)

dataset_profile(testdataset)
misclassified_data = []
for data in testdataset:
    if inWrongClass(data, weight):
        print("Misclassified The Test Data")
        print(data)
        misclassified_data.append(data)
    else:
        pass
        # print("Passed Test")
        # print(data)
print(dataset_profile(misclassified_data))
print(misclassified_data)
recognition = (1 - len(misclassified_data) / len(testdataset)) * 100
print("Test Complete. Total Incorrect:" + str(len(misclassified_data)) + " Recognition Rate : " + str(recognition) )


#*******************************************************************************************************

# # *******************************************************
# # ********Reward and Punishment PERCEPTRON *************
# # ******************************************************
# def update_instant_weight(weight, data):
#     if (data[-1] == 0):
#         constant = -1
#     else:
#         constant = 1
#     # weight = weight + learning_rate*data[:-1]*constant
#     adder = [learning_rate * constant * x for x in data[:-1]]
#     weight = list(map(add, weight, adder))
#     return weight
#
#
# # hard weight = [-0.0248,0.5048456,-5.45665,8.25,1.2]
# rnpweight = []
# for x in range(totalFeature + 1):
#     rnpweight.append(random.randint(0, 100) / 100)
# print("Total Feature :" + str(totalFeature) + "Initial Weight:")
# print(rnpweight)
# # print(dataset[1])
#
# # weight = [ 1,2,3,4,5]
# # print(weight)
# # print(dot_product(weight,dataset[1]))
# print("====================================================")
# print("*******Reward And Punishment Training Started********")
# print("====================================================")
# accuracy = 0
# total_iteration = 1000
# for i in range(total_iteration):
#     print("At Stage " + str(i))
#     misclassified_data_count = 0
#     for data in dataset:
#         if inWrongClass(data, rnpweight):
#             rnpweight = update_instant_weight(rnpweight, data)
#             misclassified_data_count += 1
#         else:
#             pass
#
#     accuracy = (1 - misclassified_data_count / len(dataset)) * 100
#     print("Total Incorrect:" + str(misclassified_data_count) + " Accuracy : " + str(accuracy) + " Updated Weight :")
#     print(rnpweight)
#     if accuracy == 100:
#         break
#
# print("===================================================")
# print("*******Reward And Punishment Training Ended********")
# print("===================================================")
#
# # misclassified , acquired_accuracy = calc_aacuracy(dataset,weight)
#
#
# with open('testLinearlySeparable.txt', 'r') as data:
#     filedata = data.readlines()
#
# testdataset = []
# for row in filedata[1:]:
#     frame = row.split()
#     frame = [float(i) for i in frame]
#     frame[-1] = int(frame[-1]) - 1
#     frame.insert(-1, 1)
#     testdataset.append(frame)
#
# dataset_profile(testdataset)
# misclassified_data = []
# for data in testdataset:
#     if inWrongClass(data, rnpweight):
#         print("Misclassified The Test Data")
#         print(data)
#         misclassified_data.append(data)
#     else:
#         pass
#         # print("Passed Test")
#         # print(data)
# print(dataset_profile(misclassified_data))
# print(misclassified_data)
# recognition = (1 - len(misclassified_data) / len(testdataset)) * 100
# print("Reward and Punishment Test Complete. Total Incorrect:" + str(len(misclassified_data)) + " In Reward and Punishment Recognition Rate : " + str(recognition) )
#
#
#
#
#
#
