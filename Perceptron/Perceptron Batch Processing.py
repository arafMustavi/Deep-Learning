import random
from operator import add

# *****************************************************************************
# *************************Data Reading from File******************************
# *****************************************************************************

def dataset_profile(givenset):
    dict = {}
    for frame in givenset:
        if frame[-1] not in dict.keys():
            dict[frame[-1]] = []
        dict[frame[-1]].append(frame[:-1])
    print(dict)
    for key in dict.keys():
        print(key, len(dict[key]))

with open('trainLinearlySeparable.txt', 'r') as data:
    filedata = data.readlines()
header = filedata[0].split()
totalFeature, nClass, nSample = (tuple(header))
totalFeature = int(totalFeature)
nClass = int(nClass)
nSample = int(nSample)

dataset = []
for row in filedata[1:]:
    frame = row.split()
    frame = [float(i) for i in frame]
    frame[-1] = int(frame[-1]) - 1
    frame.insert(-1, 1)
    dataset.append(frame)

# *****************************************************************************
# *************************Essential Functions ********************************
# *****************************************************************************

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

# *************************************************
# *****BASIC PERCEPTRON | Batch Processing ********
# *************************************************


# hardcoded weight = [-0.0248,0.5048456,-5.45665,8.25,1.2]
weight = []
for x in range(totalFeature + 1):
    weight.append(random.randint(0, 100) / 100)
print("Total Feature :" + str(totalFeature) + "Initial Weight:")
print(weight)

print("=====================================")
print("*******Batch Training Started********")
print("=====================================")
accuracy = 0
total_iteration = 1000
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
    # print("Total Incorrect:" + str(len(misclassified_data)) + " Accuracy : " + str(accuracy) + " Updated Weight :")
    # print(weight)
    if accuracy == 100:
        break

print("============================================")
print("*************Batch Train Ended**************")
print("*******Test on Batch Trained Started********")
print("============================================")

with open('testLinearlySeparable.txt', 'r') as data:
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
        # print("Misclassified The Test Data")
        # print(data)
        misclassified_data.append(data)
    else:
        pass
        # print("Passed Test")
        # print(data)
# print(dataset_profile(misclassified_data))
# print(misclassified_data)
recognition = (1 - len(misclassified_data) / len(testdataset)) * 100
print("Test Complete. Total Incorrect:" + str(len(misclassified_data)) + " Recognition Rate : " + str(recognition) )
