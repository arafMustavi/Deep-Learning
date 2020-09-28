import random
from operator import add
# *******************Reward and Punishment PERCEPTRON ************************

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

def update_instant_weight(weight, data):
    if (data[-1] == 0):
        constant = -1
    else:
        constant = 1
    # weight = weight + learning_rate*data[:-1]*constant
    adder = [learning_rate * constant * x for x in data[:-1]]
    weight = list(map(add, weight, adder))
    return weight


# hard weight = [-0.0248,0.5048456,-5.45665,8.25,1.2]
rnpweight = []
for x in range(totalFeature + 1):
    rnpweight.append(random.randint(0, 100) / 100)
print("Total Feature :" + str(totalFeature) + "Initial Weight:")
print(rnpweight)
# print(dataset[1])
# weight = [ 1,2,3,4,5]
# print(weight)
# print(dot_product(weight,dataset[1]))

print("====================================================")
print("*******Reward And Punishment Training Started********")
print("====================================================")
accuracy = 0
total_iteration = 1000
for i in range(total_iteration):
    print("At Stage " + str(i))
    misclassified_data_count = 0
    for data in dataset:
        if inWrongClass(data, rnpweight):
            rnpweight = update_instant_weight(rnpweight, data)
            misclassified_data_count += 1
        else:
            pass

    accuracy = (1 - misclassified_data_count / len(dataset)) * 100
    print("Total Incorrect:" + str(misclassified_data_count) + " Accuracy : " + str(accuracy) + " Updated Weight :")
    print(rnpweight)
    if accuracy == 100:
        break

print("===================================================")
print("*******Reward And Punishment Training Ended********")
print("**************Test on R&P Started*************")
print("===================================================")

# misclassified , acquired_accuracy = calc_aacuracy(dataset,weight)


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
    if inWrongClass(data, rnpweight):
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
print("Reward and Punishment Test Complete. Total Incorrect:" + str(len(misclassified_data)) + " In Reward and Punishment Recognition Rate : " + str(recognition) )
