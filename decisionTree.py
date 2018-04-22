import numpy as np
import math
from csv import reader
from random import randrange
import random

#loading csv file
def load_csv(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    #print(dataset)
    return dataset

#generating range for random data split
def random_dset(datalen):
    training_range = random.sample(range(0,datalen),datalen//2)
    return training_range

#dividing dataset into train and test data
def divide_dset(dset):
    training_range = random_dset(len(dset))
    train_data = []
    test_data = []
    for i in range(len(dset)):
        if i not in training_range:
            test_data.append(dset[i])
        else:
            train_data.append(dset[i])
    return train_data,test_data

#Splitting of node into right and left
def initial_split(index, value, train_data):
    leftdata = list()
    rightdata = list()
    splitdata = []
    for row in train_data:
        if row[index] < value:
            leftdata.append(row)
        else:
            rightdata.append(row)
    splitdata.append(leftdata)
    splitdata.append(rightdata)
    return splitdata

#calculate gini
def gini_index(splits, classes):
    split_samples = float(sum(len(split) for split in splits))
    ginisplit = 0.0
    for split in splits:
        size = float(len(split))
        if size ==0:
            continue
        score = 0.0
        for class_value in classes:
            proportion = [row[-1] for row in split].count(class_value) / size
            score += proportion * proportion
            #print("score::", score)
        ginisplit += (1.0-score)*(size/split_samples)
        #print("ginisplit::", ginisplit)
    return ginisplit

#calculating info gain
def info_gain(node, allclass_values):
    gain = 0
    p = 0
    for class_val in allclass_values:
        class_count = 0
        for row in node:
            if row[-1] == class_val:
                class_count += 1
        if class_count == 0:
            p = 0
        else:
            p += (float(class_count)/float(len(node)))*(math.log(float(class_count)/float(len(node))))
    gain = -(p)  
    return gain

#selecting the best split from all the calculated splits
def best_split(train_data):
   # print("train data :: ", train_data)
    allclass_values = list(set(row[-1] for row in train_data))
    best_index = 999
    best_value = 999
    best_score = 999
    best_group = None
    best_gain = -999
    ginisplit = 999
    p_gain = info_gain(train_data, allclass_values)
    for index in range(len(train_data[0])-1):
        #print(index)
        for row in train_data:
            initialsplit = initial_split(index, row[index], train_data)
            if option == 1:
                ginisplit = gini_index(initialsplit, allclass_values)
                if ginisplit < best_score:
                    best_index = index
                    best_value = row[index]
                    best_score = ginisplit
                    best_group = initialsplit
            elif option == 2:
                l_gain = info_gain(initialsplit[0], allclass_values)
                r_gain = info_gain(initialsplit[1], allclass_values)
                gain = p_gain - (l_gain+r_gain)
                if gain > best_gain:
                    best_index = index
                    best_value = row[index]
                    best_gain = gain
                    best_group = initialsplit                
            else:
                print("Please provide input between 1 and 2.")
                return
    return {'index':best_index, 'value':best_value, 'initialsplit':best_group}

#naming the leaf node according to class name
def terminal_node(split):
    outcomes = [row[-1] for row in split]
    return max(set(outcomes), key= outcomes.count)
 
#recursively splitting data              
def splitting(node):
    left, right= node['initialsplit']
    del(node['initialsplit'])
    #processing the left child
    if len(left) <=0 or len(right) <=0: 
        if len(left) > 0:
            node['left'] = node['right'] =terminal_node(left)
            print("node left here...")
        else:
            node['left'] = node['right'] =terminal_node(right)
        return
    node['left']= best_split(left)
    splitting(node['left'])
    #processing the right child
    node['right']= best_split(right)
    splitting(node['right'])    
    return 

#build root node and rest of the tree
def building_tree(train_data):
    rootnode = best_split(train_data)
    splitting(rootnode)
    return rootnode

#predict the class of test data
def predict_class(node,row):
    if row[node['index']] < node['value']:
        if isinstance (node['left'],dict):
            return predict_class(node['left'],row)
        else:
            return node['left']
    else:
        if isinstance (node['right'],dict):
            return predict_class(node['right'],row)
        else:
            return node['right']
    return

#calculating accuracy      
def accuracy_calc(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

#main function to call other functions and repeating it 5 times
def decision_tree(dset):  
    score_list = []
    for i in range(0,5):
        datalen = len(dset)
        train_data,test_data=divide_dset(dset)
        tree = building_tree(train_data)
        predictions = []
        for row in test_data:
        #prediction = prediction(tree,row)
            p = predict_class(tree,row)
            predictions.append(p)
        actual = [row[-1] for row in test_data]
        accuracy = accuracy_calc(actual, predictions)
        score_list.append(accuracy)
    #print("score :: ", score_list)
    final_accuracy = sum(score_list)/len(score_list)
    print('Accuracy for data : ', '=',final_accuracy)
    return predictions

#fname = input("Please provide a file name you want to try:")
print("Gini Index : 1")
print("Information Gain : 2")
option = input("How do you want to build a decision tree? ")
option = int(option)


datalen = 0
fname = "data_banknote_authentication.csv"
#fname = "Cryotherapy_1.csv"
#fname = "blood-transfusion.csv"

dset = load_csv(fname)
if len(dset) <= 0:
    print("Please provide a valid .csv file")
else:
    decision_tree(dset)
