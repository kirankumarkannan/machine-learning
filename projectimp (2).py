#!/usr/bin/env python
# coding: utf-8

# In[244]:
#used jupyter notebook


import numpy as np
import pandas as pd
eps=np.finfo(float).eps
from numpy import log2 as log
dataset ={"HasJob":["yes", "yes", "yes", "no", "no", "yes", "yes", "yes", "no", "no"], "HasInsurance":["yes", "no", "no", "no", "no", "yes", "no", "no", "no", "no"], "Votes":["yes", "yes", "no", "yes", "no", "yes", "yes", "no", "yes", "no"], "Action":["leave-alone", "leave-alone", "force-into", "leave-alone", "force-into", "leave-alone", "leave-alone", "force-into", "leave-alone", "force-into"]}
df = pd.DataFrame(dataset)  #arranges data in form
df                          #prints the data in tabular form
entropy_node = 0  #Initialize Entropy
values = df.Action.unique()  
for value in values:
    fraction = df.Action.value_counts()[value]/len(df.Action)
    entropy_node += -fraction*np.log2(fraction) #calcualtion of entropy

entropy_node #print entropy

attribute = 'HasJob'
target_variables = df.Action.unique()  
variables = df[attribute].unique()    #This gives different features in that attribute 
entropy_attribute = 0
for variable in variables:
    entropy_each_feature = 0
    for target_variable in target_variables:
        num = len(df[attribute][df[attribute]==variable][df.Action ==target_variable]) #numerator
        den = len(df[attribute][df[attribute]==variable])  #denominator
        fraction = num/(den+eps)  
        entropy_each_feature += -fraction*log(fraction+eps) #This calculates entropy for one feature
    fraction2 = den/len(df)
    entropy_attribute += -fraction2*entropy_each_feature
abs(entropy_attribute)

IG= entropy_node-abs(entropy_attribute) #calculates the information gain
IG
def find_entropy(df):
    Class = df.keys()[-1]   #To make the code generalized
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*np.log2(fraction)
    return entropy
def find_entropy_attribute(df,attribute):
  Class = df.keys()[-1]   #To make the code generic, 
  target_variables = df[Class].unique()  
  variables = df[attribute].unique()    #This gives different features in that attribute 
  entropy2 = 0
  for variable in variables:
      entropy = 0
      for target_variable in target_variables:
          num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
          den = len(df[attribute][df[attribute]==variable])
          fraction = num/(den+eps)
          entropy += -fraction*log(fraction+eps)
      fraction2 = den/len(data1)
      entropy2 += -fraction2*entropy
  return abs(entropy2)
def find_winner(df):
    Entropy_att = []
    IG = []
    for key in df.keys()[:-1]:

        IG.append(find_entropy(df)-find_entropy_attribute(df,key))
    return df.keys()[:-1][np.argmax(IG)]
def get_subtable(df, node,value):
  return df[df[node] == value].reset_index(drop=True)

def buildTree(data1,tree=None):
    Class = df.keys()[-1]   #To make the code generic

        #Get attribute with maximum information gain by comapring all
    node = find_winner(df)

    #Get distinct value of that attribute
    attValue = np.unique(df[node])

    #Create an empty dictionary path to create tree    
    if tree is None:
        tree={}
        tree[node] = {}

   #We make loop to build a tree by calling this function recursively. 
     

    for value in attValue:

        subtable = get_subtable(data1,node,value)
        clValue,counts = np.unique(subtable['Action'],return_counts=True)

        if len(counts)==1:#Checking purity
            tree[node][value] = clValue[0]
        else:
            tree[node][value] = buildTree(subtable) #This calls the function in recursive way

    return tree

tree = buildTree(df)


tree #prints tree


import pprint
pprint.pprint(tree) #another way to print tree


def predict(inst,tree):
    #This function is used for prediction using tree structure

    #here recursive operation takes place

    for nodes in tree.keys():

        value = inst[nodes]
        tree = tree[nodes][value]
        prediction = 0

        if type(tree) is dict:
            prediction = predict(inst, tree)
        else:
            prediction = tree
            break;

    return prediction


inst = df.iloc[6] #gets the sixth one on the dataset

inst  #displays the sixth one in the dataset

data = {'HasJob':'no', 'HasInsurance':'yes', 'Votes':'yes'} #prediction 1
inst = pd.Series(data)
prediction = predict(inst, tree)
prediction


data = {'HasJob':'yes', 'HasInsurance':'no', 'Votes':'yes'}#prediction 2
inst = pd.Series(data)
prediction = predict(inst, tree)
prediction


data = {'HasJob':'no', 'HasInsurance':'yes', 'Votes':'no'}#prediction 3
inst = pd.Series(data)
prediction = predict(inst, tree)
prediction

