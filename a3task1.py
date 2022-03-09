# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 10:21:38 2022

@author: cnava
"""

import sklearn

print('The scikit-learn version is {}.'.format(sklearn.__version__))

# The nltk version is 3.0.0.
# The scikit-learn version is 0.15.2.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
# load the data from file 
df = pd.read_csv("Engine_Condition_Comp.csv")
feature_names = ['EV1', 'EV2']
X = df[feature_names]
y = df['Status']



df_arr = np.array(df)
df_arr = df_arr[0:38, :]
D1 = df_arr[0:17, :]
D0 = df_arr[17:38,:]


# Create a function that predicts the label of a given example (entered as a row vector) given a set of weights.
def predict(row, weights):
    activation = weights[0]                  #initialize
    for i in range(len(row)-1):              #loop through rows input to the function
        activation += weights[i+1]*row[i]    #This is just w_j*x_{i,j} and the += is building w_0+w_1*x_{i,1}+w_2_{i,2} + ...
    g = 1/(1 + np.exp(-activation))
    return 1.0 if g >= 0.5 else 0.0          #Output the prediction, which is a 1 or 0 based on sigmoid(h(x)) >= 0.5 now.

def logisticfun(row, weights):
    activation = weights[0]                  #initialize
    for i in range(len(row)-1):              #loop through rows input to the function
        activation += weights[i+1]*row[i]    #This is just w_j*x_{i,j} and the += is building w_0+w_1*x_{i,1}+w_2_{i,2} + ...
    return 1/(1 + np.exp(-activation))       #output the value from the logistic (sigmoid) function             
    
# Create a function that finds the weights for the logistic regression model based on L2 loss (note, sklearn using logistic loss, which we use later)
def train_weights(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]   #initialize
    accuracy = []                                   #initialize
    for epoch in range(n_epoch):                    #loop through the planned number of epochs (trips through the data)
        for row in train:                           #loop through the rows in 'train' which is input data to this function
            g = logisticfun(row, weights)
            error = row[-1] - g                     #compute error (row[-1] is the last column of the row)
            weights[0] = weights[0] + l_rate * error * g * (1-g)      #update the bias weight (the w_0 entry)
            for i in range(len(row)-1):                                               #loop through the feature weights now
                weights[i+1] = weights[i+1] + l_rate * error * g * (1-g)* row[i]  #logistic regression update
            correct = 0                                                #reset correct to 0 each time we check accuracy
            for row in train:                                          #loop through the rows of data input
                prediction_accuracy = predict(row, weights)            #predict the label (0 or 1) for the current row
                if row[-1] == prediction_accuracy:                     #compare predicted label to actual label
                    correct += 1                                       #increment 'correct' by 1 if label is correct
            accuracy.append(correct / float(len(train[:, 2])) * 100.0) #append accuracy score each time through the loop
    return weights, accuracy     

#Holdout Validation


X_train, X_test, y_train, y_test = train_test_split(df_arr[:,0:2], df_arr[:, 2], train_size=0.8, random_state=42)

l_rate = 0.1
n_epoch = 150
weights, accuracy = train_weights(np.hstack((X_train, y_train.reshape(-1, 1))), l_rate, n_epoch)

print('Learned Weights:')
print(weights)

# plot the actual boundary
x1 = np.linspace(4.4, 6.3, num=2)
x2 = - weights[0]/weights[2] - weights[1]/weights[2]*x1

plt.scatter(D1[:, 0], D1[:, 1], marker='o', s=50, color='g', edgecolor='k', label='Good')
plt.scatter(D0[:, 0], D0[:, 1], marker='s', s=50, color='r', edgecolor='k', label='Diminished')
plt.plot(x1, x2, label='Decision Boundary')
plt.xlabel("EV1")
plt.ylabel("EV2")
plt.legend()




import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split( X, y, random_state=0)
clf = SVC(random_state=0)
clf.fit(X_train, y_train)
SVC(random_state=0)
plot_confusion_matrix(clf, X_test, y_test)  
plt.show()

test = np.hstack((X_test, y_test.reshape(-1, 1)))

y_pred = []
y_true = []
for row in test:
    y_pred.append(predict(row, weights))
    y_true.append(row[-1])

disp= ConfusionMatrixDisplay.from_predictions(y_true, y_pred)

target_names = ['Good', 'Diminished']
print(classification_report(y_true, y_pred, target_names=target_names))




plt.show()




#Plotting our data

# define and train the model
# extract trained weights and report them
# visualize samples and decision boundary
# evaluate the test set and display confusion matrix