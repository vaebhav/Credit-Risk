#!/usr/local/bin/python3


import math
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from numpy import loadtxt, where
import collections
import re
import enum
import random
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression,LinearRegression
from pylab import scatter, show, legend, xlabel, ylabel
from sklearn.metrics import roc_curve, auc, roc_auc_score, log_loss, accuracy_score, confusion_matrix
import DataCleaning as dc
import itertools




def GD_matrix(X_train,Y_train,weights,lr,m,epoch):

    cost_history = []

    # # Sigmoid Calc

    Z = np.array(np.dot(X_train, weights),dtype=np.float32)

    #Z = np.dot(X_train, weights)

    A = 1 / (1 + np.exp(-Z))
    cost_history.append(Z)

    #Cost Computation
    J = np.sum(-(Y_train * np.log(A) + (1 - Y_train) * np.log(1 - A))) / m

    # Gradient computation

    dZ = A - Y_train
    dw = np.dot(dZ, X_train) / m

    db = np.sum(dZ) / m

    # Update weights

    weights = weights - lr * dw
    #biais = biais - lr * db

    #if epoch % 10 == 0:
        #print("epoch %s - loss %s" % (epoch, J))
    #J_arr.append(J)

    return weights,J

def Sigmoid(z):
    G_of_Z = float(1.0 / float((1.0 + np.exp(-1.0*z))))
    #print(z)
    return G_of_Z

##The hypothesis is the linear combination of all the known factors x[i] and their current estimated coefficients theta[i]
##This hypothesis will be used to calculate each instance of the Cost Function
def Hypothesis(theta, x):
    z = 0
    for i in range(len(theta)):
		#print("xi",x[i])
		#print("thetha",theta[i])
        z += x[i]*theta[i]
    return Sigmoid(z)

def Logistic_Regression(X,Y,alpha,theta,num_iters,learningRate_method='normal'):
    m = len(Y)
    cost = 0
    for x in range(num_iters):
        #print("Old Thetha--->",theta)
        old_cost = cost
        new_theta,cost = GD_matrix(X,Y,theta,alpha,m,x)
        theta = new_theta

        ####Bold Driver approach for adaptive learning rate

        if learningRate_method == 'bold driver':
            if old_cost - cost < 0:
                alpha = alpha - alpha * 0.5
            else:
                alpha = alpha + alpha * 0.05


        #print("New Theta--->",theta)
        #print("New Thetha--->",theta)
        #if x % 100 == 0:
			#here the cost function is used to present the final hypothesis of the model in the same form for each gradient-step iteration
			#Cost_Function(X,Y,theta,m)
			#print("------>",x)
            #print('theta---->',theta)
            #print('New Cost----->',cost)
            #print('Old Cost----->',old_cost)
            #print('Learning Rate----->',alpha)
    print('Learning Rate----->',alpha)
    return theta

def Declare_Winner(X_test,Y_test,theta):
    score = 0
    winner = ""
    pred_arr = []
    ans_arr = []
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    length = len(X_test)

    for i in range(length):
        prediction = round(Hypothesis(X_test.iloc[i],theta))
        answer = Y_test.iloc[i]
        #print("Pred->",prediction)
        #print("Ans->",answer)

        if prediction == answer:
            score += 1

        pred_arr.append(prediction)
        ans_arr.append(answer)

        if answer == 1 and prediction == 1:
            TP += 1
        elif answer == 0 and prediction == 0:
            TN += 1
        elif answer == 0 and prediction == 1:
            FN += 1
        elif answer == 1 and prediction == 0:
            FP += 1


    conf_matrix = np.array([[TP,TN],[FN,FP]])
    index = np.array(np.unique(ans_arr))

    conf_pivot = pd.crosstab(pd.Series(ans_arr),pd.Series(pred_arr))

    my_score = float(score) / float(length)

    print('Your score: ', my_score)
    return my_score


def Predict(dataframe,theta):
    score = 0
    winner = ""
    predictions = []
    length = len(dataframe)
    for i in range(length):
        #pred = round(Hypothesis(dataframe.iloc[i],theta))
        pred = round(Hypothesis(dataframe.iloc[i],theta))
        predictions.append(pred)

    return np.array(predictions)

def PredictProb(dataframe,theta):
    score = 0
    winner = ""
    predictions = []
    length = len(dataframe)
    for i in range(length):
        #pred = round(Hypothesis(dataframe.iloc[i],theta))
        pred = Hypothesis(dataframe.iloc[i],theta)
        predictions.append(pred)

    return np.array(predictions)




def plot_cm(ax, y_true, y_pred, classes, title, th=0.5, cmap=plt.cm.Blues):
    y_pred_labels = (y_pred>th).astype(int)

    cm = confusion_matrix(y_true, y_pred_labels)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

def plot_auc(ax, y_train, y_train_pred, y_test, y_test_pred, th=0.5):

    y_train_pred_labels = (y_train_pred>th).astype(int)
    y_test_pred_labels  = (y_test_pred>th).astype(int)

    fpr_train, tpr_train, _ = roc_curve(y_train,y_train_pred)
    roc_auc_train = auc(fpr_train, tpr_train)
    acc_train = accuracy_score(y_train, y_train_pred_labels)

    fpr_test, tpr_test, _ = roc_curve(y_test,y_test_pred)
    roc_auc_test = auc(fpr_test, tpr_test)
    acc_test = accuracy_score(y_test, y_test_pred_labels)

    ax.plot(fpr_train, tpr_train)
    ax.plot(fpr_test, tpr_test)

    ax.plot([0, 1], [0, 1], 'k--')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve')

    train_text = 'train acc = {:.3f}, auc = {:.2f}'.format(acc_train, roc_auc_train)
    test_text = 'test acc = {:.3f}, auc = {:.2f}'.format(acc_test, roc_auc_test)
    ax.legend([train_text, test_text])

def plot_auc_train(ax, y_train, y_train_pred,th=0.5):

    y_train_pred_labels = (y_train_pred>th).astype(int)

    fpr_train, tpr_train, _ = roc_curve(y_train,y_train_pred)
    roc_auc_train = auc(fpr_train, tpr_train)
    acc_train = accuracy_score(y_train, y_train_pred_labels)

    ax.plot(fpr_train, tpr_train)

    ax.plot([0, 1], [0, 1], 'k--')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve')

    train_text = 'acc = {:.3f}, auc = {:.2f}'.format(acc_train, roc_auc_train)
    ax.legend([train_text])
