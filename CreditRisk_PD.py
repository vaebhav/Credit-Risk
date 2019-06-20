#!/usr/local/bin/python3

import math
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn import metrics
import DataCleaning as dc
import LogisticRegression_GD as logr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xlrd



def feature_scaling(df):
    df -= df.min()
    df /= df.max()
    return df

def feature_scaling_log(x):
    if x == 0:
        return 0
    elif x < 0:
        return 0
    else:
        return np.log(x)


credit_card_fraud_file = "/Users/vaebhav/Documents/Python/Machine Learning/Credit Risk/creditcardclients_default.xls"

credit_card_data = pd.read_excel(credit_card_fraud_file,header=1)

credit_card_data['status'] = credit_card_data['default payment next month']
credit_card_data = credit_card_data.drop(['ID','default payment next month'], axis=1)


edu_mask = (credit_card_data.EDUCATION == 5) | (credit_card_data.EDUCATION == 6) | (credit_card_data.EDUCATION == 0)
credit_card_data.loc[edu_mask, 'EDUCATION'] = 4


credit_card_data.loc[credit_card_data.MARRIAGE == 0, 'MARRIAGE'] = 3


credit_card_data['Limit_bal_log'] = credit_card_data['LIMIT_BAL'].apply(feature_scaling_log)

credit_card_data['age_cat'] = pd.cut(credit_card_data['AGE'], range(0, 100, 10),labels=np.arange(1,10),right=False)


credit_card_data['TotalBillAmt'] = credit_card_data[['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']].sum(axis=1)
credit_card_data['TotalPaidAmt'] = credit_card_data[['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']].sum(axis=1)




credit_card_data['DebitRatio'] =  credit_card_data['TotalPaidAmt'] / credit_card_data['TotalBillAmt']

#### As some bill and pay amts are 0 which results in inf for some DebitRatio values
#Hence inf was replaced with np.nan which was further filled as 0

credit_card_data.replace([np.inf, -np.inf],np.nan,inplace=True)
credit_card_data['DebitRatio'].fillna(0,inplace=True)

#credit_card_data['DebitRatio'].hist(bins=20)
#plt.show()


for i in np.arange(1,7):
    credit_card_data.drop('PAY_AMT'+str(i),axis=1,inplace=True)
    credit_card_data.drop('BILL_AMT'+str(i),axis=1,inplace=True)
    #credit_card_data.drop('PAY_'+str(i),axis=1,inplace=True)


#credit_card_data = pd.get_dummies(credit_card_data, columns=['SEX','MARRIAGE'])
credit_card_data = pd.get_dummies(credit_card_data, columns=['SEX','MARRIAGE','EDUCATION'])

#credit_card_data.drop(['SEX','EDUCATION','MARRIAGE','AGE'],axis=1,inplace=True)
##credit_card_data.drop(['AGE','EDUCATION'],axis=1,inplace=True)

credit_card_data.drop(['AGE'],axis=1,inplace=True)
credit_card_data.drop(['LIMIT_BAL','TotalPaidAmt','TotalBillAmt'],axis=1,inplace=True)

X_sampled = credit_card_data
X_sampled = X_sampled.drop(['status'],axis=1)
Y_sampled = credit_card_data.loc[:,'status']



X_train,X_test,Y_train,Y_test = train_test_split(X_sampled,Y_sampled,test_size=0.33)


cv = cross_validation.KFold(len(X_train), n_folds=5)
initial_theta = np.zeros(len(X_train.columns))


alpha = [0.001,0.01,0.1,0.5,0.005]
#alpha = 0.01

iterations = 1000


counter = 0

learningRate_method = 'bold driver'

max_score = 0
max_alpha = 0
max_theta = 0

for traincv, testcv in cv:
    #print("Alpha Value--->>>",alpha[counter])
    print("----------------------------")
    print("Kfold-------->",counter)
    print("----------------------------")
    #theta = logr.Logistic_Regression(X_train.iloc[traincv],Y_train.iloc[traincv],alpha,initial_theta,iterations)
    theta = logr.Logistic_Regression(X_train.iloc[traincv].as_matrix(),Y_train.iloc[traincv].as_matrix(),alpha[counter],initial_theta,iterations,learningRate_method)
    score = logr.Declare_Winner(X_test,Y_test,theta)
    if max_score < score:
        print(counter)
        max_alpha = alpha[counter]
        max_score = score
        #max_alpha = alpha
        max_theta = theta
    counter += 1
print("-------------------------------")
print("Max Alpha---->",max_alpha)
print("Max Score---->",max_score)
print("-------------------------------")


Y_train_pred_logreg = logr.PredictProb(X_train,max_theta)

Y_test_pred_logreg = logr.PredictProb(X_test,max_theta)


logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
logreg_score = logreg.score(X_test,Y_test)
print('score Scikit learn LogisticRegression: ', logreg_score)


Y_test_pred_skLogreg = logreg.predict_proba(X_test)[:,1]
Y_train_pred_skLogreg = logreg.predict_proba(X_train)[:,1]


rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5)
rf.fit(X_train,Y_train)
rf_score = rf.score(X_test,Y_test)
print("Score Scikit Learn Random Forest: ",rf_score)


if max_score > logreg_score:
    print("Self Logistic Reg implementation outperformed Scikit Learns Logsitic Reg")

if max_score > rf_score:
    print("Self Logistic Reg implementation outperformed Scikit Learns Random Forest")

Y_test_pred_skRF = rf.predict_proba(X_test)[:,1]
Y_train_pred_skRF = rf.predict_proba(X_train)[:,1]


fig_logreg,ax_logreg = plt.subplots(1,4)
fig_sklogreg,ax_sklogreg = plt.subplots(1,4)
fig_rf,ax_rf = plt.subplots(1,4)


fig_rf.set_size_inches(15,5)
fig_logreg.set_size_inches(15,5)
fig_sklogreg.set_size_inches(15,5)

threshold = 0.5


logr.plot_auc_train(ax_logreg[0],Y_train, Y_train_pred_logreg, threshold)
logr.plot_auc_train(ax_logreg[1],Y_test, Y_test_pred_logreg, threshold)
logr.plot_cm(ax_logreg[2],  Y_train, Y_train_pred_logreg, [0,1], 'Confusion matrix (TRAIN)', threshold)
logr.plot_cm(ax_logreg[3],  Y_test, Y_test_pred_logreg, [0,1], 'Confusion matrix (TEST)', threshold)

logr.plot_auc_train(ax_sklogreg[0],Y_train, Y_train_pred_skLogreg, threshold)
logr.plot_auc_train(ax_sklogreg[1],Y_test, Y_test_pred_skLogreg, threshold)
logr.plot_cm(ax_sklogreg[2],  Y_train, Y_train_pred_skLogreg, [0,1], 'Confusion matrix (TRAIN)', threshold)
logr.plot_cm(ax_sklogreg[3],  Y_test, Y_test_pred_skLogreg, [0,1], 'Confusion matrix (TEST)', threshold)


logr.plot_auc_train(ax_rf[0],Y_train, Y_train_pred_skRF, threshold)
logr.plot_auc_train(ax_rf[1],Y_test, Y_test_pred_skRF, threshold)
logr.plot_cm(ax_rf[2],  Y_train, Y_train_pred_skRF, [0,1], 'Confusion matrix (TRAIN)', threshold)
logr.plot_cm(ax_rf[3],  Y_test, Y_test_pred_skRF, [0,1], 'Confusion matrix (TEST)', threshold)



plt.show()
