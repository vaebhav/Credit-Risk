#!/usr/local/bin/python3

import math
import numpy as np
import pandas as pd
import collections
import re
import enum
import itertools
import matplotlib as plt


def getIndex(iptList):
    opList=list()
    for x in iptList:
        if x is None:
            next
        else:
            opList.append(iptList[iptList == x].index[0])
    return(opList)

def checkTye(val,dataframe):
    regex = r"(^[0-9]$)"
    #regex = r"([0-9])"
    string = str(val.values)
    x = re.search(regex, string)
    if x :
        col = val.name
        dataframe[col] = dataframe[col].astype(float,copy=True)
        return(x.group())
    else:
        next

def EnumHash(objList,dataFrame):
	ReplaceHash = dict()
	for col in objList:
		uniq_set = tuple(pd.unique(dataFrame[col]))
		DynamicEnum_list = []
		uniq_set = sorted(uniq_set)
		if len(uniq_set) > 3:
			next
		else:
			for val in uniq_set:
				if isinstance(val,float):
					next
				else:
					DynamicEnum_list = enum.Enum('DynamicEnum',uniq_set,start=0)
					if col in ReplaceHash:
						if  DynamicEnum_list[val].name in ReplaceHash[col]:
							next
						else:
							ReplaceHash[col][DynamicEnum_list[val].name] = DynamicEnum_list[val].value
					else:
						ReplaceHash[col] = dict()
						if  DynamicEnum_list[val].name in ReplaceHash[col]:
							next
						else:
							ReplaceHash[col][DynamicEnum_list[val].name] = DynamicEnum_list[val].value
	return ReplaceHash

def ConvDF2List(df_train,series_train):
	count=0
	TempList = []
	for index in (df_train.index.values):
		TempList.append(df_train.ix[index].values)
		TempList[count] = np.append(TempList[count],series_train[index])
		count+=1
	return TempList

def FillMissing(dataFrame):
    dataFrame.replace("\?",np.nan,inplace=True,regex=True)
    dataFrame[:] = dataFrame.apply(lambda x: x.str.strip("\t") if x.dtype == "object" else x)
    dataFrame[:] = dataFrame.apply(lambda x: x.str.strip(" ") if x.dtype == "object" else x)
    df_dtype = dataFrame.dtypes
    df_column_dtype = collections.defaultdict(list)
    for col,datatype in df_dtype.items():
        if datatype in df_column_dtype:
            df_column_dtype["{0}".format(datatype)].append(col)
        else:
            df_column_dtype["{0}".format(datatype)].append(col)

        for key,value in df_column_dtype.items():
            if key == 'object':
                object_dtype = []
                for col in value:
                    #test_df[col] = test_df[col].interpolate(method='linear',inplace=False)
                    object_dtype.append(col)
                    dataFrame[col] = dataFrame[col].fillna(method='bfill')
            elif key =='int64' or key == 'int32' or key == 'int16' or key == 'int8':
                int_dtypes = []
                for col in value:
                    int_dtypes.append(col)
                    #mean = test_df[col].mean()
                    dataFrame[col] = dataFrame[col].fillna(dataFrame[col].mean())
            elif key =='float64' or key == 'float32' or key == 'float16' or key == 'float8':
                float_dtype = []
                for col in value:
                    float_dtype.append(col)
                    #mean = test_df[col].mean()
                    dataFrame[col] = dataFrame[col].fillna(dataFrame[col].mean())
    var_check = 0
    if 'int_dtypes' in locals():
        var_check = var_check + 2
    else:
        int_dtypes = list()
    if 'object_dtype' in locals():
        var_check = var_check + 4
    else:
        object_dtype = list()
    if 'float_dtype' in locals():
        var_check = var_check + 8
    else:
        #float_dtype = []
        float_dtype = list()

    tmpList = dataFrame[object_dtype].apply(checkTye,dataframe=dataFrame)
    tmpList = getIndex(tmpList)

    for x in tmpList:
        object_dtype.remove(x)
        float_dtype.append(x)

    return (dataFrame,object_dtype,int_dtypes,float_dtype)


def ComputeDtype(dataFrame):
    #dataFrame.replace("\?",np.nan,inplace=True,regex=True)
    dataFrame = dataFrame.apply(lambda x: x.str.strip("\t") if x.dtype == "object" else x).copy()
    dataFrame = dataFrame.apply(lambda x: x.str.strip(" ") if x.dtype == "object" else x).copy()
    df_dtype = dataFrame.dtypes
    df_column_dtype = collections.defaultdict(list)
    for col,datatype in df_dtype.items():
        if datatype in df_column_dtype:
            df_column_dtype["{0}".format(datatype)].append(col)
        else:
            df_column_dtype["{0}".format(datatype)].append(col)

        for key,value in df_column_dtype.items():
            if key == 'object':
                object_dtype = []
                for col in value:
                    #test_df[col] = test_df[col].interpolate(method='linear',inplace=False)
                    object_dtype.append(col)
            elif key =='int64' or key == 'int32' or key == 'int16' or key == 'int8':
                int_dtypes = []
                for col in value:
                    int_dtypes.append(col)
                    #mean = test_df[col].mean()
            elif key =='float64' or key == 'float32' or key == 'float16' or key == 'float8':
                float_dtype = []
                for col in value:
                    float_dtype.append(col)
                    #mean = test_df[col].mean()
    var_check = 0
    if 'int_dtypes' in locals():
        var_check = var_check + 2
    else:
        int_dtypes = list()
    if 'object_dtype' in locals():
        var_check = var_check + 4
    else:
        object_dtype = list()
    if 'float_dtype' in locals():
        var_check = var_check + 8
    else:
        #float_dtype = []
        float_dtype = list()

    tmpList = dataFrame[object_dtype].apply(checkTye,dataframe=dataFrame)
    tmpList = getIndex(tmpList)

    for x in tmpList:
        object_dtype.remove(x)
        float_dtype.append(x)

    return (object_dtype,int_dtypes,float_dtype)



def outliers_modified_z_score(ys):
    threshold = 3.5

    median_y = np.median(ys)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                         for y in ys]
    return ys.iloc[np.where(np.abs(modified_z_scores) > threshold)]
#plt.show()

def outliers_z_score(ys):
    threshold = 3

    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    return ys.iloc[np.where(np.abs(z_scores) > threshold)]

def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return ys.iloc[np.where((ys > upper_bound) | (ys < lower_bound))]
