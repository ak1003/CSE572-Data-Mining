import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature_extraction import Features
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
import os
class DataSetFormation:
    def __init__(self):
        self.mealDataFrame=None
        self.noMealDataFrame=None
        self.normalized_data=None
        # self.plotCGMData()

    def read_csv(self):
        print("Here")
        meal_data=[]
        no_meal_data=[]
        meal_label=[]
        nomeal_label=[]
        columns=["c1","c2","c3","c4","c5","c6","c7","c8","c9","c10","c11","c12","c13","c14","c15"
        ,"c16","c17","c18","c19","c20","c21","c22","c23","c24","c25","c26","c27","c28","c29","c30"]
        # a=pd.read_csv('MealNoMealData/mealData1.csv',names=columns)
        for val in [1,2,3,4,5]:
            meal_data.append(pd.read_csv('MealNoMealData/mealData'+str(val)+'.csv',names=columns))
        for val in [1,2,3,4,5]:
            no_meal_data.append(pd.read_csv('MealNoMealData/Nomeal'+str(val)+'.csv',names=columns))
        self.mealDataFrame = pd.concat(meal_data,ignore_index=True)
        self.noMealDataFrame = pd.concat(no_meal_data,ignore_index=True)

        # print(self.noMealDataFrame)
        # print("-----------")
        # print(self.mealDataFrame)
    
    def createFeatureMatrixCGM(self):
        row,column=self.noMealDataFrame.shape
        for i in range(row):    
            self.noMealDataFrame.dropna(thresh=4,axis = 0,inplace=True)

        row,column=self.mealDataFrame.shape
        for i in range(row):    
            self.mealDataFrame.dropna(thresh=4,axis = 0,inplace=True)
        self.noMealDataFrame=self.noMealDataFrame.interpolate(method ='linear', limit_direction ='backward')
        self.mealDataFrame=self.mealDataFrame.interpolate(method ='linear', limit_direction ='backward')

        # print("here",self.noMealDataFrame)
        # print("0-------0")
        # print("here",self.mealDataFrame)

    def applyPCA(self,normalized_features,number):
        pca = PCA(n_components=number)
        pComponents= pca.fit_transform(normalized_features.values)
        pComponentsDataFrame = pd.DataFrame(data = pComponents
             , columns = ['pc1', 'pc2','pc3','pc4','pc5'])
        # print(pComponentsDataFrame)
        print((pca.components_ ))
        print(pca.explained_variance_ratio_)
        return pComponentsDataFrame

    def normalizeData(self,extracted_features):
        columns=['fft1','fft2','fft3','fft4','velocity1','velocity2','velocity3','velocity4',
        'rolling1','rolling2',
        'dwt1','dwt2','dwt3','dwt4']
        data=pd.DataFrame(extracted_features,columns=columns)
        data=data.dropna()
        print(data.head())
        data=StandardScaler().fit_transform(data.values)
        data=pd.DataFrame(data,columns=columns)
        return data

s=DataSetFormation()
s.read_csv()
s.createFeatureMatrixCGM()
mealFeatures=Features(4)
s.mealDataFrame.to_csv("myMealData.csv")
noMealFeatures=Features(4)
s.noMealDataFrame.to_csv("myNoMealData.csv")
finalMealDataFrame=pd.read_csv("myMealData.csv")
finalNoMealDataFrame=pd.read_csv("myNoMealData.csv")
meal=mealFeatures.completefeatures(finalMealDataFrame)
print(meal)
print("Final Meal DataSet")
mealPrincipalComponentDataFrame=s.normalizeData(meal)
nomeal=noMealFeatures.completefeatures(finalNoMealDataFrame)
print(nomeal)
print("Here",mealPrincipalComponentDataFrame)
mealPrincipalComponentDataFrame['Label'] = 1
print("Final NoMeal DataSet")
noMealPrincipalComponentDataFrame=s.normalizeData(nomeal)
noMealPrincipalComponentDataFrame['Label'] = 0
print("Here",noMealPrincipalComponentDataFrame)

#Concatinating 2 dataframes
finalMealNoMealDataFrame = pd.concat([mealPrincipalComponentDataFrame, noMealPrincipalComponentDataFrame], axis=0)
columns=['fft1','fft2','fft3','fft4','velocity1','velocity2','velocity3','velocity4',
        'rolling1','rolling2',
        'dwt1','dwt2','dwt3','dwt4']
# columns=['fft1','velocity1','velocity2',
#         'rolling1',
#         'dwt1']
# finalPCADataFrame=s.applyPCA(finalMealNoMealDataFrame[columns],5)
# print(finalPCADataFrame.shape)
# print(finalMealNoMealDataFrame['Label'].shape)
# finalPCADataFrame['Label']=finalMealNoMealDataFrame['Label'].values
# finalPCADataFrame+finalMealNoMealDataFrame['Label']
finalMealNoMealDataFrame.to_csv("finalMealNoMeal.csv")

readingDataSet=pd.read_csv("finalMealNoMeal.csv")
readingDataSet.to_csv("finalMealNoMeal.csv")



