import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature_extraction import Features
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
import os
import seaborn as sns
class DataSetFormation:
    def __init__(self,number):
        self.CGMData=[]
        self.CGMDataTime=None
        self.CGMDataLunch=None
        self.read_csv(number)
        self.createFeatureMatrixCGM()
        self.normalized_data=None
        # self.plotCGMData()

    def read_csv(self,number):
        self.CGMDataTime=pd.read_csv('DataFolder/CGMDatenumLunchPat'+str(number)+'.csv')
        print(self.CGMDataLunch)
        self.CGMDataLunch=pd.read_csv('DataFolder/CGMSeriesLunchPat'+str(number)+'.csv')

    def createFeatureMatrixCGM(self):
        print(self.CGMDataTime.shape)
        row,column=self.CGMDataTime.shape
        # print("I am Here")
        CGMDataTimeInverted=None
        CGMDataLunchTimeInverted=None
        # print(self.CGMDataLunch.loc[0])
        # print('--------------------')
        # print(self.CGMDataLunch.loc[0][::-1])
        for i in range(row):
            CGMDataTimeInverted=self.CGMDataTime.loc[i][::-1]
            CGMDataLunchTimeInverted=self.CGMDataLunch.loc[i][::-1]
            # print(type(CGMDataLunchTimeInverted))
            lst=list(zip(CGMDataTimeInverted/100000,CGMDataLunchTimeInverted))
            data=pd.DataFrame(lst,columns=['Time','GlucoseLevel'])
            data=data.interpolate(method ='linear', limit_direction ='backward')
            self.CGMData.append(data)
        print("here",self.CGMData[0])


    def plotCGMData(self,person):
        for i in range(len(self.CGMData)):
            plt.figure(figsize=(8,8))
            plt.grid(True)
            plt.plot(self.CGMData[i]['Time'],self.CGMData[i]['GlucoseLevel'],color='red',label='GlucoseLevel')
            plt.xlabel('Value taken every 5 min in Day')
            plt.ylabel('Value of glucose/insulin level at given time')
            plt.savefig('Person'+str(person)+'/fig'+str(i)+'.png')
            plt.close()
    def plotPCA(self,pComponentsDataFrame,folder_name,person):
        print(person)
        plt.clf()
        plt.figure(figsize=(16,16))
        plt.subplot(6, 1, 1)
        plt.grid(True)
        plt.ylabel('PC1')
        plt.plot(pComponentsDataFrame['pc1'],color='blue')
        plt.subplot(6, 1, 2)
        plt.grid(True)
        plt.ylabel('PC2')
        plt.plot(pComponentsDataFrame['pc2'],color='blue')
        plt.subplot(6, 1, 3)
        plt.grid(True)
        plt.ylabel('PC3')
        plt.plot(pComponentsDataFrame['pc3'],color='blue')
        plt.subplot(6, 1, 4)
        plt.grid(True)
        plt.ylabel('PC4')
        plt.plot(pComponentsDataFrame['pc4'],color='blue')
        plt.subplot(6, 1, 5)
        plt.grid(True)
        plt.ylabel('PC5')
        plt.plot(pComponentsDataFrame['pc5'],color='blue')
        access_right=0o777
        try:
            if not os.path.isdir(directoryPath+'/Person'+str(person)+'/'+folder_name):
                os.mkdir(directoryPath+'/Person'+str(person)+'/'+folder_name,access_right)
        except OSError:
            print('Directoy not created')
        plt.savefig('Person'+str(person)+'/'+folder_name+'/fig.png')
        plt.close()

    def applyPCA(self,normalized_features,number,person,folder_name):
        pca = PCA(n_components=number)
        pComponents= pca.fit_transform(normalized_features.values)
        pComponentsDataFrame = pd.DataFrame(data = pComponents
             , columns = ['pc1', 'pc2','pc3', 'pc4','pc5'])
        pComponentsDataFrame.to_csv('Features.csv')
        print(pComponentsDataFrame)
        self.plotPCA(pComponentsDataFrame,'PCA',person)
        print('--------- Eigen Vector')
        df=pd.DataFrame(pca.components_)
        df.to_csv('EigenVector.csv')
        print((pca.components_ ))
        print('Eigen values')
        
        m=pd.DataFrame(pca.components_,columns=normalized_features.columns)
        plt.figure(figsize=(15,10))
        sns.heatmap(m,cmap='cividis_r')
        plt.savefig('Person'+str(person)+'/'+folder_name+'/heatmap.png')
        plt.close()
        print(pca.explained_variance_ratio_)
        
    def normalizeData(self,extracted_features,person):
        # data=pd.DataFrame()
        columns=['fft1','fft2','fft3','fft4','velocity1','velocity2','velocity3','velocity4',
        'rolling1','rolling2','rolling3','rolling4','expwindow1','expwindow2','expwindow3','expwindow4',
        'dwt1','dwt2','dwt3','dwt4']
        data=pd.DataFrame(extracted_features,columns=columns)
        data=data.dropna()
        print(data.head())
        data=MinMaxScaler().fit_transform(data.values)
        data=pd.DataFrame(data,columns=columns)
        self.applyPCA(data,5,person,'PCA')
print("""----------------------------------------|
|      Enter a Person Number                     |
|                                                |
|-----------------------------------------|""")
n=input()
directoryPath=os.getcwd()
access_right=0o777
try:
    if not os.path.isdir(directoryPath+'/Person'+str(n)):
        os.mkdir(directoryPath+'/Person'+str(n),access_right)
except OSError:
    print('Directoy not created')
s=DataSetFormation(int(n))
s.plotCGMData(int(n))
b=Features(4,s.CGMData)
final_extracted_feature_matrix=b.completefeatures(int(n))
df=pd.DataFrame(final_extracted_feature_matrix)
df.to_csv('FeaturesExtracted.csv')
s.normalizeData(final_extracted_feature_matrix,n)

