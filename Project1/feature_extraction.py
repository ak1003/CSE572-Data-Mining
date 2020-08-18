import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import os
class Features:

    def __init__(self,featuresNumber,feature_matrix):
        self.featuresNumber=featuresNumber
        self.feature_matrix=feature_matrix
        self.final_matrix=[]
        self.final_fft=None
        self.velocity=None
        self.rolling12=None
        self.rolling20=None
        self.expandingWindow=None
        # self.completefeatures()
    def plotFeatures(self,current_value,actual_value,i,folder_name,person):
        pathDirectory=os.getcwd()
        access_right=0o777
        try:
            if not os.path.isdir(pathDirectory+'/Person'+str(person)+'/'+folder_name):
                os.mkdir(pathDirectory+'/Person'+str(person)+'/'+folder_name,access_right)
        except OSError:
            print('Directory was not created')
        plt.clf()
        plt.figure(figsize=(8,8))
        plt.subplot(2, 1, 1)
        plt.grid(True)
        plt.stem(current_value)
        plt.xlabel('Features  after conversion')
        plt.ylabel('Value of the features')
        plt.subplot(2,1,2)
        plt.grid(True)
        plt.stem(actual_value)
        plt.xlabel('Top Features selected after conversion')
        plt.ylabel('Value of the features')
        plt.savefig('Person'+str(person)+'/'+folder_name+'/fig'+str(i)+'.png')
        plt.close()
    #Fast Fourier Transform
    def completefeatures(self,person):
        # print("len(self.feature_matrix)",len(self.feature_matrix))
        for i in range(len(self.feature_matrix)):
            self.current_list=[]
            #First we are going to find the fast fourier tranform of the features of the first
            #person and taking only top feature by sorting higest to lowest.For now featurNumber is 6
            glucose_level=self.feature_matrix[i]['GlucoseLevel'].values
            self.final_fft=np.fft.fft(glucose_level)
            unsorted_fft=self.final_fft
            self.final_fft=sorted(np.abs(self.final_fft),reverse=True)
            self.current_list.extend(self.final_fft[1:self.featuresNumber+1])
            self.plotFeatures(unsorted_fft,self.final_fft[1:self.featuresNumber+1],i,'FFT',person)
            #Second we will find the velocity between t+1 and t
            if i+1<len(self.feature_matrix):
                self.velocity=np.abs(self.feature_matrix[i]['GlucoseLevel'].values-self.feature_matrix[i+1]['GlucoseLevel'].values)
                unsorted_velocity=self.velocity
                self.velocity=sorted(self.velocity,reverse=True)
                self.current_list.extend(self.velocity[:self.featuresNumber])
            self.plotFeatures(unsorted_velocity,self.velocity[:self.featuresNumber],i,'Velocity',person)
            #Third feature is going to rolling mean in the given CGM Data with window of 12 and 20
            # self.rolling12=self.feature_matrix[i]['GlucoseLevel'].rolling(12,center=True).mean().shift()
            self.rolling20=self.feature_matrix[i]['GlucoseLevel'].rolling(16,center=True).mean().shift()
            # self.rolling12=self.rolling12.fillna(0)
            self.rolling20=self.rolling20.fillna(0)
            # print("rolling12",type(self.rolling12))
            # self.rolling12=sorted(self.rolling12,reverse=True)
            # self.current_list.extend(self.rolling12[:self.featuresNumber])
            unsorted_rolling=self.rolling20
            self.rolling20=sorted(self.rolling20,reverse=True)
            self.current_list.extend(self.rolling20[:self.featuresNumber])
            self.plotFeatures(unsorted_rolling,self.rolling20[:self.featuresNumber],i,'RollingWindow',person)


            #Fourth Feature expanding window feature for time series
            self.expandingWindow=self.feature_matrix[i]['GlucoseLevel'].expanding(4).mean()
            self.expandingWindow=self.expandingWindow.fillna(0)
            unsorted_expandingwindow=self.expandingWindow
            self.expandingWindow=sorted(self.expandingWindow,reverse=True)
            self.current_list.extend(self.expandingWindow[:self.featuresNumber])
            self.plotFeatures(unsorted_expandingwindow,self.expandingWindow[:self.featuresNumber],i,'ExpandingWindow',person)
            # print("Current",len(self.current_list))
            #Final Current List of created Feature
            #Discrete Wavelate Transform
            
            #Fifth feature is discrete wavelenth transform which is better feature than 
            #fft as it captures the change in the vlaue more accurately
            glucose_level=self.feature_matrix[i]['GlucoseLevel'].values
            # print("Glucose",glucose_level)
            (current_value,n_current_value)=pywt.dwt(glucose_level,'db1',mode='sym')
            unsorted_dwt=current_value
            current_value=sorted(current_value,reverse=True)
            # print("current")
            # print(current_value)
            self.current_list.extend(current_value[:self.featuresNumber])
            self.final_matrix.append(self.current_list)
            self.plotFeatures(unsorted_dwt,current_value[0:self.featuresNumber],i,'DFT',person)
      

        return self.final_matrix