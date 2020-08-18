import numpy as np
import pandas as pd
import pywt
import os
class Features:
    def __init__(self,featuresNumber):
        self.featuresNumber=featuresNumber
        self.final_matrix=[]
        self.final_fft=None
        self.velocity=None
        self.rolling20=None
    #Fast Fourier Transform
    def completefeatures(self,feature_matrix):
        # print("len(self.feature_matrix)",len(feature_matrix))
        for i in range(len(feature_matrix)):
            self.current_list=[]
            #First we are going to find the fast fourier tranform of the features of the first
            #person and taking only top feature by sorting higest to lowest.For now featurNumber is 6
            glucose_level=feature_matrix.loc[i]
            # print(glucose_level.values)
            self.final_fft=np.fft.fft(glucose_level)
            # print(self.final_fft)
            unsorted_fft=self.final_fft
            self.final_fft=sorted(np.abs(self.final_fft),reverse=True)
            self.current_list.extend(self.final_fft)
            #we will find zero crossing
            #Second we will find the velocity between t+1 and t
            if i+1<len(feature_matrix):
                self.velocity=feature_matrix.loc[i]-feature_matrix.loc[i+1]
                unsorted_velocity=self.velocity
                self.velocity=sorted(self.velocity,reverse=True)
                self.current_list.extend(self.velocity)
            #Third feature is going to rolling mean in the given CGM Data with window of 12 and 20
            # self.rolling12=self.feature_matrix[i]['GlucoseLevel'].rolling(12,center=True).mean().shift()
            self.rolling20=feature_matrix.loc[i].rolling(4).mean()
            # self.rolling12=self.rolling12.fillna(0)
            self.rolling20=self.rolling20.fillna(0)
            unsorted_rolling=self.rolling20
            self.rolling20=sorted(self.rolling20,reverse=True)
            self.current_list.extend(self.rolling20)

            #Fifth feature is discrete wavelenth transform which is better feature than 
            #fft as it captures the change in the vlaue more accurately
            glucose_level=feature_matrix.loc[i]
            (current_value,n_current_value)=pywt.dwt(glucose_level,'db1',mode='sym')
            unsorted_dwt=current_value
            current_value=sorted(current_value,reverse=True)
            self.current_list.extend(current_value)
            self.final_matrix.append(self.current_list)

            # glucose_level=feature_matrix.loc[i]
            # # print(len(glucose_level))
            # x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
            # # print("I am here")
            # # print(np.polyfit(x,glucose_level,4))
            # self.current_list.extend(np.polyfit(x,glucose_level,5))
            # self.final_matrix.append(self.current_list)



      

        return self.final_matrix