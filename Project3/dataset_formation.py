import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature_extraction import Features
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN,KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier
class DataSetFormation:
    def __init__(self):
        self.mealDataFrame=None
        self.normalized_data=None
        self.carbIntakeDataFrame=None
        self.completeDataFrame=None
        self.finalPCADataFrame=None
        self.labels=None
        self.finalDataFrame=None
        #, >0 to 20, 21 to 40, 41 to 60, 61 to 80, 81 to 100. 
        self.groundTruthDictionary={}
        # self.plotCGMData()

    def read_csv(self):
        meal_data=[]
        carb_intake=[]
        final_carb_intake=[]
        columns_glucose=["c1","c2","c3","c4","c5","c6","c7","c8","c9","c10","c11","c12","c13","c14","c15"
        ,"c16","c17","c18","c19","c20","c21","c22","c23","c24","c25","c26","c27","c28","c29","c30"]
        columns_carb=['carb']
        for val in [1,2,3,4,5]:
            meal_data.append(pd.read_csv('mealData'+str(val)+'.csv',names=columns_glucose))

        for val in [1,2,3,4,5]:
            length=len(carb_intake)
            carb_intake.append(pd.read_csv('mealAmountData'+str(val)+'.csv',names=columns_carb,nrows=51))
            # print((final_carb_intake))
        self.mealDataFrame = pd.concat(meal_data,ignore_index=True)
        # print(self.mealDataFrame.head)
        # print(len(self.mealDataFrame))
        self.carbIntakeDataFrame=pd.concat(carb_intake,ignore_index=True)
        # print(self.carbIntakeDataFrame.head())
        # print(len(self.carbIntakeDataFrame))
        self.completeDataFrame=pd.concat([self.mealDataFrame, self.carbIntakeDataFrame], axis=1)
        # print(self.completeDataFrame)
    
    def createFeatureMatrixCGM(self):

        row,column=self.completeDataFrame.shape
        for i in range(row):    
            self.completeDataFrame=self.completeDataFrame.dropna(thresh=4,axis = 0)
        self.completeDataFrame=self.completeDataFrame.interpolate(method ='linear', limit_direction ='backward')
        self.completeDataFrame=self.completeDataFrame.reset_index(drop=True)
        # print(self.completeDataFrame)
    def applyPCA(self,normalized_features,number):
        pca = PCA(n_components=number)
        pComponents= pca.fit_transform(normalized_features.values)
        pComponentsDataFrame = pd.DataFrame(data = pComponents
             , columns = ['pc1', 'pc2'])
        # print(pComponentsDataFrame)
        # print((pca.components_ ))
        # print(pca.explained_variance_ratio_)
        return pComponentsDataFrame

    def normalizeData(self,extracted_features):
        columns_selected=['fft1','fft2','fft3','fft4','velocity1','velocity2','velocity3','velocity4',
        'rolling1','rolling2',
        'dwt1','dwt2','dwt3']
        data=pd.DataFrame(extracted_features,columns=columns_selected)
        data=data.interpolate(method ='linear', limit_direction ='backward')
        data=data[columns_selected]
        data=StandardScaler().fit_transform(data.values)
        data=pd.DataFrame(data,columns=columns_selected)
        return data
    def createGroundTruth(self):
        bin0=[]
        bin20=[]
        bin40=[]
        bin60=[]
        bin80=[]
        bin100=[]
        for i in range(len(self.completeDataFrame)):
            val=self.completeDataFrame['carb'].loc[i]
            if val==0:
                bin0.append(i)
            elif val>0 and val<=20:
                bin20.append(i)
            elif val>20 and val<=40:
                bin40.append(i)
            elif val>40 and val<=60:
                bin60.append(i)
            elif val>60 and val<=80:
                bin80.append(i)
            else:
                bin100.append(i)
        
        self.groundTruthDictionary['0']=bin0
        self.groundTruthDictionary['20']=bin20
        self.groundTruthDictionary['40']=bin40
        self.groundTruthDictionary['60']=bin60
        self.groundTruthDictionary['80']=bin80
        self.groundTruthDictionary['100']=bin100

        print("Dictionary")
        print(self.groundTruthDictionary)
    def getFeatures(self):
        columns_glucose=["c1","c2","c3","c4","c5","c6","c7","c8","c9","c10","c11","c12","c13","c14","c15"
        ,"c16","c17","c18","c19","c20","c21","c22","c23","c24","c25","c26","c27","c28","c29","c30"]
        meal=mealFeatures.completefeatures(self.completeDataFrame[columns_glucose])
        return meal

    def createDBSCANClusterFromFeatures(self,mealPrincipalComponentDataFrame):
        columns=['velocity1','velocity2','fft1','fft2','dwt1']
        self.finalPCADataFrame=s.applyPCA(mealPrincipalComponentDataFrame[columns],2)
        print(self.finalPCADataFrame.head())
        # db_default = DBSCAN(eps = 0.396, min_samples =6).fit(self.finalPCADataFrame) 
        # db_default = DBSCAN(eps = 0.376, min_samples =6).fit(self.finalPCADataFrame) 
        db_default = DBSCAN(eps = 0.341, min_samples =4).fit(self.finalPCADataFrame) 
        y_pred = db_default.fit_predict(self.finalPCADataFrame)
        self.labels = db_default.labels_
        print(self.labels)
        print(y_pred)
    
    def createDBSCANClusterFromFeaturesMax(self):
        extracted_features=[]
        for i in range(len(self.labels)):
            print(self.labels[i])
            if self.labels[i]==0:
                extracted_features.append(self.finalPCADataFrame.loc[i])
        extractedDataFrame=pd.DataFrame(extracted_features)
        print(extractedDataFrame)
        kmeans = KMeans(n_clusters=3, random_state=0).fit(extractedDataFrame)

    def createKMeansCluster(self,mealPrincipalComponentDataFrame):
        columns=['velocity1','velocity2','velocity3','fft1','fft2','dwt1','rolling2']
        self.finalPCADataFrame=s.applyPCA(mealPrincipalComponentDataFrame[columns],2)
        kmeans = KMeans(n_clusters=6, random_state=22).fit(self.finalPCADataFrame)
        y_pred = kmeans.fit_predict(self.finalPCADataFrame)
        self.labels = kmeans.labels_
     

    
    def SSEMetrics(self,):
        neighbours = NearestNeighbors(n_neighbors=2)
        neighbour = neighbours.fit(self.finalPCADataFrame)
        distances, indices = neighbour.kneighbors(self.finalPCADataFrame)
        distances = np.sort(distances, axis=0)
        distances = distances[:,1]
        plt.plot(distances)
        plt.show()

    def plotPointCluster(self):
        pass
    def return_cluster(self,cluster):
        if cluster=='0':
            return 1
        elif cluster=='20':
            return 2
        elif cluster=='40':
            return 3
        elif cluster=='60':
            return 4
        elif cluster=='80':
            return 5
        elif cluster=='100':
            return 6
    def calculateAccuracy(self):
        bin_calulated_0=[]
        bin_calulated_1=[]
        bin_calulated_2=[]
        bin_calulated_3=[]
        bin_calulated_4=[]
        bin_calulated_5=[]
        for i in range(len(self.labels)):
            if self.labels[i]==0:
                bin_calulated_0.append(i)
            elif self.labels[i]==1:
                bin_calulated_1.append(i)
            elif self.labels[i]==2:
                bin_calulated_2.append(i)
            elif self.labels[i]==3:
                bin_calulated_3.append(i)
            elif self.labels[i]==4:
                bin_calulated_4.append(i)
            elif self.labels[i]==5:
                bin_calulated_5.append(i)
        maxValue_0=0
        maxValue_1=0
        maxValue_2=0
        maxValue_3=0  
        maxValue_4=0  
        maxValue_5=0
        features=[]
        label_cluster=[]     
        for item,value in self.groundTruthDictionary.items():
            currentValue=0
            # print("Bin0")
            # print(value)
            # print("------------------")
            # print(bin_calulated_0)
            for i in range(len(value)):
                if value[i] in bin_calulated_0:
                    currentValue+=1
            # print(currentValue)
            if currentValue>=maxValue_0:
                maxValue_0=currentValue
                cluster=item
        print("Selected",cluster)
        print(maxValue_0)
        for i in range(len(bin_calulated_0)):
            features.append(bin_calulated_0[i])
            label_cluster.append(self.return_cluster(cluster))
        del self.groundTruthDictionary[cluster]
        print(self.groundTruthDictionary)
        for item,value in self.groundTruthDictionary.items():
            currentValue=0
            # print("Bin1")
            # print(value)
            # print("------------------")
            # print(bin_calulated_1)
            for i in range(len(value)):
                if value[i] in bin_calulated_1:
                    currentValue+=1
            # print(currentValue)
            if currentValue>=maxValue_1:
                maxValue_1=currentValue
                cluster=item
        print("Selected",cluster)
        print(maxValue_1)
        for i in range(len(bin_calulated_1)):
            features.append(bin_calulated_1[i])
            label_cluster.append(self.return_cluster(cluster))
        del self.groundTruthDictionary[cluster]
        print(self.groundTruthDictionary)
        for item,value in self.groundTruthDictionary.items():
            currentValue=0
            # print("Bin2")
            # print(value)
            # print("------------------")
            # print(bin_calulated_2)
            for i in range(len(value)):
                if value[i] in bin_calulated_2:
                    currentValue+=1
            # print(currentValue)
            if currentValue>=maxValue_2:
                maxValue_2=currentValue
                cluster=item
        print("Selected",cluster)
        print(maxValue_2)
        for i in range(len(bin_calulated_2)):
            features.append(bin_calulated_2[i])
            label_cluster.append(self.return_cluster(cluster))
        del self.groundTruthDictionary[cluster]
        print(self.groundTruthDictionary)
        for item,value in self.groundTruthDictionary.items():
            print(item)
            currentValue=0
            # print("Bin3")
            # print(value)
            # print("------------------")
            # print(bin_calulated_3)
            for i in range(len(value)):
                if value[i] in bin_calulated_3:
                    currentValue+=1
            if currentValue>=maxValue_3:
                maxValue_3=currentValue
                cluster=item
        print("Selected",cluster)
        print(maxValue_3)
        for i in range(len(bin_calulated_3)):
            features.append(bin_calulated_3[i])
            label_cluster.append(self.return_cluster(cluster))
        del self.groundTruthDictionary[cluster]
        print(self.groundTruthDictionary)
        for item,value in self.groundTruthDictionary.items():
            currentValue=0
            # print("Bin4")
            # print(value)
            # print("------------------")
            # print(bin_calulated_4)
            for i in range(len(value)):
                if value[i] in bin_calulated_4:
                    currentValue+=1
            # print(currentValue)
            if currentValue>=maxValue_4:
                maxValue_4=currentValue
                cluster=item
        print("Selected",cluster)
        print(maxValue_4)
        for i in range(len(bin_calulated_4)):
            features.append(bin_calulated_4[i])
            label_cluster.append(self.return_cluster(cluster))
        del self.groundTruthDictionary[cluster]
        for item,value in self.groundTruthDictionary.items():
            # print("Bin5")
            currentValue=0
            # print(value)
            # print("------------------")
            # print(bin_calulated_5)
            for i in range(len(value)):
                if value[i] in bin_calulated_5:
                    currentValue+=1
            # print(currentValue)
            if currentValue>=maxValue_5:
                maxValue_5=currentValue
                cluster=item
        print("Selected",cluster)
        print(maxValue_5)
        for i in range(len(bin_calulated_5)):
            features.append(bin_calulated_5[i])
            label_cluster.append(self.return_cluster(cluster))
        del self.groundTruthDictionary[cluster]
        print("Accuracy")
        print((maxValue_0+maxValue_1+maxValue_2+maxValue_4+maxValue_5))

        df=pd.DataFrame({
            "index":features,
            "label":label_cluster
        })
        finalPCA_features=[]
        final_label=[]
        print(len(df))
        for i in range(len(df)):
            finalPCA_features.append(self.finalPCADataFrame.loc[df.loc[i]['index']])
            final_label.append(df.loc[i]['label'])
        print(len(finalPCA_features))
        final_df_features=pd.DataFrame(finalPCA_features)
        final_label_df=pd.DataFrame(final_label)
        # print((final_df_features))
        # print(final_label_df)
        self.finalDataFrame=pd.concat([final_df_features,final_label_df], axis=1)
        df.to_csv("New.csv")
        self.finalDataFrame.to_csv("FinalDataFrame.csv")
        # # db_default = DBSCAN(eps = 0.375, min_samples = 5).fit(finalPCADataFrame) 
#create a matrix to check if the vl
s=DataSetFormation()
s.read_csv()
s.createFeatureMatrixCGM()
s.createGroundTruth()
#Creating a ground truth table of 6 clusters
#, >0 to 20, 21 to 40, 41 to 60, 61 t o 80, 81 to 100. 
mealFeatures=Features(4)
features=s.getFeatures()

mealPrincipalComponentDataFrame=s.normalizeData(features)
print(len(mealPrincipalComponentDataFrame))


s.createDBSCANClusterFromFeatures(mealPrincipalComponentDataFrame)
# s.createDBSCANClusterFromFeaturesMax()
# s.SSEMetrics()
# X_train, X_test, y_train, y_test = train_test_split(self.mealPrincipalComponentDataFrame,self.carbIntakeDataFrame,test_size=0.33, random_state=42)
# print(X_train)
# s.createKMeansCluster(mealPrincipalComponentDataFrame)
print(len(mealPrincipalComponentDataFrame))
# s.plotPointCluster()
# s.createDBSCANClusterFromFeaturesMax()
s.calculateAccuracy()
dataframe=pd.read_csv("FinalDataFrame.csv")
print(dataframe)
dataframe=dataframe.dropna()
train_data=dataframe[['pc1','pc2']]
train_label=dataframe['0']
print(train_label)

neighbour = KNeighborsClassifier(n_neighbors=3)
neighbour.fit(train_data, train_label)

y_label=neighbour.predict(train_data)
print(y_label)
filename = 'knn_model_db.pickle'
pickle.dump(neighbour, open(filename, 'wb'))





# s.createDBSCANClusterFromFeatures(mealPrincipalComponentDataFrame)

# # kmeans = KMeans(n_clusters=6, random_state=0).fit(finalPCADataFrame)

# # print(finalPCADataFrame.shape)
# # print(finalMealNoMealDataFrame['Label'].shape)
# # finalPCADataFrame['Label']=finalMealNoMealDataFrame['Label'].values
# # finalPCADataFrame+finalMealNoMealDataFrame['Label']





