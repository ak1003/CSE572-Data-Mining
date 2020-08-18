import pickle
from feature_extraction import Features
from dataset_formation import DataSetFormation
import pandas as pd
import os
columns=["c1","c2","c3","c4","c5","c6","c7","c8","c9","c10","c11","c12","c13","c14","c15"
        ,"c16","c17","c18","c19","c20","c21","c22","c23","c24","c25","c26","c27","c28","c29","c30"]
test_dataframe=pd.read_csv("proj3_test.csv",names=columns)
# print(test_dataframe.head())
row,column=test_dataframe.shape
# print(row)
s=DataSetFormation()
f=Features(4)
data=f.completefeatures(test_dataframe)
data=normalized_data=s.normalizeData(data)
data=s.applyPCA(data,2)
loaded_model = pickle.load(open("knn_model_db.pickle", 'rb'))
db_scan=loaded_model.predict(data)
print(db_scan)
loaded_model = pickle.load(open("knn_model_km.pickle", 'rb'))
km_scan=loaded_model.predict(data)
print(km_scan)
df=pd.DataFrame({
    "DBScan":db_scan,
    "Kmeans":km_scan
})

df.to_csv("labels_cluster.csv",header=None)
