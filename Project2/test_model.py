import pickle
from feature_extraction import Features
from dataset_formation import DataSetFormation
import pandas as pd
filename='SVM_final.pickle'
loaded_model = pickle.load(open(filename, 'rb'))
columns=["c1","c2","c3","c4","c5","c6","c7","c8","c9","c10","c11","c12","c13","c14","c15"
        ,"c16","c17","c18","c19","c20","c21","c22","c23","c24","c25","c26","c27","c28","c29","c30"]
test_dataframe=pd.read_csv('MealNoMealData/mealData3.csv',names=columns)
# print(test_dataframe)
row,column=test_dataframe.shape
for i in range(row):    
    test_dataframe.dropna(thresh=4,axis = 0)
print("test_data")
# print(test_dataframe)
test_dataframe=test_dataframe.interpolate(method ='linear', limit_direction ='backward')
print(test_dataframe)
# test_dataframe=test_dataframe.dropna()

# print(test_dataframe)
s=DataSetFormation()
f=Features(4)
data=f.completefeatures(test_dataframe)
data=normalized_data=s.normalizeData(data)
# data=s.applyPCA(data,3)
data["Label"]=1
print(data)
column=['fft1','fft2','fft3','fft4','velocity1','velocity2','velocity3','velocity4',
        'rolling1','rolling2','rolling3','rolling4',
        'dwt1','dwt2','dwt3','dwt4']
column_p=['pc1','pc2','pc3']
column_v=['velocity1','velocity2','rolling2','rolling1']
value=loaded_model.predict(data[column_v])
print(value)
result = loaded_model.score(data[column_v], data['Label'])
print(result)
