import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import pickle
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import GridSearchCV
# def applyPCA(normalized_features,number):
#         pca = PCA(n_components=number)
#         pComponents= pca.fit_transform(normalized_features.values)
#         pComponentsDataFrame = pd.DataFrame(data = pComponents
#              , columns = ['pc1', 'pc2','pc3'])
#         # print(pComponentsDataFrame)
#         print((pca.components_ ))
#         print(pca.explained_variance_ratio_)
#         return pComponentsDataFrame

data = pd.read_csv("finalMealNoMeal.csv")
# columns=['fft1','fft3','velocity1','velocity2','velocity3','velocity4',
#     'rolling1','rolling2','rolling3','rolling4',
#         'dwt1','dwt2','dwt3','dwt4']

# columns=['velocity1','velocity2','velocity3','velocity4']
# columns=['dwt1','dwt2','dwt3','dwt4']
# columns=['fft1','fft2','fft3','fft4']
# columns=['rolling1','rolling2']
columns=['velocity1','velocity2','rolling2','rolling1']
# columns = ['pc1', 'pc2','pc3']
X_Data=data[columns]
# PCA_Data=applyPCA(X_Data,3)
# print(PCA_Data)
Y=data["Label"]
# print(Y)
X_train, X_test, y_train, y_test = train_test_split(X_Data, Y,test_size=0.2,random_state=43)
clf = RandomForestClassifier(min_samples_split=2,n_estimators=100, criterion='entropy', min_samples_leaf=1)
# clf = XGBClassifier()
# parameters =[{'nthread': [4],'objective': ['binary:logistic'],
# 'learning_rate': [0.05],
# 'max_depth': [6],
# 'min_child_weight': [11],
# 'silent': [1],
# 'subsample': [0.8],
# 'colsample_bytree': [0.7],
# 'n_estimators': [100],
# 'missing': [-999],
# 'seed': [1337]}]

# g_search = GridSearchCV(clf, parameters,scoring='accuracy',cv=10,n_jobs=-1)
# clf = RandomForestClassifier(min_samples_split=12,n_estimators=50, criterion='gini', min_samples_leaf=5)
# clf = svm.SVC()
# parameters=[{'C':[1,10,100,1000,10000],'kernel':['linear']},
# {'C':[1,10,100,1000,10000],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]
# g_search=GridSearchCV(estimator=clf,param_grid=parameters,
# scoring='accuracy',cv=10,n_jobs=-1)
# g_search.fit(X_train, y_train)
# accuracy=g_search.best_score_
# print(accuracy)
# print(g_search.best_params_)
# clf = svm.SVC(kernel='rbf',gamma=0.1,C=1000)#for 3 PC
#final Model
# clf = svm.SVC(kernel='rbf',gamma=0.1,C=10)
# clf = svm.SVC(kernel='rbf',gamma=0.6,C=8000)
# clf = svm.SVC(kernel='rbf',gamma=0.3,C=10000)
# clf = svm.SVC(kernel='rbf',gamma=0.7,C=10000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
filename = 'RandomForestFinal.pickle'
pickle.dump(clf, open(filename, 'wb'))
print(y_pred)
print("-----")
print(y_test.values)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# from sklearn.ensemble import RandomForestClassifier

# rf = RandomForestClassifier(random_state=43, n_jobs=-1)

# param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10], "min_samples_split" : [2, 4, 10, 12, 16], "n_estimators": [50, 100, 400, 700, 1000]}

# gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)

# gs = gs.fit(X_train, y_train)

# accuracy=gs.best_score_
# print(accuracy)
# print(gs.best_params_)
# XV_train, XV_test, yv_train, yv_test = train_test_split(X_train, y_train,test_size=0.2, random_state=43)
# print(XV_train)
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import SGD
# from keras.layers import Dropout
# model = Sequential()
# model.add(Dense(1000, input_dim=14, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(500 ,activation='relu'))
# model.add(Dense(250, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# sgd = SGD(lr=0.02, momentum=0.8)
# model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
# model.fit(XV_train, yv_train, epochs=1000, batch_size=50,validation_data=(XV_test, yv_test))
# _, accuracy = model.evaluate(X_test, y_test)
# print('Accuracy: %.2f' % (accuracy*100))
# filename = 'LogisticReg.pickle'
# pickle.dump(model, open(filename, 'wb'))