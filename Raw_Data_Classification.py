import pandas as pd 
import numpy as np
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 

import matplotlib as mpl
import matplotlib.pyplot as plt

#Merging data across sheets
final = pd.read_csv('1_1.csv')  
#Start with the initial file the function will add all others
final.drop('time', axis=1, inplace=True)
final.rename(columns={'class':'Class'}, inplace=True)

#Function: Append data from new sheet 
def add_sheet_data(filename, final):
    sheet = pd.read_csv(filename)
    sheet.drop('time', axis=1, inplace=True)
    sheet.rename(columns={'class':'Class'}, inplace=True)
    new = pd.concat([final,sheet])
    final =new
    return final

final = add_sheet_data('2_1.csv',final) # Function calls like these will add new sheet to final.

#Remove unmarked data and create taining and testing sets
x = final.sort_values(['Class'],ascending =True, axis=0)
x = x[x.Class != 0]
print(x)
y = x.Class
print(y)
x.drop('Class',axis=1,inplace =True)

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 0) 

#Start Classification from here
# training a KNN classifier 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 6).fit(X_train, y_train) 
  
# accuracy on X_test 
accuracy = knn.score(X_test, y_test) 
print (accuracy )
  
# creating a confusion matrix 
knn_predictions = knn.predict(X_test)  
cm = confusion_matrix(y_test, knn_predictions) 
#print(cm)

###################################################################################

from sklearn.svm import SVC 
# training a linear SVM classifier 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 

# model accuracy for X_test 
accuracy = svm_model_linear.score(X_test, y_test) 
print(accuracy)

#creating a confusion matrix 
cm = confusion_matrix(y_test, svm_predictions) 
#print(cm)

####################################################################################

from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(X_train, y_train) 
gnb_predictions = gnb.predict(X_test) 
  
# accuracy on X_test 
accuracy = gnb.score(X_test, y_test) 
print (accuracy) 
  
# # creating a confusion matrix 
# cm = confusion_matrix(y_test, gnb_predictions) 
# print(cm)