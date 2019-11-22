import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#to split the given data into training and testing
#'pima.csv' file contains column labels as x1 to x8 for features and y for output
#Only x2, x3, x4, and y features are selected and split into X_train, X_test, y_train, and y_test as lists   
def SplitData():
    df = pd.read_csv("pima.csv")
    df=df.drop(columns=['x1','x5','x6','x7','x8'])
    X=df.drop(columns=['y']).values.tolist()
    y=df.drop(columns=['x2','x3','x4']).values.reshape(-1).tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50)
    return (X_train, X_test, y_train, y_test)

#The function calculates y_pred for predicted outputs
def KNNClassifier(X_train, X_test, y_train, y_test, K):
    classifier = KNeighborsClassifier(n_neighbors=K)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return(y_pred)

#calculates total correctly and incorrectly classified data by comparing y_pred and y_test lists
def findAccuracy(y_test, y_pred):
    correct=0
    wrong=0
    for i in range(len(y_pred)):
        if(y_test[i]==y_pred[i]):
            correct+=1
        else:
            wrong+=1
    return(correct, wrong)

#Iterate the classifier for different K values and for each value of K,
#it iterates the classifier for 10 time as required by question
listAccuracy=[]
for K in [1,5,11]:
    for i in range(10):
        X_train, X_test, y_train, y_test=SplitData()
        y_pred=KNNClassifier(X_train, X_test, y_train, y_test, K)
        correct, wrong= findAccuracy(y_test, y_pred)
        listAccuracy.append(float(correct)/(correct+wrong)*100)
        meanAccuracy=np.average(listAccuracy)
        sd=np.std(listAccuracy)
    print "\nDetails for K= ", K
    print "List of Accuracy for correct classification in percentage for 10 iteration\n", listAccuracy
    print "Mean accuracy= ", meanAccuracy
    print "Standard Deviation= ",sd 

#Ramdom OUTPUT
# Details for K= 1
# List of Accuracy for correct classification in percentage for 10 iteration
# [65.10416666666666, 64.84375, 65.625, 68.48958333333334, 63.541666666666664, 65.10416666666666, 65.10416666666666, 63.020833333333336, 65.10416666666666, 65.88541666666666]
# Mean accuracy= 65.18229166666666
# Standard Deviation= 1.3831533046759916

# Details for K= 5
# List of Accuracy for correct classification in percentage for 10 iteration
# [65.10416666666666, 64.84375, 65.625, 68.48958333333334, 63.541666666666664, 65.10416666666666, 65.10416666666666, 63.020833333333336, 65.10416666666666, 65.88541666666666, 69.01041666666666, 67.70833333333334, 72.65625, 69.79166666666666, 72.65625, 70.83333333333334, 67.96875, 71.61458333333334, 71.35416666666666, 71.09375]
# Mean accuracy= 67.82552083333334
# Standard Deviation= 3.061945235679779

# Details for K= 11
# List of Accuracy for correct classification in percentage for 10 iteration
# [65.10416666666666, 64.84375, 65.625, 68.48958333333334, 63.541666666666664, 65.10416666666666, 65.10416666666666, 63.020833333333336, 65.10416666666666, 65.88541666666666, 69.01041666666666, 67.70833333333334, 72.65625, 69.79166666666666, 72.65625, 70.83333333333334, 67.96875, 71.61458333333334, 71.35416666666666, 71.09375, 72.91666666666666, 71.09375, 71.875, 73.4375, 72.39583333333334, 71.09375, 70.83333333333334, 73.95833333333334, 73.95833333333334, 73.4375]
# Mean accuracy= 69.38368055555553
# Standard Deviation= 3.398534503546765

#'pima.csv' file outline
# x1,x2,x3,x4,x5,x6,x7,x8,y
# 6,148,72,35,0,33.6,0.627,50,1
# 1,85,66,29,0,26.6,0.351,31,0
# 8,183,64,0,0,23.3,0.672,32,1
# 1,89,66,23,94,28.1,0.167,21,0
# 0,137,40,35,168,43.1,2.288,33,1