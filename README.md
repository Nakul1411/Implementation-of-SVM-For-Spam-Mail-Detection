# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python packages using import statements.
2. Read the given csv file using read_csv() method and print the number of contents to be dislayed using df.head().
3. Split the dataset using train_test_split.
4.Calculate Y_pred and accuracy .
5. Print all the outputs.
6. End the program.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: NAKUL R
RegisterNumber: 212223240102
*/
```
import pandas as pd
data= pd.read_csv("C:/Users/admin/Desktop/INTR MACH/spam.csv", encoding= 'Windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test , y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test= cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train , y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy= metrics.accuracy_score(y_test, y_pred)
accuracy

from sklearn.metrics import confusion_matrix, classification_report
con= confusion_matrix(y_test, y_pred)
print(con)cl=classification_report(y_test,y_pred)
print(cl)

## Output:
![Screenshot 2024-11-11 111813](https://github.com/user-attachments/assets/1d9e109b-69c3-4d38-994e-bec5fd76bf92)
![Screenshot 2024-11-11 111821](https://github.com/user-attachments/assets/f185bbde-db91-4f14-802b-db055ce8872a)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
