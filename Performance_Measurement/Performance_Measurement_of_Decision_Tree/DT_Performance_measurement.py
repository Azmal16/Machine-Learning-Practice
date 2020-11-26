from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
# GitHub URL of the dataset
#url = "https://raw.githubusercontent.com/shashankvmaiya/Height-Weight-Gender-Classification/master/data/01_heights_weights_genders.csv"
dataset = pd.read_csv(
    "/Users/azmalawsaf/Desktop/1603018_Azmal_Awasaf/01_heights_weights_genders.csv")
print(dataset)
print(dataset.shape)
Y = dataset.iloc[:, :1].values
X = dataset.iloc[:, 1:3].values
print(X)
print(Y)
# Converting Gender to number
# 1 == Male
# 0 == Female
labelEncoder_gender = LabelEncoder()
Y = labelEncoder_gender.fit_transform(Y)
# Splitting data into Training and Testing Set:
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)
# Fitting the Data into DT classifier model:
DT = DecisionTreeClassifier(random_state=0)
DT.fit(X_train, Y_train)
# Prediction:
Y_pred = DT.predict(X_test)
# Confusion Matrix
cm = metrics.confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:", cm)
accuracy = ((1091 + 1088)/(1091+146+175+1088))
print(accuracy)
precision = (1088/(1088+146))
print(precision)
recall = (1088/(1088+175))
print(recall)
f1 = 2*(1/((1/precision)+(1/recall)))
print(f1)
