import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import seaborn as sb
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
data=pd.read_csv('churn_data.csv')
#Cleaning data
newdata=data.drop(['gender','OnlineSecurity','MultipleLines','OnlineBackup','DeviceProtection','Dependents','PaymentMethod','MultipleLines','InternetService','TechSupport','StreamingTV','StreamingMovies','customerID'],axis=1)

newdata['TotalCharges']= pd.to_numeric(newdata.TotalCharges, errors='coerce')
#print(newdata)
newdata.isnull().sum()
newdata = newdata.interpolate()
#newdata.info()

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
newdata_1hot = encoder.fit_transform(newdata['Churn'])
#print(newdata_1hot)
newdata_1hot1 = encoder.fit_transform(newdata['PaperlessBilling'])
#print(newdata_1hot1)
newdata_1hot2 = encoder.fit_transform(newdata['Partner'])
newdata_1hot3 = encoder.fit_transform(newdata['PhoneService'])

newdata_1hot4 = encoder.fit_transform(newdata['PhoneService'])

categorical_cols = ['Partner', 'PhoneService', 'Contract','PaperlessBilling']
l_encoder = LabelEncoder()
# apply le on categorical feature columns
newdata[categorical_cols] = newdata[categorical_cols].apply(lambda col: l_encoder.fit_transform(col))
ohe = OneHotEncoder()
#One-hot-encode the categorical columns.
#Unfortunately outputs an array instead of dataframe.
array_hot_encoded = ohe.fit_transform(newdata[categorical_cols])
#Convert it to df
data_hot_encoded = pd.DataFrame(array_hot_encoded, index=newdata.index)
#Extract only the columns that didnt need to be encoded
data_other_cols = newdata.drop(columns=categorical_cols)
#Concatenate the two dataframes :
data_out = pd.concat([data_hot_encoded, data_other_cols], axis=1)
#print(data_out)
data_out['Churn'] = encoder.fit_transform(newdata['Churn'])
#data_out.info()
#print(data_out)

corr_matrix=data_out.corr()
#print(corr_matrix)
#corr_matrix['Churn'].sort_values(ascending=False)

sb.heatmap(corr_matrix)

X = newdata.iloc[:,:-1]
Y = newdata.iloc[:,-1]
#data_dmatrix = xgb.DMatrix(data=X,label=Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=25)
model = XGBClassifier()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
predictions = [round(value) for value in Y_pred]
accuracy = accuracy_score(Y_test,predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
#saving model
import pickle
pickle_out = open("classifier.pkl", mode = "wb")
pickle.dump(model, pickle_out)
pickle_out.close()

model_xg = XGBClassifier().fit(X_train, Y_train)
probs_xg = model_xg.predict_proba(X_test)[:, 1]

Y_test_int = Y_test.replace({'Yes': 1,'No':0})

auc_xg = roc_auc_score(Y_test_int, probs_xg)
fpr_xg, tpr_xg, thresholds_xg = roc_curve(Y_test_int, probs_xg)

plt.figure(figsize=(12, 7))
plt.plot(fpr_xg, tpr_xg, label=f'AUC (XGBoost) = {auc_xg:.2f}')
plt.xlabel('churned', size=10)
plt.ylabel('not churned', size=10)
