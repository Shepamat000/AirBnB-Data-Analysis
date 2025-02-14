import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Data Preprocessing Q1
# 1. Remove records with unknown (?) values from both train and test data sets and remove all continuous attributes.
# Read file and remove whitespace
dfTrain = pd.read_csv('./data/train_users_2.csv',skipinitialspace=True)
dfTest = pd.read_csv('./data/test_users.csv',skipinitialspace=True)

dfTrain = dfTrain[(dfTrain.values !="-unknown-").all(axis=1)]

dfTrain = dfTrain.drop(['date_account_created','timestamp_first_active','date_first_booking','age','signup_flow'], axis=1)
dfTest = dfTest.drop(['date_account_created','timestamp_first_active','date_first_booking','age','signup_flow'], axis=1)



# One-hot encoder
def oneHotBind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[feature_to_encode])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop(feature_to_encode, axis=1)
    return(res)

dfTrain = oneHotBind(dfTrain,['gender','signup_method','language','affiliate_channel','affiliate_provider','first_affiliate_tracked','signup_app','first_device_type','first_browser'])
dfTest  = oneHotBind(dfTest, ['gender','signup_method','language','affiliate_channel','affiliate_provider','first_affiliate_tracked','signup_app','first_device_type','first_browser'])

# Add missing attributes


for attributes in dfTrain.keys():
    if attributes not in dfTest.keys():
        print("Adding missing feature {}".format(attributes))
        dfTest[attributes] = 0

for attributes in dfTest.keys():
    if attributes not in dfTrain.keys():
        print("Adding missing feature {}".format(attributes))
        dfTrain[attributes] = 0

print(dfTrain)
X_train,Y_train = dfTrain.iloc[:, 2:].values, dfTrain.iloc[:, 1].values
X_test,Y_test = dfTest.iloc[:, 2:].values, dfTest.iloc[:, 1].values

#Logistic Regression
from sklearn.linear_model import LogisticRegression

le = LabelEncoder()
y = dfTrain['country_destination']
le.fit_transform(y)

id_test = dfTest.id

logreg = LogisticRegression(max_iter=1)
logreg.fit(X_train, Y_train)
prediction = logreg.predict_proba(X_test)


ids = []
countries = []
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5  # Duplicate the id 5 times for the top 5 countries
    countries += le.inverse_transform(np.argsort(prediction[i])[::-1])[:5].tolist()  # Get top 5 countries by predicted probability

# Create a DataFrame for submission
sub = pd.DataFrame(np.column_stack((ids, countries)), columns=['id', 'country'])

sub.to_csv('./data/LogisticRegression.csv', index=False)