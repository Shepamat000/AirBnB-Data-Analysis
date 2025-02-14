import pandas as pd
import numpy as np
from datetime import datetime

dfTrain = pd.read_csv ("dataset/trainUsers.csv", skipinitialspace=True)
dfTest = pd.read_csv ("dataset/Trees/testUsers.csv", skipinitialspace=True)

dfTrain["gender"] = dfTrain["gender"].str.replace("-unknown-","")
dfTest["gender"] = dfTest["gender"].str.replace("-unknown-","")

#dfTrain = dfTrain ((dfTrain.values != "-unknown-").all(axis=1))

# Remove continuous attributes
dfTrain = dfTrain.drop(['date_first_booking'], axis=1)
dfTest = dfTest.drop(['date_first_booking'], axis=1)
dfTrain = dfTrain.drop(['date_account_created'], axis=1)
dfTest = dfTest.drop(['date_account_created'], axis=1)

def numericalBinary (dataset, features):
    dataset[features] = np.where(dataset[features] >= dataset[features].mean(), 1, 0)


def dateToInt (dataset, features):
    dataset[features[0]] = dataset[features[0]].apply(
        lambda x: int(datetime.strptime(x, "%Y-%m-%d").timestamp()))
    dataset[features] = np.where(dataset[features]  >= dataset[features].mean(), 1, 0)


# Remove continuous attributes
numericalBinary(dfTrain, ['age'])
numericalBinary(dfTest, ['age'])
numericalBinary(dfTrain, ['timestamp_first_active'])
numericalBinary(dfTest, ['timestamp_first_active'])


# One-hot encoder
def oneHotBind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[feature_to_encode])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop(feature_to_encode, axis=1)
    return(res)





dfTrain = oneHotBind(dfTrain,['age', 'timestamp_first_active', 'gender','signup_method', 'signup_flow','language','affiliate_channel','affiliate_provider','first_affiliate_tracked','signup_app','first_device_type','first_browser'])
dfTest = oneHotBind(dfTest,['age', 'timestamp_first_active','gender','signup_method', 'signup_flow','language','affiliate_channel','affiliate_provider','first_affiliate_tracked','signup_app','first_device_type','first_browser'])

# Add missing attributes
for attributes in dfTrain.keys():
    if attributes not in dfTest.keys():
        print("Adding missing feature {}".format(attributes))
        dfTest[attributes] = 0


# Add missing attributes
for attributes in dfTest.keys():
    if attributes not in dfTrain.keys():
        print("Adding missing feature {}".format(attributes))
        dfTrain[attributes] = 0

print(dfTrain)
X_train,Y_train = dfTrain.iloc[:, 2:].values, dfTrain.iloc[:, 1].values
X_test,Y_test = dfTest.iloc[:, 2:].values, dfTest.iloc[:, 1].values

# Decision Tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

tree = DecisionTreeClassifier()
tree.fit(X_train, Y_train)
predictions = tree.predict(X_test)


print("=======================================================")
print("Decision Tree Model:")
#This will not work, as we do not have test samples on Y
#print("Accuracy: " + str(accuracy_score(Y_test, predictions)))
#print(classification_report(Y_test, predictions))


id_test = dfTest.id
le = LabelEncoder()
y = dfTrain['country_destination']
y = le.fit_transform(y)


y_pred = tree.predict_proba(X_test)

# Prepare the final submission (top 5 predicted classes for each test sample)
ids = []
countries = []
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5  # Duplicate the id 5 times for the top 5 countries
    countries += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()  # Get top 5 countries by predicted probability

# Create a DataFrame for submission
sub = pd.DataFrame(np.column_stack((ids, countries)), columns=['id', 'country'])

# Save the submission DataFrame to a CSV file
sub.to_csv('src/trees/output/DecisionOutput.csv', index=False)


# Naive Bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, Y_train)

X_test = X_test[(X_test != "-unknown-").all(axis=1)]

print("=======================================================")
print("Naive Bayes Model:")
#This will not work, as we do not have test samples on Y
#print("Accuracy: " + str(accuracy_score(Y_test, predictions)))
#print(classification_report(Y_test, predictions))

# Print to output file
i = 0


y_pred = gnb.predict_proba(X_test)

# Prepare the final submission (top 5 predicted classes for each test sample)
ids = []
countries = []
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5  # Duplicate the id 5 times for the top 5 countries
    countries += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()  # Get top 5 countries by predicted probability

# Create a DataFrame for submission
sub = pd.DataFrame(np.column_stack((ids, countries)), columns=['id', 'country'])

# Save the submission DataFrame to a CSV file
sub.to_csv('src/trees/output/BayesianOutput.csv', index=False)

