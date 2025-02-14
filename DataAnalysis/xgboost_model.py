import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from xgboost.sklearn import XGBClassifier
import warnings

# Load the datasets into Pandas DataFrames
df_age_gender = pd.read_csv('age_gender_bkts.csv')  # Data containing age and gender information
df_sessions = pd.read_csv('sessions.csv')  # Data containing user sessions
df_test = pd.read_csv('test_users.csv')  # Data containing test user information
df_train = pd.read_csv('train_users_2.csv')  # Data containing training user information

# Merge the training and testing data (excluding the target variable 'country_destination' in the training set)
df_merge = pd.concat([df_train.drop('country_destination', axis=1), df_test], axis=0)
df_merge.reset_index(drop=True, inplace=True)  # Reset index after merging

# Drop the 'date_first_booking' column as it is not needed for the model
df_merge.drop('date_first_booking', axis=1, inplace=True)

# Correct age values greater than 500 by calculating the age as the difference between 2015 and the age value
age_index_1 = df_merge.age > 500
df_merge[age_index_1].age.describe()  # Check age statistics where age is greater than 500

# Update the 'age' column for the erroneous values
df_merge.loc[df_merge.age > 500, 'age'] = 2015 - df_merge.loc[age_index_1, 'age']

# Replace age values greater than 100 or less than 18 with NaN (invalid ages)
df_merge.loc[df_merge.age > 100, 'age'] = np.nan
df_merge.loc[df_merge.age < 18, 'age'] = np.nan

# Convert 'date_account_created' and 'timestamp_first_active' to datetime
df_merge['date_account_created'] = pd.to_datetime(df_merge['date_account_created'])
df_merge['timestamp_first_active'] = pd.to_datetime(df_merge['timestamp_first_active'])

# Extract features from the 'date_account_created' and 'timestamp_first_active' columns
df_merge['weekday_account_created'] = df_merge.date_account_created.dt.strftime("%w")
df_merge['day_account_created'] = df_merge.date_account_created.dt.day
df_merge['month_account_created'] = df_merge.date_account_created.dt.month
df_merge['year_account_created'] = df_merge.date_account_created.dt.year

df_merge['weekday_first_active'] = df_merge.timestamp_first_active.dt.strftime("%w")
df_merge['day_first_active'] = df_merge.timestamp_first_active.dt.day
df_merge['month_first_active'] = df_merge.timestamp_first_active.dt.month
df_merge['year_first_active'] = df_merge.timestamp_first_active.dt.year

# Calculate the 'time_lag' feature: the difference in days between 'date_account_created' and 'timestamp_first_active'
df_merge['time_lag'] = (df_merge['date_account_created'] - df_merge['timestamp_first_active'])
df_merge['time_lag'] = df_merge['time_lag'].apply(lambda x: x.days)

# Drop the original 'date_account_created' and 'timestamp_first_active' columns after feature extraction
df_merge.drop(['date_account_created', 'timestamp_first_active'], axis=1, inplace=True)

# Rename the 'user_id' column in df_sessions to 'id' to match the user identifier in df_merge
df_sessions.rename(columns={'user_id': 'id'}, inplace=True)

# Create session-related features by aggregating session data
# Count occurrences of each action type, action, and action_detail per user (id)
action_count = df_sessions.groupby(['id', 'action'])['secs_elapsed'].agg(len).unstack()
action_type_count = df_sessions.groupby(['id', 'action_type'])['secs_elapsed'].agg(len).unstack()
action_detail_count = df_sessions.groupby(['id', 'action_detail'])['secs_elapsed'].agg(len).unstack()
device_type_sum = df_sessions.groupby(['id', 'device_type'])['secs_elapsed'].agg(sum).unstack()

# Concatenate the action-related features into a single DataFrame
df_sessions_action = pd.concat([action_count, action_type_count, action_detail_count, device_type_sum], axis=1)

# Rename columns to indicate they represent counts
df_sessions_action.columns = df_sessions_action.columns.map(lambda x: str(x) + '_count')

# Add a column to track the most used device for each user (maximum value of device_type per user)
df_sessions_action['most_used_device'] = df_sessions.groupby('id')['device_type'].max()

# Reset index of df_sessions_action for proper merging
df_sessions_action.reset_index(inplace=True)
df_sessions_action.rename(columns={'index': 'id'}, inplace=True)

# Aggregate session time (secs_elapsed) statistics for each user
secs_elapsed = df_sessions.groupby('id')['secs_elapsed']
secs_elapsed = secs_elapsed.agg(
    [('secs_elapsed_sum', np.sum),  # Sum of session durations
    ('secs_elapsed_mean', np.mean),  # Mean session duration
    ('secs_elapsed_min', np.min),    # Minimum session duration
    ('secs_elapsed_max', np.max),    # Maximum session duration
    ('secs_elapsed_median', np.median),  # Median session duration
    ('secs_elapsed_std', np.std),    # Standard deviation of session durations
    ('secs_elapsed_var', np.var),    # Variance of session durations
    ('day_pauses', lambda x: (x > 86400).sum()),  # Count of pauses longer than a day
    ('long_pauses', lambda x: (x > 300000).sum()),  # Count of pauses longer than 5,000 minutes
    ('short_pauses', lambda x: (x < 3600).sum()),  # Count of short pauses less than an hour
    ('session_length', np.count_nonzero)]  # Count of non-zero session durations
)

# Reset the index for secs_elapsed after aggregation
secs_elapsed.reset_index(inplace=True)

# Merge the session statistics with the action counts
sessions_secs_elapsed = pd.merge(df_sessions_action, secs_elapsed, on='id', how='left')

# Merge the session-related data with the user data in df_merge
df_merge = pd.merge(df_merge, sessions_secs_elapsed, on='id', how='left')

# Drop the 'id' column after merging since it's no longer needed
df_merge.drop('id', axis=1, inplace=True)

# Remove any duplicate columns that may have resulted from the merge
duplicate_columns = df_merge.columns[df_merge.columns.duplicated()]
df_merge = df_merge.loc[:, ~df_merge.columns.duplicated()]

# Define categorical features that need to be one-hot encoded
categorical_features = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser', 'most_used_device', 'weekday_account_created', 'weekday_first_active']

# Perform one-hot encoding for categorical variables
df_merge = pd.get_dummies(df_merge, columns=categorical_features)

# Encode the target variable 'country_destination' using LabelEncoder
le = LabelEncoder()
y = df_train['country_destination']
y = le.fit_transform(y)

# Extract the test set id (idxmax is incorrect, should be replaced by the proper indexing)
id_test = df_test.id

# Split the merged data into training and testing sets
train_num = df_train.shape[0]
X = df_merge[:train_num]  # Training features
X_sub = df_merge[train_num:]  # Testing features (for predictions)

# Suppress warnings (usually to ignore performance warnings from the classifier)
warnings.filterwarnings('ignore')

# Initialize the XGBoost classifier with a multi-class classification objective
xgb = XGBClassifier(
    objective='multi:softprob',  # Multi-class classification
    eval_metric='mlogloss',  # Logarithmic loss evaluation metric
    colsample_bytree=0.5,  # Subsample ratio of columns
    max_depth=6,  # Maximum depth of the decision tree
    learning_rate=0.1,  # Learning rate
    subsample=0.5  # Subsample ratio of rows
)

# Train the model on the training data
xgb.fit(X, y)

# Generate class probabilities for the test set
y_pred = xgb.predict_proba(X_sub)

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
sub.to_csv('submission.csv', index=False)