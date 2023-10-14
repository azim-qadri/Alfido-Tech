# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# %%
df = pd.read_csv("tested.csv")
df

# %%
# Drop the column since it has most values missing
df['Cabin'].isna().sum()
# PassengerId, Name are not useful so drop those columns
df = df.drop(['PassengerId', 'Name','Cabin','Ticket'], axis=1)
df

# %%
# Preprocessing for missing values
# print count of missing values in all the columns

for i in df:
    print(i,':',df[i].isnull().sum())

# To help fill the missing value in Fare we will try to figure out some relations
df["Pclass"].value_counts()

# Heatmap to find out the missing value in fare which depends on pclass sex age and embarked
sns.barplot(x="Pclass", y="Fare", data=df)
plt.show()
sns.barplot(x="Age", y="Fare", data=df)
plt.show()
sns.barplot(x="Sex", y="Fare", data=df)
plt.show()
sns.barplot(x="Embarked", y="Fare", data=df)

# Display the plot
plt.show()


# %%
# This shows there is good amount of relation between fare and pclass,sex,embarked
# Use the average fare based on the pclass, sex,embarked for each group
df_grouped = df.groupby(['Pclass', 'Sex', 'Embarked'])['Fare'].mean().reset_index()


df = pd.merge(df, df_grouped, on=['Pclass', 'Sex', 'Embarked'], how='left', suffixes=('', '_average'))
df['Fare'].fillna(df['Fare_average'], inplace=True)
df.drop(columns=['Fare_average'], inplace=True)
df




# %%
print(df.iloc[:][152:153])

# %%
# checking the estimated value lies in the range of its group
filtered_data = df[(df['Pclass'] == 3) & (df['Sex'] == 'male') & (df['Embarked'] == 'S')&(df['Age']>45)]

# Display the filtered data
print(filtered_data)

# %%
# print the records where age is missing to analyze the data
missing_age_data = df[(df['Age'].isna())]
print(missing_age_data)

# %%
correlation = df[['Age', 'SibSp', 'Parch']].corr()
print(correlation)
plt.figure(figsize=(10,8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.show()

# %%
"""
The Heatmap shows that there is no strong relationship between Age,Sibsp and Parch
"""

# %%
df.boxplot()
df.describe()

# %%
# Remove the outlier
df = df[df['Fare'] < 500]
df.boxplot()
df.describe()

# %%
# We will use two methods to fill the missing data for age simple imputer and deep learning method using data twig then find out which performs better


# %%
corr = df.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# %%
df

# %%
import numpy as np
from sklearn.impute import SimpleImputer

def fill_missing_values(df):
    # Create a SimpleImputer object
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    # Fill missing values
    df["Age"] = imputer.fit_transform(df[["Age"]])

    return df


# %%
df_simple_imp = df.copy()
fill_missing_values(df_simple_imp)

df_simple_imp

# %%
df_mlp = df.copy()
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Convert 'Sex' column to numerical values
df_mlp['Sex'] = df_mlp['Sex'].map({'male': 0, 'female': 1})

# One-hot encode the 'Embarked' column
df_mlp = pd.get_dummies(df_mlp, columns=['Embarked'], drop_first=True)

# Split data into rows with missing Age and rows without
missing_age_data = df_mlp[df_mlp['Age'].isnull()]
known_age_data = df_mlp.dropna(subset=['Age'])

# Separate features and target for known_age_data
X = known_age_data.drop('Age', axis=1)
y = known_age_data['Age']

# MICE imputation for other columns (excluding Age)
imputer = IterativeImputer(max_iter=10, random_state=42, skip_complete=True)
X_imputed = imputer.fit_transform(X)
X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)

# Splitting the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_imputed_df, y, test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the MLP regressor model
regressor = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=500, alpha=0.0001,
                         solver='adam', verbose=10, random_state=21, tol=0.000000001)

# Train the model
regressor.fit(X_train, y_train)

# Predict on test set
y_pred = regressor.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")



# %%
"""
The simple MLP Regressor doesnt give a good performace with rmse = 140
"""

# %%
df_knn_impute = df.copy()
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Convert 'Sex' column to numerical values
df_knn_impute['Sex'] = df_knn_impute['Sex'].map({'male': 0, 'female': 1})

# One-hot encode the 'Embarked' column
df_knn_impute = pd.get_dummies(df_knn_impute, columns=['Embarked'], drop_first=True)

# Separate rows with and without missing Age values
missing_age_data = df_knn_impute[df_knn_impute['Age'].isnull()]
known_age_data = df_knn_impute.dropna(subset=['Age'])

# Separate features for known_age_data (keep 'Age' for now)
X_known = known_age_data
y_known = known_age_data['Age']

# Splitting the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_known, y_known, test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
X_train_imputed = knn_imputer.fit_transform(X_train_scaled)
X_test_imputed = knn_imputer.transform(X_test_scaled)

# Extract the imputed Age values for the test set
imputed_ages = X_test_imputed[:, X_train.columns.get_loc("Age")]

# Calculate the RMSE for the imputed ages on the test set
mse = mean_squared_error(y_test, imputed_ages)
rmse = mse**0.5
print(f"Root Mean Squared Error: {rmse:.4f}")




# %%
"""
Since rmse is 29 its not good for the model performance
"""

# %%
"""
No we will consider two cases either by dropping the age column and using simple imputer

"""

# %%
print(df_simple_imp)
print(df)


# %%
"""
Using mean imputation for age
"""

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import pandas as pd

def evaluate_models(X, y):
    # Encode 'Sex' column
    X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
    
    # One-hot encode the 'Embarked' column
    X = pd.get_dummies(X, columns=['Embarked'], drop_first=True)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Define the models
    models = {
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(probability=True),
        'Logistic Regression': LogisticRegression(max_iter=1000)
    }
    
    # Compute accuracy scores and RMSE for each model
    accuracy_scores = {}
    rmse_scores = {}
    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores[model_name] = accuracy
        
        # RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_scores[model_name] = rmse

    # Print the scores for each model
    for model_name in models.keys():
        print(f"{model_name} - Accuracy: {accuracy_scores[model_name]:.4f}, RMSE: {rmse_scores[model_name]:.4f}")
    
    return accuracy_scores, rmse_scores

X, y = df_simple_imp.drop("Survived",1),df["Survived"]
accuracy_results, rmse_results = evaluate_models(X, y)


# %%
# using drop the age column completely
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import confusion_matrix 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import pandas as pd

def evaluate_models(X, y):
    # Encode 'Sex' column
    X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
    
    # One-hot encode the 'Embarked' column
    X = pd.get_dummies(X, columns=['Embarked'], drop_first=True)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Define the models
    models = {
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(probability=True),
        'Logistic Regression': LogisticRegression(max_iter=1000)
    }
    
    # Compute accuracy scores and RMSE for each model
    accuracy_scores = {}
    rmse_scores = {}
    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores[model_name] = accuracy
        
        # RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_scores[model_name] = rmse

    # Print the scores for each model
    for model_name in models.keys():
        print(f"{model_name} - Accuracy: {accuracy_scores[model_name]:.4f}, RMSE: {rmse_scores[model_name]:.4f}")
    
    
    for k,v in models.items():
       cm = confusion_matrix(y_test, v.predict(X_test)) 
       #extracting TN, FP, FN, TP
       TN, FP, FN, TP = confusion_matrix(y_test, v.predict(X_test)).ravel()
       print(cm)
       print('Model[{}] Testing Accuracy = "{} !"'.format(k,  (TP + TN) / (TP + TN + FN + FP)))
       print()# Print a new line
    
    return accuracy_scores, rmse_scores

X, y = df.drop(["Survived","Age"],1),df["Survived"]
accuracy_results, rmse_results = evaluate_models(X, y)


# %%
"""
Since, training accuracy and testing accuracy for Random Forest classifier is 100%. We will use this model
"""

# %%
df_simple_imp["Embarked"].value_counts()

# %%
"""
So, from the above its clear that we can use any of Random Forest or Logistic Regression on any of the two datasets so as to predict the output
However further we will go with Random forest method as its primarily used for classification
"""

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score


def preprocess_data(df1):
    # Encode 'Sex' column
    df1['Sex'] = df1['Sex'].map({'male': 0, 'female': 1})
    
    # One-hot encode the 'Embarked' column
    df1 = pd.get_dummies(df1, columns=['Embarked'], prefix='Embarked')
    
    
    return df1

# Split the data
df_simple_imp['Embarked'] = pd.Categorical(df_simple_imp['Embarked'], categories=['Q', 'S', 'C'])
df_simple_imp = preprocess_data(df_simple_imp)
X = df_simple_imp.drop('Survived', axis=1)
y = df_simple_imp['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Random Forest model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
# Make predictions on the test set
y_pred = clf.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")





# %%
# Predict survival based on input data
def predict_survival(input_data):
    # Preprocess the input data
    input_df = pd.DataFrame([input_data],columns=df.drop("Survived",1).columns)

    input_df['Sex'] = input_df['Sex'].map({'male': 0, 'female': 1})
    # One-hot encode the 'Embarked' column
    input_df['Embarked'] = pd.Categorical(input_df['Embarked'], categories=['Q', 'S', 'C'])
    input_df = pd.get_dummies(input_df, columns=['Embarked'], prefix='Embarked')
    print(input_df)
    # Scale the input data
    input_scaled = scaler.transform(input_df)
    
    # Predict using the trained model
    prediction = clf.predict(input_scaled)
    return "Survived" if prediction[0] == 1 else "Did not survive"

# Usage
input_data = {
    'Pclass': 3,
    'Sex': 'female',
    'Age': 45.0,
    'SibSp': 1,
    'Parch': 0,
    'Fare': 7.25,
    'Embarked': 'S'
}
print(predict_survival(input_data))
