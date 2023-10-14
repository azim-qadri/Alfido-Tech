# %%
import pandas as pd

# %%
df = pd.read_csv('IRIS.csv')
df.head(10)

# %%
print(df.info)


# %%
df.describe(include='all')

# %%
# Checking null values
df.isnull().sum()

# %%
# Checking duplicated values
df.duplicated().sum()

# %%
# Removing duplicated values
df.drop_duplicates(inplace=True)
# Again checking duplicated values
df.duplicated().sum()

# %%
# Checking for outliers
import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(data=df)
plt.title("Boxplot")
plt.show()

# %%
"""
Since, outliers are close to the other values and we have less records we will not remove them
"""

# %%
df.species.value_counts()

# %%
"""
Exploratory Data Analysis
"""

# %%
# Pairplot to visualize relationships between variables
sns.pairplot(df, hue="species", markers=["s", "o", "D"])

# %%
# Violin plot to visualize distribution and density by species
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.violinplot(x="species", y="petal_length", data=df)
plt.subplot(2, 2, 2)
sns.violinplot(x="species", y="petal_width", data=df)
plt.subplot(2, 2, 3)
sns.violinplot(x="species", y="sepal_width", data=df)
plt.subplot(2, 2, 4)
sns.violinplot(x="species", y="sepal_length", data=df)
plt.tight_layout()

# %%
# Correlation heatmap
df1 = df.copy()
df1.drop("species", axis=1, inplace=True)
correlation = df1.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# %%
# Input features dataset
x = df.drop(columns="species", axis=1)
x


# %%
species_map = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}
y = df['species'].map(species_map)
y


# %%
"""
split data for training and testing
"""

# %%
from sklearn.model_selection import train_test_split
# Splitting the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# %%
y_train

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Define the models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC()
}

# Train the models and evaluate them
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"{name}:\n\tAccuracy: {accuracy}\n\tPrecision: {precision}\n\tRecall: {recall}\n\tF1 Score: {f1}")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion matrix: {name}")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



# %%
"""
Out of all these we found that decision trees perform the best with an accuracy of 96.6%
"""