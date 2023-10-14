# %%
import pandas as pd
movies = pd.read_csv('movies.dat', sep='::', names=['MovieNo','Title', 'Genres'], engine='python', encoding='latin1')
ratings = pd.read_csv('ratings.dat',  sep='::', names= ['MovieID','Rating', 'Timestamp' ], engine='python', encoding='latin1')
users = pd.read_csv('users.dat', sep='::', names=['UserId', 'Sex', 'Age', 'Occupation', 'Zip-code'], engine='python', encoding='latin1')

# %%
# Drop rows with missing values
movies.dropna(inplace=True)

# Display the first few rows
movies.head()

# %%
# Drop rows with missing values
ratings.dropna(inplace=True)

# Display the first few rows
ratings.head()

# %%
# Drop rows with missing values
users.dropna(inplace=True)

# Display the first few rows
users.head()

# %%
# Merge User and Ratings data on 'MovieID'
user_ratings = users.merge(ratings, left_on='UserId', right_on='MovieID')

# Display the merged DataFrame
user_ratings

# %%
# Merge User_ratings with movies_df based on 'MovieID' and 'MovieNo'
df = user_ratings.merge(movies, left_on='MovieID', right_on='MovieNo')

# Display the final merged DataFrame
df

# %%
df.info()

# %%
df.describe()

# %%
df.isnull().sum()

# %%
# 2. Convert Timestamp to a readable date-time format:

df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

# 3. Ensure data types are consistent:
# Ensure Rating is numeric
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

# Ensure Title and Genres are strings
df['Title'] = df['Title'].astype(str)
df['Genres'] = df['Genres'].astype(str)

# 4. Remove duplicates:
df.drop_duplicates(inplace=True)

# Display the cleaned data
df.head()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you've loaded your dataset into a DataFrame called 'df'
# df = pd.read_csv('path_to_your_dataset.csv')

# Set the style for seaborn plots
sns.set_style("whitegrid")

# 1. Distribution of Age, Rating, Occupation:
fig, ax = plt.subplots(5, 1, figsize=(10, 20))

sns.histplot(df['Age'], ax=ax[0], kde=True)
ax[0].set_title('Distribution of Age')

sns.histplot(df['Rating'], ax=ax[1], kde=True)
ax[1].set_title('Distribution of Rating')

sns.histplot(df['Occupation'], ax=ax[2], kde=True)
ax[2].set_title('Distribution of Occupation')

sns.countplot(data=df, x='Sex', ax=ax[3])
ax[3].set_title('Distribution of Sex')

top_zip_codes = df['Zip-code'].value_counts().head(10).index
sns.countplot(data=df, y='Zip-code', order=top_zip_codes, ax=ax[4])
ax[4].set_title('Top 10 Zip-codes')

plt.tight_layout()
plt.show()

# 2. Average ratings per movie, per user:
avg_rating_per_movie = df.groupby('Title')['Rating'].mean()
avg_rating_per_user = df.groupby('UserId')['Rating'].mean()

print("Average rating per movie:\n", avg_rating_per_movie.head())
print("\nAverage rating per user:\n", avg_rating_per_user.head())

# 3. Number of ratings per movie:
ratings_count_per_movie = df.groupby('Title')['Rating'].count().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
ratings_count_per_movie.head(10).plot(kind='bar')
plt.title('Top 10 Movies by Number of Ratings')
plt.xlabel('Movie Title')
plt.ylabel('Number of Ratings')
plt.show()

# 4. Distribution of Genres:
all_genres = df['Genres'].str.split('|', expand=True).stack()
plt.figure(figsize=(15, 7))
sns.countplot(y=all_genres, order=all_genres.value_counts().index)
plt.title('Distribution of Genres')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.show()

# 5. Distribution of Timestamp:
df['Year'] = df['Timestamp'].dt.year
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='Year')
plt.title('Distribution of Ratings Over Years')
plt.xlabel('Year')
plt.ylabel('Number of Ratings')
plt.show()

# 6. Most rated movies:
top_rated_movies = df['MovieID'].value_counts().head(10).index
plt.figure(figsize=(10, 5))
sns.countplot(data=df, y='MovieID', order=top_rated_movies)
plt.title('Top 10 Most Rated Movies')
plt.xlabel('Number of Ratings')
plt.ylabel('MovieID')
plt.show()


# %%
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

class SVDMovieRecommender:
    def __init__(self):
        self.model = SVD()
        self.trainset = None
        self.testset = None
        self.movie_id_to_name = {}
        self.movie_name_to_id = {}

    def fit(self, df):
        reader = Reader(rating_scale=(df['Rating'].min(), df['Rating'].max()))
        data = Dataset.load_from_df(df[['UserId', 'MovieID', 'Rating']], reader)
        
        # Splitting the data into training and test sets
        self.trainset, self.testset = train_test_split(data, test_size=0.25, random_state=42)
        
        self.model.fit(self.trainset)

        # Create mapping from MovieID to Title and vice-versa
        self.movie_id_to_name = pd.Series(df.Title.values, index=df.MovieID).to_dict()
        self.movie_name_to_id = pd.Series(df.MovieID.values, index=df.Title).to_dict()

    def predict(self, user_id, movie_name):
        movie_id = self.movie_name_to_id[movie_name]
        return self.model.predict(user_id, movie_id).est

    def evaluate(self):
        predictions = self.model.test(self.testset)
        rmse = accuracy.rmse(predictions)
        return rmse

# Usage
recommender = SVDMovieRecommender()
recommender.fit(df)

# Evaluation
rmse = recommender.evaluate()
print(f"Root Mean Square Error (RMSE) for SVD: {rmse:.2f}")



# %%
# Prediction
user_id = input("Enter user ID: ")
movie_name = input("Enter movie name: ")
print(f"SVD Prediction for User {user_id} and Movie '{movie_name}': {recommender.predict(user_id, movie_name)}")
