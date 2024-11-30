import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_dataset(df, task_type):
    if df.empty:
        raise ValueError("The DataFrame is empty after reading the CSV.")

    df.replace("None", pd.NA, inplace=True)

    # Separate features and target variable
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # Preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Preprocessing for NLP task
    if task_type == 'nlp':
        corpus = []
        ps = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        all_stopwords = set(stopwords.words('english'))

        # Process text from the first column
        for i in range(len(X)):
            text = re.sub('[^a-zA-Z]', ' ', X.iloc[i, 0])  # Extract text from the first column
            text = text.lower()  # Convert to lowercase
            text = text.split()  # Split into words
            # Stem and lemmatize, while removing stopwords
            text = [lemmatizer.lemmatize(ps.stem(word)) for word in text if word not in all_stopwords]
            text = ' '.join(text)  # Join words back to a single string
            corpus.append(text)

        # Use TF-IDF Vectorization instead of Count Vectorization
        vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))  # Unigrams and bigrams
        X_transformed = vectorizer.fit_transform(corpus).toarray()

        # Perform label encoding for the target variable
        le = LabelEncoder()
        y = le.fit_transform(y)

    else:
        # Combine numerical and categorical preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        X_transformed = preprocessor.fit_transform(X)

        # Perform label encoding for the target variable for classification tasks
        if task_type == 'classification':
            le = LabelEncoder()
            y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, None, X.columns.tolist()
