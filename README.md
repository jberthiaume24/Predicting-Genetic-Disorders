# Predicting Genetic Disorders with PySpark: A Detailed Guide

## Introduction

In this notebook, we aim to predict various genetic disorders using PySpark. We will prepare the data, perform exploratory analysis, create a recommendation system, and build decision tree classifiers for different genetic disorders. 

## Setup Instructions for Google Colab

### 1. Setting Up PySpark in Google Colab

Google Colab is an excellent platform for running PySpark code. However, it requires some setup since PySpark is not pre-installed. Follow these steps to set up PySpark in Colab:

1. **Install PySpark:**

   Run the following cell to install PySpark:
   ```python
   !pip install -q pyspark
   ```

2. **Install Java:**

   PySpark requires Java. Install it using the following commands:
   ```python
   !apt-get install openjdk-11-jdk-headless -qq > /dev/null
   ```

3. **Set Environment Variables:**

   Set environment variables for Java:
   ```python
   import os
   os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-11-openjdk-amd64'
   ```

4. **Verify Installation:**

   Check if PySpark is installed correctly:
   ```python
   import pyspark
   pyspark.__version__
   ```

### 2. Import Necessary Libraries

Run the following code to import necessary libraries and initialize the PySpark environment:
```python
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Initialize Spark Session
conf = SparkConf().setAppName('GeneticDisordersPrediction')
sc = SparkContext.getOrCreate(conf=conf)
spark = SparkSession.builder.getOrCreate()
```

## Data Preparation

### 1. Uploading and Loading Data

Upload your `train.csv` file to the Colab environment and load it using Pandas:
```python
import pandas as pd

# Upload the CSV file
from google.colab import files
uploaded = files.upload()

# Load the data into a DataFrame
train_data = pd.read_csv("train.csv")
```

### 2. Data Cleaning

Clean the data by dropping irrelevant columns and handling missing values:
```python
# Dropping irrelevant columns
drop_columns = ['Patient Id', 'Patient First Name', 'Family Name', "Father's name", "Institute Name", "Location of Institute", "Status", "Parental consent", "Genetic Disorder"]
train_data.drop(columns=drop_columns, inplace=True)

# Dropping rows with missing Disorder Subclass values
train_data = train_data.dropna(subset=['Disorder Subclass'])

# Replacing missing values in numeric columns with median
numeric_columns = ['No. of previous abortion', 'White Blood cell count (thousand per microliter)', 'Patient Age', 'Blood cell count (mcL)', "Mother's age", "Father's age"]
for col in numeric_columns:
    train_data[col].fillna(train_data[col].median(), inplace=True)
    train_data[col] = train_data[col].astype('int64')

# Replacing missing values in categorical columns with 'Unknown'
categorical_columns = ['Blood test result', 'Birth defects', 'Gender', 'Heart Rate (rates/min', 'Respiratory Rate (breaths/min)', 'Follow-up', 'Place of birth']
train_data[categorical_columns] = train_data[categorical_columns].fillna('Unknown')

# Replacing various 'NA' and similar values with 'No'
train_data.replace(["-", "Not applicable", "Not available", "None", "No record"], "No", inplace=True)

# Converting binary categorical variables to 0 and 1
binary_var = ["Maternal gene", "Genes in mother's side", 'Inherited from father', 'Paternal gene', 'Birth asphyxia', 'Folic acid details (peri-conceptional)', 'H/O serious maternal illness', 'H/O radiation exposure (x-ray)', 'H/O substance abuse', 'Assisted conception IVF/ART', 'History of anomalies in previous pregnancies', 'Autopsy shows birth defect (if applicable)']
train_data[binary_var] = train_data[binary_var].replace({"Yes": 1, "No": 0})
train_data[binary_var] = train_data[binary_var].fillna(0).astype('int64')

# Creating binary columns for each disorder subclass
disorders = ["Leigh syndrome", "Mitochondrial myopathy", "Cystic fibrosis", "Tay-Sachs", "Diabetes", "Hemochromatosis", "Leber's hereditary optic neuropathy", "Alzheimer's", "Cancer"]
for disorder in disorders:
    train_data[disorder.replace("'", "").replace(" ", "_").lower()] = (train_data['Disorder Subclass'] == disorder).astype(int)
```

### 3. Save Cleaned Data

Save the cleaned DataFrame to a new CSV file:
```python
train_data.to_csv('train_data_cleaned.csv', index=False)
```

## Exploratory Data Analysis

### 1. Exploring the Data

Perform basic data exploration to understand the distribution and characteristics of the data:
```python
# Display summary statistics
train_data.describe()

# Count of each genetic disorder
train_data["Disorder Subclass"].value_counts()
```

## Recommendation System

### 1. Create a Recommendation System Using Collaborative Filtering

Create a recommendation system based on the similarity of binary features:
```python
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

# Extract binary data for recommendation system
df = train_data[["Maternal gene", "Genes in mother's side", 'Inherited from father', 'Paternal gene', 'Birth asphyxia', 'Folic acid details (peri-conceptional)', 'H/O serious maternal illness', 'H/O radiation exposure (x-ray)', 'H/O substance abuse', 'Assisted conception IVF/ART', 'History of anomalies in previous pregnancies', 'Autopsy shows birth defect (if applicable)', 'Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5', 'Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4', 'Symptom 5'] + [d for d in disorders]].copy()

# Normalize the user vectors to unit vectors
magnitude = np.sqrt(np.square(df).sum(axis=1))
df_normalized = df.divide(magnitude, axis='index')

# Calculate cosine similarity
def calculate_sim(df):
    df_sparse = sparse.csr_matrix(df)
    similarities = cosine_similarity(df_sparse.transpose())
    return pd.DataFrame(data=similarities, index=df.columns, columns=df.columns)

# Create similarity matrix
df_matrix = calculate_sim(df)

# Display top 5 variables most similar to each disorder
for disorder in [d.replace("'", "").replace(" ", "_").lower() for d in disorders]:
    print(f"Top 5 most similar variables for {disorder.replace('_', ' ').title()}:")
    print(df_matrix.loc[disorder].nlargest(6))
```

## Building Decision Tree Classifiers

### 1. Setting Up for PySpark

Import the necessary PySpark libraries:
```python
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
```

### 2. Import Cleaned Data into PySpark

Load the cleaned CSV data into PySpark:
```python
data = spark.read.csv("train_data_cleaned.csv", inferSchema=True, header=True)
```

### 3. Create Feature Vectors

Use `VectorAssembler` to combine features into a single vector column:
```python
assembler = VectorAssembler(
    inputCols=['Patient Age', "Genes in mother's side", 'Inherited from father', 'Maternal gene', 'Paternal gene', 'Blood cell count (mcL)', "Mother's age", "Father's age", 'Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5', 'Birth asphyxia', 'Autopsy shows birth defect (if applicable)', 'Folic acid details (peri-conceptional)', 'H/O serious maternal illness', 'H/O radiation exposure (x-ray)', 'H/O substance abuse', 'Assisted conception IVF/ART', 'History of anomalies in previous pregnancies', 'White Blood cell count (thousand per microliter)', 'Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4', 'Symptom 5'],
    outputCol="features")
output = assembler.transform(data)
```

### 4. Train Decision Tree Models

Build and evaluate decision tree classifiers for each genetic disorder:
```python
def train_and_evaluate_classifier(disorder_col):
    model_df = output.select("features", disorder_col)
    training_df, test_df = model_df.randomSplit([0.7, 0.3], seed=1000)
    
    classifier = DecisionTreeClassifier(labelCol=disorder_col)
    model = classifier.fit(training_df)
    predictions = model.transform(test_df)
    
    evaluator = MulticlassClassificationEvaluator(labelCol=disorder_col)
    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
    precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
    auc = evaluator.evaluate(predictions, {e

valuator.metricName: "weightedTruePositiveRate"})
    
    print(f"Disorder: {disorder_col}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"AUC: {auc:.2f}\n")

for disorder in [d.replace("'", "").replace(" ", "_").lower() for d in disorders]:
    train_and_evaluate_classifier(disorder)
```

## Conclusion

In this notebook, we set up PySpark in Google Colab, performed data cleaning and exploratory analysis, created a collaborative filtering-based recommendation system, and built decision tree classifiers to predict genetic disorders. The accuracy, precision, and AUC metrics provide insights into the model's performance, allowing further refinement and improvement of predictions.

Feel free to experiment with different models, feature engineering techniques, and hyperparameters to achieve better results!

---
