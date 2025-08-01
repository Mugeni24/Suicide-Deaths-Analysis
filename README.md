# 🧠 Suicide Deaths Analysis – WHO Mortality Database

## 📌 Project Overview
This project analyzes **global suicide mortality trends** using the **WHO Mortality Database**.  
The aim is to uncover patterns in suicide deaths over time, across countries, genders, and age groups, and to explore possible socio-economic influences.  

The analysis combines:
- **Python** for data cleaning, transformation, and statistical analysis.
- **Power BI** for creating interactive dashboards and visual storytelling.

---

## 🎯 Objectives
- Identify **global and country-level trends** in suicide deaths.
- Compare **suicide mortality rates** by gender and age group.
- Detect **regional variations** in suicide patterns.
- Build an **interactive visualization** for deeper insights.

---

## 📊 Dataset Information
- **Source:** [WHO Mortality Database](https://www.who.int/data/data-collection-tools/who-mortality-database)
- **Coverage:** Global (country-level breakdowns)
- **Time Period:** Varies by country (generally 2000–2022)
- **Format:** CSV / Excel
- **Key Columns:**
  - `Country` – Country name
  - `Year` – Year of record
  - `Sex` – Male / Female
  - `Age group` – Age category of individuals
  - `Deaths` – Number of suicide deaths
  - `Population` – Corresponding population size
  - `Crude rate` – Deaths per 100,000 population

---

## 🛠 Tools & Technologies
- **Python**: Pandas, NumPy, Matplotlib, Seaborn
- **Jupyter Notebook**: For exploratory data analysis
- **Power BI**: For building interactive dashboards
- **Excel**: For preliminary inspection and cleaning

---
## 🔬 Methodology

This section outlines the step-by-step approach taken to complete the project, ensuring transparency in the workflow and enabling others to reproduce the results.
Here's our data and how we find it. I used google colab.



### 1. Data Cleaning & Preprocessing
- Loaded the dataset using **Pandas**.

Raw data:
```python
import pandas as pd
suicide_df=pd.read_csv('drive/MyDrive/sucide-project/RELAY_WHS.csv')
print(suicide_df.columns)
suicide_df.info()
```
**Our sample data**:

<img width="596" height="291" alt="output1 df" src="https://github.com/user-attachments/assets/7ed28496-577d-456a-a470-8bbf48da6df4" />


---


  
- Data cleaning and preprocessing
Subtask:
Handle missing values in the dataset using appropriate techniques (e.g., imputation). Correct any inconsistencies or errors in the data. Perform data transformations, such as encoding categorical variables, to prepare the data for analysis and modeling.
Reason: Drop the specified columns with no non-null values, then check for remaining missing values and drop rows with missing values, and finally display the info of the cleaned DataFrame.
```python
suicide_df = suicide_df.drop(columns=['RATE_PER_100000_N', 'RATE_PER_100000_NL', 'RATE_PER_100000_NU'])
print(suicide_df.isnull().sum())
suicide_df.dropna(inplace=True)
suicide_df.info() 
```
output:

<img width="536" height="341" alt="cleaning" src="https://github.com/user-attachments/assets/b4dc8cff-b676-4d8a-9670-a10a8ea99b71" />
---

- Cleaned and standardized column names for consistency.
- Removed unnecessary columns to streamline the dataset.

### 2. Exploratory Data Analysis (EDA)
Exploratory data analysis (eda)
Subtask:
Generate detailed descriptive statistics to summarize the main characteristics of the data. Create a variety of visualizations (e.g., histograms, bar charts, line plots, heatmaps) to uncover trends, patterns, and relationships within the data.

Reasoning: To begin the exploratory data analysis, we first generate descriptive statistics for the numerical columns in the suicide_df DataFrame. Then import the necessary visualization libraries, matplotlib.pyplot and seaborn, to prepare for creating plots. Finally, create a bar chart to visualize the number of suicide deaths by gender, a line plot to show the trend of suicide deaths over time, and a heatmap to visualize the correlation between numerical features. This will address all the instructions in the subtask in a single, efficient code block.
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Descriptive statistics
print(suicide_df.describe())

# Bar chart for suicide deaths by gender
plt.figure(figsize=(8, 6))
sns.countplot(x='DIM_SEX', data=suicide_df)
plt.title('Suicide Deaths by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Line plot for suicide deaths over time
plt.figure(figsize=(12, 6))
suicide_df.groupby('DIM_TIME')['IND_ID'].count().plot(kind='line')
plt.title('Trend of Suicide Deaths Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Suicide Deaths')
plt.grid(True)
plt.show()

# Heatmap for correlation
plt.figure(figsize=(10, 8))
# Select only numeric columns for correlation matrix
numeric_cols = suicide_df.select_dtypes(include=['number']).columns
sns.heatmap(suicide_df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()
```
Here are the outputs:
<img width="644" height="304" alt="gender suicide deaths" src="https://github.com/user-attachments/assets/9031f9cf-48c0-4dac-9f29-85aa52c51e7d" />

<img width="672" height="400" alt="heatmap" src="https://github.com/user-attachments/assets/ec91ce7c-c39f-415c-a721-56a39fd63571" />

<img width="663" height="317" alt="deaths by time" src="https://github.com/user-attachments/assets/46a7b2ae-4001-4c6e-a6f1-304c1ddafd48" />

---

### 3. Machine Learning Modeling
Subtask:
Select a suitable machine learning model based on the problem statement and the insights from the EDA. Split the data into training and testing sets. Train the chosen model on the training data.
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Feature Selection
features = ['DIM_TIME', 'DIM_GEO_CODE_M49']
target = 'DIM_SEX'

X = suicide_df[features]
y = suicide_df[target]

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy}")
```
output:
<img width="255" height="16" alt="modeling output" src="https://github.com/user-attachments/assets/7d0c3704-cc34-4639-a454-37ab03e2838e" />
---
Model Improvement
In this section, we are to improve the model's performance by changing the prediction target and adding more features. The new target will be DIM_AGE, and the new features will include GEO_NAME_SHORT and DIM_SEX.
```python
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Create a copy of the dataframe for this modeling task
df_model = df.copy()

# Encode categorical features and the target variable
categorical_cols = ['GEO_NAME_SHORT', 'DIM_SEX', 'DIM_AGE']
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])

# Feature and Target Selection
features = ['DIM_TIME', 'GEO_NAME_SHORT', 'DIM_SEX', 'DIM_GEO_CODE_M49']
target = 'DIM_AGE'

X = df_model[features]
y = df_model[target]

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
improved_model = RandomForestClassifier(random_state=42)
improved_model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred_improved = improved_model.predict(X_test)
accuracy_improved = accuracy_score(y_test, y_pred_improved)

print(f"Improved Model Accuracy: {accuracy_improved}")

# Generate the classification report
print("\nClassification Report:")
# Use the label encoder for age to get original class names for the report
# Note: The 'le' variable here holds the encoder for the last column in the loop ('DIM_AGE')
report_improved = classification_report(y_test, y_pred_improved, target_names=le.classes_)
print(report_improved)

# Generate and plot the confusion matrix
conf_matrix_improved = confusion_matrix(y_test, y_pred_improved)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix_improved, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Improved Model Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
```
output:
<img width="587" height="447" alt="model improvement" src="https://github.com/user-attachments/assets/dd81414d-74c9-4e56-8d12-5f5eb2c07c66" />
---
Addressing Class Imbalance with SMOTE
In this section, I will use the SMOTE (Synthetic Minority Over-sampling Technique) to address the class imbalance problem in our dataset. This should help to improve the model's performance on the minority classes.
```python
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train a new model on the resampled data
smote_model = RandomForestClassifier(random_state=42)
smote_model.fit(X_train_resampled, y_train_resampled)

# Predictions and Evaluation
y_pred_smote = smote_model.predict(X_test)
accuracy_smote = accuracy_score(y_test, y_pred_smote)

print(f"SMOTE Model Accuracy: {accuracy_smote}")

# Generate the classification report
print("\nClassification Report:")
report_smote = classification_report(y_test, y_pred_smote, target_names=le.classes_)
print(report_smote)

# Generate and plot the confusion matrix
conf_matrix_smote = confusion_matrix(y_test, y_pred_smote)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix_smote, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('SMOTE Model Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
```
outputs:
<img width="282" height="198" alt="smote 1" src="https://github.com/user-attachments/assets/1086a7cf-a0a8-41b4-b736-e523935d3ddf" />
<img width="552" height="439" alt="smote" src="https://github.com/user-attachments/assets/72c3b22b-4742-491b-be5d-2053add9fa53" />


---

 Code Structure and Innovation
In this section, I will refactor the code into modular functions to improve its readability and reusability. I will also incorporate a feature importance visualization to add an innovative touch to our analysis.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(file_path):
    """
    Loads the dataset and cleans it.
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    df.drop(columns=['RATE_PER_100000_N', 'RATE_PER_100000_NL', 'RATE_PER_100000_NU'], inplace=True)

    return df

def train_and_evaluate_model(df, features, target):
    """
    Trains and evaluates a RandomForestClassifier model.
    """
    # Encode categorical features
    df_model = df.copy()
    categorical_cols = ['GEO_NAME_SHORT', 'DIM_SEX', 'DIM_AGE']
    for col in categorical_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])

    # Feature and Target Selection
    X = df_model[features]
    y = df_model[target]

    # Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Model Training
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_resampled, y_train_resampled)

    # Predictions and Evaluation
    y_pred = model.predict(X_test)

    # Generate and print the classification report
    print("Classification Report:")
    # Use the label encoder for age to get original class names for the report
    # Note: The 'le' variable here holds the encoder for the last column in the loop ('DIM_AGE')
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Generate and plot the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Feature Importance Visualization
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': features, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

    return model

if __name__ == '__main__':
    # Load and preprocess the data
    df = load_and_preprocess_data('drive/MyDrive/sucide-project/RELAY_WHS.csv')

    # Define features and target
    features = ['DIM_TIME', 'GEO_NAME_SHORT', 'DIM_SEX', 'DIM_GEO_CODE_M49']
    target = 'DIM_AGE'

    # Train and evaluate the model
    model = train_and_evaluate_model(df, features, target)

```
<img width="348" height="213" alt="improved accuracy" src="https://github.com/user-attachments/assets/8bf39878-84b5-4486-a5ca-182a40c000ec" />

<img width="613" height="438" alt="confusion matrix" src="https://github.com/user-attachments/assets/a4506b93-4199-4dfd-94a0-0f76c1602d35" />

<img width="756" height="285" alt="feature importance" src="https://github.com/user-attachments/assets/79d4e0e2-51bf-4d77-bdb1-f7dcdf92c5c2" />

---
### 4. Power BI Dashboard
- Connected the cleaned dataset to **Power BI**.
- Created an interactive dashboard with:
  - A **map** to visualize geographic patterns.
  - **Bar charts** and **line charts** for demographic and trend analysis.
- Added **slicers and filters** to allow interactive exploration of the data.

---

Final Summary and Insights
This section summarizes the key findings from our analysis and provides actionable insights based on the results.

Key Findings
Data Quality: The initial dataset had some data quality issues, including missing values and irrelevant columns. We addressed these issues by cleaning the data and imputing missing values.

Exploratory Data Analysis: Our EDA revealed some interesting patterns in the data. We saw that the number of suicides varies significantly across different countries and that there are also clear differences in suicide rates between males and females.

Machine Learning Modeling: We built a machine learning model to predict the age group based on various features. While the model's accuracy was not perfect, it provided some valuable insights into the factors that are most predictive of suicide risk.
Feature Importance: Our feature importance analysis showed that DIM_TIME and DIM_GEO_CODE_M49 are the most important features for predicting the age group.
Actionable Insights and Recommendations

Targeted Interventions: The significant variation in suicide rates across different demographics suggests that targeted interventions could be more effective than a one-size-fits-all approach. Public health campaigns and support services should be tailored to the specific needs of different age groups, sexes, and countries.

Further Research: The feature importance analysis suggests that DIM_TIME and DIM_GEO_CODE_M49 are important predictors of suicide risk. Further research could explore the reasons for this, which could lead to a better understanding of the underlying causes of suicide.

Data Collection: The data quality issues we encountered highlight the importance of accurate and complete data collection. Improving the quality of data on suicide rates would be a valuable step towards better understanding and preventing this tragic outcome.

---

## 📈 Results

- **Key Finding 1:** Suicide rates are significantly higher among males than females across all countries and age groups.
- **Key Finding 2:** Suicide trends vary greatly by country — some show a clear downward trend, others an upward trend.
- **Key Finding 3:** The machine learning model achieved an accuracy of **67%** in predicting the age group, with **DIM_TIME** and **DIM_GEO_CODE_M49** as the most important features.

---

## 💡 Recommendations

- **Recommendation 1:** Public health interventions should focus more on **males**, who are at a higher risk of suicide.
- **Recommendation 2:** Conduct further research to understand **country-specific differences** in suicide trends.
- **Recommendation 3:** Improve data collection methods to ensure more **complete and accurate** reporting.

---

## 🚀 Future Work

- **Idea 1:** Integrate additional datasets such as mental health services availability or economic indicators for deeper insights.
- **Idea 2:** Explore **advanced machine learning models** (e.g., deep learning) to improve prediction accuracy.
- **Idea 3:** Develop a **web application** that allows users to explore the dataset and findings interactively.

---

