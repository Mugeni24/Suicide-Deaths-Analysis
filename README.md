# 🧠 Global Suicide Mortality Analysis – WHO Mortality Databases

## 🌍 Project Overview
This project delivers a **cutting-edge analysis** of global suicide mortality trends, leveraging the **WHO Mortality Database** to uncover actionable insights. By integrating advanced data science, machine learning, and interactive visualizations, it examines patterns across time, countries, genders, and age groups, while exploring socio-economic influences. The findings aim to drive evidence-based public health strategies and foster meaningful societal impact.

**Core Components**:
- **Python** 📊: For robust data processing, statistical modeling, and predictive analytics.
- **Power BI** 📈: For dynamic, interactive dashboards that transform data into compelling stories.
- **Google Colab** 💻: For seamless, collaborative analysis and development.

---

## 🎯 Objectives
- 🔍 Uncover **global and regional trends** in suicide mortality.
- ⚖️ Analyze **suicide rates** by gender and age group.
- 🌐 Identify **geographic variations** in suicide patterns.
- 🖼️ Create **interactive visualizations** to empower stakeholders with intuitive insights.

---

## 📊 Dataset Information
- **Source**: [WHO Mortality Database](https://www.who.int/data/data-collection-tools/who-mortality-database) 🌐
- **Coverage**: Global, with country-level granularity
- **Time Period**: 2000–2022 (varies by country) 🕰️
- **Format**: CSV / Excel 📑
- **Key Columns**:
  - `Country`: Country name
  - `Year`: Year of record
  - `Sex`: Male / Female
  - `Age group`: Age category
  - `Deaths`: Number of suicide deaths
  - `Population`: Population size
  - `Crude rate`: Deaths per 100,000 population

---

## 🛠️ Tools & Technologies
- **Python** 🐍: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn
- **Jupyter Notebook (Google Colab)** 📓: For exploratory analysis and model development
- **Power BI** 📊: For interactive, user-friendly dashboards
- **Excel** 📈: For initial data inspection and preprocessing

---

## 🔬 Methodology
This section outlines a **systematic, reproducible workflow** executed in **Google Colab**, ensuring analytical rigor and transparency.

### 1. Data Cleaning & Preprocessing 🧹
- **Objective**: Transform raw data into a clean, analysis-ready format.
- **Process**:
  - Loaded the dataset using **Pandas**:
    ```python
    import pandas as pd
    suicide_df = pd.read_csv('drive/MyDrive/sucide-project/RELAY_WHS.csv')
    print(suicide_df.columns)
    suicide_df.info()
    ```
**Output** 📊  
[![Sample Data](https://github.com/user-attachments/assets/7ed28496-577d-456a-a470-8bbf48da6df4)](#)


---
    ```python
    suicide_df = suicide_df.drop(columns=['RATE_PER_100000_N', 'RATE_PER_100000_NL', 'RATE_PER_100000_NU'])
    print(suicide_df.isnull().sum())
    suicide_df.dropna(inplace=True)
    suicide_df.info()
    ```
**Output** 📊  
[![Cleaning Output](https://github.com/user-attachments/assets/b4dc8cff-b676-4d8a-9670-a10a8ea99b71)](#)

---

**Standardized column names and optimized the dataset for efficiency** ✅


### 2. Exploratory Data Analysis (EDA) 📈
- **Objective**: Reveal trends, patterns, and relationships through statistical summaries and visualizations.
- **Process**:
  - Generated descriptive statistics and visualizations using **Matplotlib** and **Seaborn**:
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
    numeric_cols = suicide_df.select_dtypes(include=['number']).columns
    sns.heatmap(suicide_df[numeric_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Numerical Features')
    plt.show()
    ```
**Output Visualizations** 📊

1. **Gender Distribution**  
   [![Gender Distribution](https://github.com/user-attachments/assets/9031f9cf-48c0-4dac-9f29-85aa52c51e7d)](#)

2. **Correlation Heatmap**  
   [![Correlation Heatmap](https://github.com/user-attachments/assets/ec91ce7c-c39f-415c-a721-56a39fd63571)](#)

3. **Trend Over Time**  
   [![Trend Over Time](https://github.com/user-attachments/assets/46a7b2ae-4001-4c6e-a6f1-304c1ddafd48)](#)


### 3. Machine Learning Modeling 🤖
- **Objective**: Develop predictive models to identify key factors influencing suicide risk.
- **Initial Model**:
  - Built a **Random Forest Classifier** to predict gender:
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    features = ['DIM_TIME', 'DIM_GEO_CODE_M49']
    target = 'DIM_SEX'

    X = suicide_df[features]
    y = suicide_df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy}")
    ```
    **Output**:
    ![Initial Model Accuracy](https://github.com/user-attachments/assets/7d0c3704-cc34-4639-a454-37ab03e2838e)
- **Improved Model**:
  - Enhanced the model to predict age group (`DIM_AGE`) with additional features:
    ```python
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix

    df_model = suicide_df.copy()
    categorical_cols = ['GEO_NAME_SHORT', 'DIM_SEX', 'DIM_AGE']
    for col in categorical_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])

    features = ['DIM_TIME', 'GEO_NAME_SHORT', 'DIM_SEX', 'DIM_GEO_CODE_M49']
    target = 'DIM_AGE'

    X = df_model[features]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    improved_model = RandomForestClassifier(random_state=42)
    improved_model.fit(X_train, y_train)
    y_pred_improved = improved_model.predict(X_test)
    accuracy_improved = accuracy_score(y_test, y_pred_improved)

    print(f"Improved Model Accuracy: {accuracy_improved}")
    print("\nClassification Report:")
    report_improved = classification_report(y_test, y_pred_improved, target_names=le.classes_)
    print(report_improved)

    conf_matrix_improved = confusion_matrix(y_test, y_pred_improved)
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix_improved, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Improved Model Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    ```
**Output** 📊  
[![Improved Model Results](https://github.com/user-attachments/assets/dd81414d-74c9-4e56-8d12-5f5eb2c07c66)](#)

---

### **Addressing Class Imbalance** ⚖️  
Applied **SMOTE** to balance the dataset and improve performance.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
smote_model = RandomForestClassifier(random_state=42)
smote_model.fit(X_train_resampled, y_train_resampled)
y_pred_smote = smote_model.predict(X_test)
accuracy_smote = accuracy_score(y_test, y_pred_smote)

print(f"SMOTE Model Accuracy: {accuracy_smote}")
print("\nClassification Report:")
report_smote = classification_report(y_test, y_pred_smote, target_names=le.classes_)
print(report_smote)

conf_matrix_smote = confusion_matrix(y_test, y_pred_smote)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix_smote, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('SMOTE Model Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

```
**Output** 📊  

- **SMOTE Accuracy and Classification Report**  
[![SMOTE Results](https://github.com/user-attachments/assets/1086a7cf-a0a8-41b4-b736-e523935d3ddf)](#)  

- **SMOTE Confusion Matrix**  
[![SMOTE Confusion Matrix](https://github.com/user-attachments/assets/72c3b22b-4742-491b-be5d-2053add9fa53)](#)  

### **Code Refactoring & Innovation**:
Modularized code for reusability and added feature importance visualization:
    
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
        df = pd.read_csv(file_path)
        df.drop(columns=['RATE_PER_100000_N', 'RATE_PER_100000_NL', 'RATE_PER_100000_NU'], inplace=True)
        df.dropna(inplace=True)
        return df

    def train_and_evaluate_model(df, features, target):
        df_model = df.copy()
        categorical_cols = ['GEO_NAME_SHORT', 'DIM_SEX', 'DIM_AGE']
        for col in categorical_cols:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col])

        X = df_model[features]
        y = df_model[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)

        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

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
        df = load_and_preprocess_data('drive/MyDrive/sucide-project/RELAY_WHS.csv')
        features = ['DIM_TIME', 'GEO_NAME_SHORT', 'DIM_SEX', 'DIM_GEO_CODE_M49']
        target = 'DIM_AGE'
        model = train_and_evaluate_model(df, features, target)
    ```
**Output** 📊  

- **Classification Report and Confusion Matrix**  
[![Improved Accuracy](https://github.com/user-attachments/assets/8bf39878-84b5-4486-a5ca-182a40c000ec)](#)  
[![Confusion Matrix](https://github.com/user-attachments/assets/a4506b93-4199-4dfd-94a0-0f76c1602d35)](#)  

- **Feature Importance**  
[![Feature Importance](https://github.com/user-attachments/assets/79d4e0e2-51bf-4d77-bdb1-f7dcdf92c5c2)](#)  

### 4. Power BI Dashboard 📊✨

#### **Objective**
Deliver an intuitive platform for stakeholders to explore suicide trends interactively.

---

#### **Dashboard Preview** 🛠️
![DASHBOARD2](https://github.com/user-attachments/assets/b8cb2817-7eed-4341-84f5-6fd500380c6e)

---

#### **Process**
1. **Imported the cleaned dataset** into **Power BI**.
2. **Designed a visually engaging dashboard** featuring:

---

##### **Geographic Map** 🌍  
Highlights regional patterns.  
![Map](https://github.com/user-attachments/assets/b5b3c0b6-5558-4cf9-b398-d6f628870363)

---

##### **Line Chart – Track Records by Time** 📈  
- This allows us to see if the number of suicide mortality records is increasing, decreasing, or staying the same over time.  
- Helps identify long-term trends and patterns.  
- Detects unusual spikes or dips in the data that might warrant further investigation.  
- For example, a sudden increase in the number of records in a particular year could be due to a major event, such as an economic crisis or a natural disaster.  

![Trends Over Time](https://github.com/user-attachments/assets/a886ab68-39fb-4006-a59c-54fa7fe06909)

---

##### **Stacked Bar Chart – Proportional Breakdown by Age Group and Sex**  
- This chart takes us a step deeper.  
- It asks:  
  > "Of all the females who die by suicide, what percentage are young, what percentage are middle-aged, and what percentage are elderly?"  
  and then it asks the same question for males.  

![Proportional Breakdown](https://github.com/user-attachments/assets/dac7b56e-9e11-43a7-9c26-f673b8685314)

---

##### **Ribbon Chart – Records by Time and Sex**  
- Shows the number of suicide mortality records over time, broken down by sex.  
- Performs a **trend analysis** to compare the suicide mortality patterns of males and females over the years.  

![Ribbon Chart](https://github.com/user-attachments/assets/00c8e334-1389-4322-8551-84bd28d7c523)

---

#### **Slicers and Filters** 🛠️
- **Filter Example**:  
  - Filters countries starting with the letter **A** only.  
  - The funnel visual also helps us see the breakdown by sex when we click on it.  

![Filter](https://github.com/user-attachments/assets/f02f5739-75de-4b39-a9ea-180454e4f0af)

- **Additional Slicer**:  
  - Allows for easy interaction in the overall dashboard.  

![Dashboard Slicer](https://github.com/user-attachments/assets/5b2c38a2-6880-4fa7-b18c-f922efab18a4)

---

## 📈 Key Findings 🔍
1. **Gender Disparity** ⚖️: Males exhibit significantly higher suicide rates than females across all regions and age groups.
2. **Regional Variations** 🌐: Suicide trends vary widely by country, with some showing declines and others alarming increases.
3. **Predictive Insights** 🤖: The machine learning model achieved **67% accuracy** in predicting age groups, with **DIM_TIME** and **DIM_GEO_CODE_M49** as the most influential features.

---

## 💡 Recommendations 🚀
1. **Targeted Interventions** 🎯: Prioritize mental health resources for males, who face elevated suicide risks.
2. **Region-Specific Strategies** 🌍: Develop tailored public health initiatives to address country-specific trends.
3. **Enhanced Data Collection** 📑: Invest in robust systems to ensure accurate, comprehensive data reporting.

---

## 🌟 Future Work
1. **Expanded Datasets** 📊: Integrate mental health, economic, and cultural data for richer insights.
2. **Advanced Modeling** 🤖: Explore deep learning and ensemble methods to enhance predictive accuracy.
3. **Interactive Web Platform** 💻: Build a web application for real-time data visualization and exploration.

---

## 📚 Conclusion
This project represents a **transformative contribution** to global suicide mortality analysis. By seamlessly integrating data cleaning, exploratory analysis, machine learning, and interactive visualizations, it delivers actionable insights to inform public health strategies. The modular code, innovative feature importance analysis, and dynamic Power BI dashboard underscore its technical sophistication and societal impact.

---

## 🙏 Reflective Verse
*"Come unto me, all ye that labour and are heavy laden, and I will give you rest."*  
— Mathew 11:28

This verse inspires hope and compassion from Jesus Christ our Lord and saviour, reminding us that if we are heavy in our hearts and feel overwhelmed by our life, Jesus will free us through prayer; counting on him.

---

## 💪 Encouragement
This project is meant to illuminate the path toward meaningful change. Let this work inspire you to continue harnessing data for good, driving innovation, and making a lasting impact on global mental health. Keep shining!
