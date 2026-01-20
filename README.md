Titanic-ML-Adventures 🚢💻

Analyzing the Titanic dataset to explore passenger survival patterns and build machine learning models for prediction.

Project Overview

The Titanic dataset is a classic dataset used in machine learning and data analysis. This project explores patterns in passenger survival based on features such as age, sex, passenger class, and more. The project also implements various machine learning models to predict survival outcomes.

Dataset

The dataset used in this project contains information about Titanic passengers:

PassengerId

Pclass (Passenger Class)

Name

Sex

Age

SibSp (Number of siblings/spouses aboard)

Parch (Number of parents/children aboard)

Ticket

Fare

Cabin

Embarked (Port of Embarkation)

Survived (Target variable: 0 = No, 1 = Yes)

Source: Kaggle Titanic Dataset

Project Objectives

Explore and visualize passenger data to identify survival patterns.

Preprocess data for machine learning (handling missing values, encoding categorical variables, scaling features, etc.).

Build and evaluate multiple machine learning models for predicting survival:

Logistic Regression

Decision Tree

Random Forest

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Naive Bayes

Compare model performance using accuracy, precision, recall, and F1-score.

Identify the best-performing model for predicting passenger survival.

Installation

Clone the repository:

git clone https://github.com/yourusername/Titanic-ML-Adventures.git


Navigate into the project directory:

cd Titanic-ML-Adventures


Install required packages:

pip install -r requirements.txt

Usage

Load the dataset:

import pandas as pd

df = pd.read_csv("data/titanic.csv")


Perform exploratory data analysis (EDA):

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x="Survived", data=df)
plt.show()


Train machine learning models:

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

Project Structure
Titanic-ML-Adventures/
│
├── data/               # Titanic dataset CSV file
├── notebooks/          # Jupyter notebooks for EDA and model building
├── src/                # Python scripts for preprocessing and modeling
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

Evaluation Metrics

Accuracy

Precision

Recall

F1-score

Confusion Matrix


License

This project is licensed under the MIT License.
