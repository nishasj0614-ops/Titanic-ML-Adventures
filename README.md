🚢 Titanic Survival Prediction – Machine Learning Project
This project builds and compares multiple machine learning classification models to predict passenger survival on the Titanic dataset.
It includes data preprocessing, model training, evaluation, hyperparameter tuning, and visual performance comparison.

📌 Project Overview
The goal is to predict whether a passenger survived (survived = 1) or not (survived = 0) using demographic and travel-related features such as age, fare, class, gender, etc.
The project:


Cleans and preprocesses the data


Trains multiple ML models


Evaluates models using standard classification metrics


Applies GridSearchCV for hyperparameter tuning


Compares model performance before and after tuning


Visualizes results using bar charts, heatmaps, and confusion matrices



📊 Dataset


Source: Seaborn Titanic Dataset


Rows: 891


Target Variable: survived


Key Features Used


pclass


age


sibsp


parch


fare


sex


embarked



🧹 Data Preprocessing


Dropped unnecessary columns: deck, embark_town, alive, class, who


Filled missing values:


age → median


embarked → mode




Encoded categorical variables using OneHotEncoder


Scaled numerical features using StandardScaler


Used ColumnTransformer for clean preprocessing pipelines



🤖 Models Implemented
The following classification models were trained and evaluated:


Logistic Regression


Decision Tree


Random Forest


K-Nearest Neighbors (KNN)


Support Vector Classifier (SVC)


Gaussian Naive Bayes


All models were implemented using Scikit-learn Pipelines.

📈 Model Evaluation Metrics
Each model was evaluated using:


Accuracy


Precision


Recall


F1-score


Performance was compared:


Before hyperparameter tuning


After hyperparameter tuning



🔧 Hyperparameter Tuning


Used GridSearchCV (5-fold cross-validation)


Optimized for F1-score


Applied model-specific parameter grids



🏆 Best Model
After tuning, the Random Forest Classifier performed best overall.
Final Configuration:
RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)


📊 Visualizations
The project includes:


Bar chart comparing model performance before vs after tuning


Heatmap of evaluation metrics


Confusion matrix for the best model


These visualizations help clearly understand model strengths and weaknesses.

🛠️ Technologies Used


Python


NumPy


Pandas


Matplotlib


Seaborn


Scikit-learn


Jupyter Notebook





pip install numpy pandas matplotlib seaborn scikit-learn



Run the notebook or Python script.



📌 Conclusion
This project demonstrates a complete machine learning workflow:


Data cleaning


Feature engineering


Model building


Hyperparameter tuning


Model evaluation and comparison


It is ideal for beginners and intermediate learners looking to understand classification models and pipelines in real-world datasets.

👤 Author
Nisha S
Machine Learning & Data Science Enthusiast

