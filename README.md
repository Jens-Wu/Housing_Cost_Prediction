# Housing_Cost_Prediction

## Overview

This project was developed for a real estate consultancy serving clients such as developers, agencies, and investors, where accurate property pricing is critical. Traditionally, pricing decisions rely on official appraisals intended to remain impartial. However, clients must decide if a property’s true value is higher or lower than the appraisal to make informed investment decisions.
To support this, the consultancy aims to use data-driven modeling for more consistent and scalable decision-making. The project leverages a dataset from Ames, Iowa, which includes actual sale prices and ~80 property features (e.g. size, condition, backyard presence). The goal is to build a machine learning model that predicts house prices based on these features, helping clients make smarter buy/sell decisons. 

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

## Data Sources

For this case study a housing datasets from Ames, Iowa, which includes actual sale prices and ~80 property features (e.g. size, condition, backyard presence), was given:

- **Base dataset (CSV file)**:
  - `housing_iteration_5_classification.csv` --> dataset for predicting if a house is considered expensive or not
  - `housing_iteration_5_classification.txt` --> includes information about each housing feature in the dataset
  - `housing_iteration_6_regression.csv` --> dataset for predicting the price of a house
  - `housing_iteration_6_regression.txt` --> includes information about each housing feature in the dataset
  

## Methodology

1. 📥 **Data Loading & Exploration**  
Imported the Ames Housing dataset, which includes a range of numerical and categorical features describing property characteristics.
Conducted exploratory data analysis (EDA) to understand distributions, missing values, and feature correlations.<br><br>

2. 🧹 **Data Preprocessing**
- Feature selection: Removed irrelevant or redundant features.
- Imputation: Handled missing values using strategies such as mean, median, or constant filling depending on the feature type.
- Encoding: Applied one-hot or ordinal encoding to categorical features for model compatibility.
- Scaling: Standardized numerical features using StandardScaler to ensure comparability across features.<br><br>

3. ✂️ **Train-Test Split**  
The dataset was divided into training and test sets to assess generalization performance. 80% of the data was used for training and the remaining 20% for testing.<br><br>

4. 🤖 **Model Development**  
Evaluated multiple classification models for classification task:
- DecisionTreeClassifier
- KNeighborsClassifier
- RandomForestClassifier
- GradientBoostingClassifier
- SVC<br><br>

  Evaluated multiple regression models for price prediction:
- DecisionTreeRegressor
- GradientBoostingRegressor
- RandomForestRegressor<br><br>
  
  Used GridSearchCV to evaluate best hyperparameters for each model and take advantage of the benefits that come with cross validation (e.g. reduce overfitting, more realistic estimate of a model’s performance on unseen data,...).<br><br>

5. 🧪 **Model Evaluation**  
The following metrics were used to evaluate the performance of each classification model:
- accuracy_score
- recall_score
- precision_score
- f1_score
- balanced_accuracy_score
- cohen_kappa_score<br><br>

  The following metrics were used to evaluate the performance of each regression model:
- mean_absolute_error
- root_mean_squared_error
- mean_absolute_percentage_error
- r2_score<br><br>

6. 🏁 **Final Model Selection**  
Chose the best-performing model based on validation metrics for deployment or further tuning.<br><br>

## Results
**Classification**
- For the Classification task the company’s objective is to identify all houses that are expensive. It is more important to identify every expensive house than misclassifying a cheap house as an expensive one. Following this logic the recall score or sensitivity is selected as most important metric.
- The GradienBoosting model achieved the highest recall score with 0.82 and is therefore selected as preferred model for deployment.<br><br>

**Regression**
- For the Regression task R-squared is chosen as key metric as it is an universal score for judging model quality, no matter the units or scale.
- The GradienBoosting and RandomForest models perform equally well with R-squared scores of 0.853 and 0.852.<br><br>
  
  This is a basic comparison of the performance of different sklearn supervised ML models and further improvements of the models can be achieved by further feature engineering, hyperparameter tuning, etc.<br><br>

## Tools & Technologies

- Python (Pandas, Sklearn) for data manipulation, exploration,classification and regression
- Jupyter Notebook for an integrated code and report environment

## Repository Contents

- `Housing_Cost_Category_Prediction`: Jupyter notebook for the classification task (expensive or not)
- `Housing_Price_Prediction`: Jupyter notebook for the regression task (price prediction)
- `housing_data/`: Folder containing the `.csv` data files and `.txt` files used in the project.

## How to Run

1. Clone the repository.
2. Ensure the CSV files are in the `housing_data` directory.
3. Open the Jupyter notebook `Housing_Cost_Category_Prediction` or `Housing_Cost_Category_Prediction`.
4. Run the cells sequentially to reproduce the results.
5. (Optional) Take a look at the `.txt` for more information about each feature in the datasets.


## License
This project is open-source and available under the MIT License.
