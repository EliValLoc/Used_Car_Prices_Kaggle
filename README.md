# Used Car Price Prediction | Kaggle Competition

This repository contains my solution for **Assignment 3** in the course **Business Analytics and Data Science**.
The assignment was organised as a Kaggle competition, where the goal was to predict the prices of used cars based on structured vehicle data.

The competition was evaluated using **Root Mean Squared Error (RMSE)**. My final submission achieved **1st place among 43 participants**.

## Project Overview

The task was to predict the price of used cars listed on Craigslist. The dataset contains vehicle-level information such as brand, model, year, mileage, fuel type, transmission, engine size, tax, and fuel efficiency.

The modelling objective was to build a regression model that minimises the RMSE on the hidden Kaggle test set.

## Competition Description

Craigslist is one of the largest online platforms for used vehicle listings. However, used car prices can vary substantially depending on technical characteristics, brand, model, mileage, age, fuel type, and other attributes.

The goal of this project was therefore to estimate used car prices as accurately as possible using machine learning methods.

Competition page: [Assignment 3 DSBA on Kaggle](https://www.kaggle.com/competitions/assignment-3-dsba/leaderboard)

## Repository Structure

```text
.
├── notebook/
│   └── CreateKaggleSubmission_DSBA_2024.ipynb
├── presentation/
│   └── Assignment3_Presentation.pdf
├── README.md
└── requirements.txt
```

Depending on the data licence and competition rules, the raw data may either be included in the repository or accessed directly through Kaggle.

## Data

The training dataset contains the target variable `price`, while the test dataset contains the same vehicle features without the target variable.

Main features include:

| Feature        | Description                          |
| -------------- | ------------------------------------ |
| `model`        | Vehicle model                        |
| `year`         | Year of manufacture                  |
| `transmission` | Transmission type                    |
| `mileage`      | Vehicle mileage                      |
| `fuelType`     | Fuel type                            |
| `tax`          | Vehicle tax                          |
| `mpg`          | Miles per gallon                     |
| `engineSize`   | Engine size                          |
| `brand`        | Vehicle brand                        |
| `ID`           | Observation identifier               |
| `price`        | Target variable in the training data |

## Methodology

The project followed a structured modelling workflow:

1. Exploratory data analysis
2. Data preprocessing
3. Feature transformation
4. Model training
5. Hyperparameter tuning
6. Cross-validation
7. Kaggle submission generation

The main performance metric was **RMSE**, as defined by the Kaggle competition.

## Exploratory Data Analysis

The first step was to analyse the distribution of the target variable and the relationship between vehicle features and price.

The price distribution was right-skewed, which suggested that a log-transformation of the target variable could improve model performance. Numerical features such as `engineSize`, `mileage`, and `year` were analysed visually to better understand their relationship with the vehicle price.

## Models

Several model families were tested and compared.

### Linear Models

I started with linear regression models as a baseline. The categorical variables were one-hot encoded, while numerical variables were standardised.

The following linear approaches were tested:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Log-transformed target variable
* Polynomial features
* Spline transformations

The linear models were useful as a benchmark. However, their performance suggested that the relationship between the vehicle characteristics and the price was not fully linear.

### Tree-Based Models

I then trained non-linear tree-based models:

* Random Forest Regressor
* XGBoost Regressor

These models performed substantially better than the linear approaches. The best-performing model was an **XGBoost Regressor** with extensive hyperparameter tuning.

### Hyperparameter Tuning

Several tuning strategies were used:

* `GridSearchCV`
* `RandomizedSearchCV`
* `BayesSearchCV`
* Experimental tuning with Optuna

The final workflow used a systematic tuning strategy. I first explored a broader parameter space with `RandomizedSearchCV` and then refined the search space with `BayesSearchCV`.

## Text-Based Feature Processing

In addition to the structured features, I experimented with text-based feature processing for categorical variables such as `model` and `brand`.

I tested sentence embeddings using:

* `all-MiniLM-L6-v2`
* `microsoft/mpnet-base`

I also experimented with dimensionality reduction using UMAP. However, the embedding-based models did not outperform the best structured-data XGBoost model. They achieved similar performance but required substantially more computational resources.

From a practical perspective, the simpler XGBoost model without text embeddings was therefore preferable.

## Results

The best model was an **XGBoost Regressor with BayesSearchCV hyperparameter tuning**.

In cross-validation, the best model achieved an RMSE of approximately **2,097**.
On the Kaggle leaderboard, the final winning submission achieved an RMSE of approximately **1,888**.

A second approach using text embeddings achieved a very similar Kaggle RMSE of approximately **1,893**, but it was more computationally expensive and therefore not selected as the preferred final model.

## Key Learnings

This project helped me gain practical experience in several important areas of applied machine learning:

* Building a full regression pipeline for tabular data
* Comparing linear and non-linear models
* Understanding the importance of feature transformations
* Using cross-validation for model selection
* Applying hyperparameter tuning systematically
* Working with XGBoost and Random Forest models
* Experimenting with text embeddings for categorical variables
* Evaluating the trade-off between model performance and computational cost

The project also showed that more complex feature processing does not necessarily lead to better performance. In this case, the best structured-data XGBoost model performed slightly better than the embedding-based models while being easier and cheaper to train.

## How to Run the Notebook

1. Clone this repository.

```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required packages.

```bash
pip install -r requirements.txt
```

3. Open the notebook.

```bash
jupyter notebook notebook/CreateKaggleSubmission_DSBA_2024.ipynb
```

4. Run the notebook step by step.

## Requirements

The project uses Python and common data science libraries, including:

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `xgboost`
* `scikit-optimize`
* `sentence-transformers`
* `umap-learn`
* `jupyter`

## Disclaimer

This repository was created for educational purposes as part of a university assignment. The dataset originates from the corresponding Kaggle competition. Please refer to the Kaggle competition page and its rules for the applicable data usage terms.
