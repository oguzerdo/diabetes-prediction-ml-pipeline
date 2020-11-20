<a href="https://www.oguzerdogan.com/">
    <img src="https://www.oguzerdogan.com/wp-content/uploads/2020/10/logo_oz.png" width="200" align="right"></a>



![banner](https://cdn.britannica.com/s:700x500/42/93542-050-E2B32DAB/women-Pima-shinny-game-field-hockey.jpg)

# Pima Indians Diabetes Classification Project



# 🎯 Objective of Project

Diabetes, is a group of metabolic disorders in which there are high blood sugar levels over a prolonged period. Symptoms of high blood sugar include frequent urination, increased thirst, and increased hunger. If left untreated, diabetes can cause many complications. Acute complications can include diabetic ketoacidosis, hyperosmolar hyperglycemic state, or death. Serious long-term complications include cardiovascular disease, stroke, chronic kidney disease, foot ulcers, and damage to the eyes.

This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases.

The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are **females at least 21 years old of Pima Indian heritage.**

------

**Details about the dataset:** The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

**Pregnancies:** Number of times pregnant
**Glucose:** Plasma glucose concentration a 2 hours in an oral glucose tolerance test
**BloodPressure:** Diastolic blood pressure (mm Hg)
**SkinThickness:** Triceps skin fold thickness (mm)
**Insulin:** 2-Hour serum insulin (mu U/ml)
**BMI:** Body mass index (weight in kg/(height in m)^2)
**DiabetesPedigreeFunction:** Diabetes pedigree function
**Age:** Age (years)
**Outcome:** Class variable ( 0 - 1)

------

**Number of Observation Units:** 768
**Variable Number:** 9

Result; The model with the highest score after hyper parameter optimization was LGBM with 0.90 cross validation score.

---

# REPORT

**The aim of this study** was to create classification models for the diabetes data set and to predict whether a person is sick by establishing models and to obtain maximum validation scores in the established models. Here the steps;

**Exploratory Data Analysis:** The data set's structural data were checked. The types of variables in the dataset were examined. Size information of the dataset was accessed. The 0 values in the data set are missing values. Primarily these 0 values were replaced with NaN values. Descriptive statistics of the data set were examined.

**Data Preprocessing section;** The NaN values missing observations were filled with the median values of whether each variable was diabetic or not. The outliers were determined by LOF and dropped.

**In model building;** first, the base model was create and the test results were checked. Then categorical variables were edited and new features were added to the model.

**During Model Building;** Logistic Regression, KNN, CART, Random Forests, GBM, XGBoost, LightGBM like using machine learning models Cross Validation Score were calculated.

**According to test results;** GBM, XGBoost, LightGBM hyperparameter optimizations optimized to increase Cross Validation value.

**The model with the highest score after Hyper Parameter optimization was LGBM with 0.90 cross validation score**

![](https://www.oguzerdogan.com/wp-content/uploads/2020/11/results___78_0.png)

# Files

- *diabetes-prediction-with-classification-models.ipynb* - Project Notebook

- *diabetes.csv* - Dataset of project

  

## Libraries Used

```
pandas
numpy
seaborn
matplotlib
plotly
sklearn
lightgbm
xgboost
```

## Author

- Oğuz Han Erdoğan - [oguzerdo](https://github.com/oguzerdo)
