# Diabetes Prediction with Machine Learning
![banner](/images/diabetes.png)

# Business Problem

Develop a machine learning model that can predict whether people have diabetes when their characteristics are specified

This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases.

The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are **females at least 21 years old of Pima Indian heritage.**

# Dataset Info

The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

**Number of Observation Units:** 768

**Variable Number:** 9

| Feature | Definition |
| --- | --- |
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration a 2 hours in an oral glucose tolerance test |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index (weight in kg/(height in m)^2) |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age (years) |
| Outcome | Class variable ( 0 - 1) |

# Files

- [*diabetes-prediction.ipynb*](https://github.com/oguzerdo/diabetes-prediction/blob/main/diabetes_prediction.ipynb) - Project Notebook
- [*helpers.py*](https://github.com/oguzerdo/diabetes-prediction/blob/main/helpers.py) - Functions Script
- [*outputs-diabetes_rf_model.pkl*](https://github.com/oguzerdo/diabetes-prediction/blob/main/outputs/diabetes_rf_model) - Model Object


# Requirements

```
joblib==1.1.0
lightgbm==3.1.1
matplotlib==3.5.2
numpy==1.22.3
pandas==1.4.4
scikit_learn==1.1.2
seaborn==0.11.2
xgboost==1.5.0
```

## Author

- Oğuz Erdoğan - [oguzerdo](www.oguzerdogan.com)
