import os
import pickle
import json
import pandas as pd
import numpy as np
from scripts.config import models_df
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.ensemble import VotingClassifier
from scripts.config import RANDOM_STATE, CLASSIFICATION, DEBUG_NUMBER

random_state = RANDOM_STATE
classification = CLASSIFICATION
debug_number = DEBUG_NUMBER

def all_models(X, y, test_size=0.2, random_state=random_state, classification=classification, holdout=False, cv_value=10, return_=True):
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.model_selection import train_test_split, cross_val_score
    # Tum Base Modeller (Classification)

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    # Tum Base Modeller (Regression)
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    all_models = []

    if classification:
        models = [('CART', DecisionTreeClassifier(random_state=random_state)),
                  ('RF', RandomForestClassifier(random_state=random_state)),
                  ('GBM', GradientBoostingClassifier(random_state=random_state)),
                  ('XGBoost', XGBClassifier(random_state=random_state, verbosity=0)),
                  ("LightGBM", LGBMClassifier(random_state=random_state))]
        if holdout:
            for name, model in models:
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                acc_train = accuracy_score(y_train, y_pred_train)
                acc_test = accuracy_score(y_test, y_pred_test)
                values = dict(name=name, acc_train=acc_train, acc_test=acc_test)
                all_models.append(values)
        else:  # For cross validation
            for name, model in models:
                model.fit(X, y)
                CV_SCORE = cross_val_score(model, X=X, y=y, cv=cv_value)
                values = dict(name=name, CV_SCORE_STD=CV_SCORE.std(), CV_SCORE_MEAN=CV_SCORE.mean())
                all_models.append(values)
        sort_method = False
    else:  # For Regression
        models = [('CART', DecisionTreeRegressor(random_state=random_state)),
                  ('RF', RandomForestRegressor(random_state=random_state)),
                  ('GBM', GradientBoostingRegressor(random_state=random_state)),
                  ("XGBoost", XGBRegressor(random_state=random_state)),
                  ("LightGBM", LGBMRegressor(random_state=random_state))]

        if holdout:
            for name, model in models:
                model.fit(X_train, y_train)
                y_pred_test = model.predict(X_test)
                y_pred_train = model.predict(X_train)
                rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
                rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
                values = dict(name=name, RMSE_TRAIN=rmse_train, RMSE_TEST=rmse_test)
                all_models.append(values)
        else:  # For cross validation
            for name, model in models:
                model.fit(X, y)
                CV_SCORE = np.sqrt(-cross_val_score(estimator=model, X=X_train, y=y_train, cv=cv_value,
                                                    scoring="neg_mean_squared_error"))
                values = dict(name=name, CV_SCORE_STD=CV_SCORE.std(), CV_SCORE_MEAN=CV_SCORE.mean())
                all_models.append(values)

        sort_method = True

    if return_:
        all_models_df = pd.DataFrame(all_models)
        all_models_df = all_models_df.sort_values(all_models_df.columns[2], ascending=sort_method).reset_index(drop=True)
        print(all_models_df)
        return all_models_df
    else:
        all_models_df = pd.DataFrame(all_models)
        all_models_df = all_models_df.sort_values(all_models_df.columns[2], ascending=sort_method).reset_index(drop=True)
        print(all_models_df)

    del all_models

def train_model(debug=False, tuning=True, classification=classification, target_col='OUTCOME'):

    train_df = pd.read_pickle('outputs/processed_df.pkl')


    if debug:
        train_df = train_df.sample(debug_number)
        print("Debug mode is active. Running with subsample train set...", "\n")
    else:
        pass

    print("Train dataset loaded. Observation number: ", train_df.shape[0], "\n")

    y = train_df[target_col]
    X = train_df.drop([target_col], axis=1)

    # Model Report
    print("Model Report is started.")

    models = all_models(X, y, classification=classification, cv_value=5, return_=True)
    best_models = models[0:3].merge(models_df, how='left', on='name')[["name", "model", "params"]]

    if classification:
        scoring='roc_auc'
    else:
        scoring='neg_mean_squared_error'

    if tuning:
        print("Model tuning is started...")
        # Automated Hyperparameter Optimization
        print("\n########### Hyperparameter Optimization ###########\n")
        for index, row in best_models.iterrows():
            name = row['name']
            model = row['model']
            params = row['params']

            if classification:                                                         # If classification model
                print(f"########## {name} ##########")
                score = np.mean(cross_val_score(model, X, y, cv=10, scoring=scoring))  # base model rmse
                print(f"AUC: {round(score, 4)} ({name}) ")

                gs_best = GridSearchCV(model, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)  # finding best params

                final_model = model.set_params(**gs_best.best_params_)  # save best params model
                score_new = np.mean(cross_val_score(final_model, X, y, cv=10, scoring=scoring))  # base model rmse
                print(f"RMSE (After): {round(score_new, 4)} ({name}) ")

                print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

            else:                                                                                # If regression model
                print(f"########## {name} ##########")
                score = np.mean(np.sqrt(-cross_val_score(model, X, y, cv=10, scoring=scoring)))  # base model rmse
                print(f"RMSE: {round(score, 4)} ({name}) ")

                gs_best = GridSearchCV(model, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)  # finding best params

                final_model = model.set_params(**gs_best.best_params_)  # save best params model
                score_new = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring=scoring)))  # tuned model rmse score
                print(f"RMSE (After): {round(score_new, 4)} ({name}) ")

                print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

            # MODEL SAVING PART
            today = pd.to_datetime("today").strftime('%d-%m-%Y-%H.%M')
            model_info = dict(date=today, name=name, score_info=scoring,
                              score=score,
                              score_new=score_new, count=X.shape[0],
                              best_params=gs_best.best_params_)

            try:  # Save model info to JSON
                with open('outputs/model_info_data.json', 'r+') as f:
                    model_info_data = json.load(f)

                model_info_data['data'].append(model_info)

                with(open('outputs/model_info_data.json', 'w')) as f:
                    json.dump(model_info_data, f)
            except:  # If not JSON file
                print("No JSON file, JSON File is creating")
                with(open('outputs/model_info_data.json', 'w')) as f:
                    json.dump({'data': [model_info]}, f)

            # Save Models
            os.makedirs("outputs/pickles/models", exist_ok=True)
            model_dir = "outputs/pickles/models/"
            score_new_ = str(round(score_new,2)).replace('.', '')
            with open(model_dir + f'{today}-{name}-{int(score_new_)}.pkl', 'wb') as f:
                pickle.dump(final_model, f)

        voting_model = VotingClassifier(estimators=[(best_models.iloc[0]["name"], best_models.iloc[0]["model"]),
                                                   (best_models.iloc[1]["name"], best_models.iloc[1]["model"]),
                                                   (best_models.iloc[2]["name"], best_models.iloc[2]["model"])],voting='soft').fit(X, y)

        if classification:
            cv_results = cross_validate(voting_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
            print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
            print(f"F1Score: {cv_results['test_f1'].mean()}")
            print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
            voting_score = cv_results['test_roc_auc'].mean()
            voting_score_ = str(round(voting_score,2)).replace('.', '')
        else:
            voting_score = np.mean(np.sqrt(-cross_val_score(voting_model, X, y, cv=10, scoring=scoring)))

        print("\n########## Best Models ##########\n")
        print(pd.json_normalize(json.load(open("outputs/model_info_data.json", 'r'))["data"], max_level=0).sort_values('date', ascending=False)[0:3][["name", "score_new"]])
        print("\n########## Voting Score ##########\n")
        print(f"{scoring}: {round(voting_score,3)} (Voting Ensembler) ")
        voting_model.fit(X, y)

        # Save voting_model
        with open(model_dir + f'{today}-VotingModel-{voting_score_}.pkl', 'wb') as f:
            pickle.dump(voting_model, f)

        return voting_model

    else:
        print("Best model is:", best_models.iloc[0]["name"])
        return best_models.iloc[0]["model"].fit(X,y)