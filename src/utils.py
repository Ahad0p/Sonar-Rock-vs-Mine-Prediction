from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from src.exception import CustomException
import sys
import warnings
import os
import pickle

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict):
    try:
        report = {}
        best_model = None
        best_score = 0
        best_model_name = ""

        for name, model in models.items():
            if name in param and param[name]:
                gs = GridSearchCV(model, param[name], cv=3, scoring="accuracy")
                gs.fit(X_train, y_train)
                model = gs.best_estimator_
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            report[name] = score

            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = name

        return {
            "scores": report,
            "best_model": best_model,
            "best_model_name": best_model_name
        }

    except Exception as e:
        raise CustomException(e, sys)

    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)