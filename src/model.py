from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

def model_predict(pipe, X_train, y_train, X_test):
    pipe.fit(X_train, y_train)
    preds= pipe.predict(X_test)
    return preds

def hyper_parameter_tune(pipe, X_train, y_train ):
    params= { "classifier__n_estimators": randint(100, 500),
         "classifier__max_depth": randint(5, 20),
         "classifier__min_samples_split": randint(2, 10),
         "classifier__min_samples_leaf": randint(2, 9),
         "classifier__max_features": ['sqrt', 'log2'],
         "classifier__criterion":['gini', 'entropy']
        }

    random_search= RandomizedSearchCV(pipe, param_distributions= params, n_jobs= -1, cv= 5, n_iter= 100, random_state= 7)
    random_search.fit(X_train, y_train)
    
    return random_search.best_estimator_

def predict_survival(pipe, test):
    preds= pipe.predict(test)
    return preds