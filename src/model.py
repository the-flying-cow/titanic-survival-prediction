from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def model_predict(X_train, y_train, X_test):
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train,y_train)
    preds =  log_model.predict(X_test)
    return preds


def final_report(y_test, preds):

    print("On the training dataset, following are the results:\n")
    print(f"Accuracy: {accuracy_score(y_test, preds)}")
    print("Confusion Matrix:\n",confusion_matrix(y_test, preds))
    print("Classification Report:\n",classification_report(y_test, preds))
