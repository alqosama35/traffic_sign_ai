from sklearn.metrics import accuracy_score, classification_report

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return acc, report