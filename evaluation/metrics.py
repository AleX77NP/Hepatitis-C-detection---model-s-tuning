import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score

def calculate_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average=None)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    conf_mat = confusion_matrix(y_true, y_pred)
    
    return {
        'f1_score': f1,
        'accuracy': acc,
        "bal_accuracy": bal_acc,
        'precision': prec,
        'recall': recall,
        'confusion_matrix': conf_mat 
    }
    

