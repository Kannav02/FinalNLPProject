from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

# this is the file that I have made to have modular functions for evaluations

# evaluate model performance using standard classification metrics
# I have chosen accuracy, f1_score, precision, and recall as the metrics to evaluate the model
def evaluate_model(y_test, y_pred):
    """
    Evaluate a model's performance using standard classification metrics.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
    
    Returns:
        dict: Dictionary containing accuracy, f1_score, precision, and recall
    """
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }
    
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"F1 Score: {metrics['f1_score']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    
    return metrics


# I wanted to have a hugging face trainer compatible function to compute metrics
def compute_metrics_for_trainer(eval_pred):
    """
    Compute metrics function for Hugging Face Trainer.
    
    Args:
        eval_pred: EvalPrediction object with predictions and label_ids
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions),
        'precision': precision_score(labels, predictions),
        'recall': recall_score(labels, predictions)
    }