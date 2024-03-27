import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def compute_f1_false_positive(data):
    # Assuming 'iso_script' contains true labels and 'top_pred' contains predicted labels
    iso_codes = list(data['iso_script'])
    predicted_iso_codes = list(data['top_pred'])
    
    
    labels = list(set(iso_codes + predicted_iso_codes))
    label_to_index = {label: index for index, label in enumerate(labels)}
    iso_codes = [label_to_index[label] for label in iso_codes]
    predicted_iso_codes = [label_to_index[label] for label in predicted_iso_codes]
    
    
    # Compute F1 scores per label
    f1_scores = f1_score(iso_codes, predicted_iso_codes, average=None)
    precision_scores = precision_score(iso_codes, predicted_iso_codes, average=None)
    recall_scores = recall_score(iso_codes, predicted_iso_codes, average=None)    
    
    
    # Compute False Positive Rate per label
    confusion_mat = confusion_matrix(iso_codes, predicted_iso_codes)
    
    FP = confusion_mat.sum(axis=0) - np.diag(confusion_mat)  
    FN = confusion_mat.sum(axis=1) - np.diag(confusion_mat)
    TP = np.diag(confusion_mat)
    TN = confusion_mat.sum() - (FP + FN + TP)


    fp_rate = FP / (FP + TN)
    
    # Handle division by zero by setting FPR to 0 where actual_negatives is 0
    fp_rate = np.nan_to_num(fp_rate, nan=0.0)
    
    
    # Create DataFrame to store results
    result_df = pd.DataFrame({
        'label': labels,
        'f1_score': f1_scores,
        'precision_score': precision_scores,
        'recall_score': recall_scores,
        'false_positive_rate': fp_rate
    })
    
    return result_df

# Example data (replace this with your actual data)
data = {
    'sentence': ['This is sentence 1', 'Une autre phrase', 'Phrase numéro trois'],
    'iso_script': ['eng_Latn', 'fra_Latn', 'spa_Latn'],  # Assuming this is the true label
    'top_pred': ['eng_Latn', 'eng_Latn', 'fas_Arab']  # Predicted language labels
}

df_2 = pd.DataFrame(data)

# Compute F1, Recall, Precision scores and false positive rate
result_df = compute_f1_false_positive(df_2)

print("Result DataFrame:")
print(result_df)
