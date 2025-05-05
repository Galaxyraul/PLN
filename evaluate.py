import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

def evaluate_folder(predictions_folder, true_labels_path,true_label,label_col='label'):
    # Load true labels
    true_df = pd.read_csv(true_labels_path)
    y_true = true_df[true_label].tolist()

    results = []

    for filename in os.listdir(predictions_folder):
        if filename.endswith('.csv'):
            pred_path = os.path.join(predictions_folder, filename)
            pred_df = pd.read_csv(pred_path)

            if len(pred_df) != len(y_true):
                print(f"⚠️ Skipping {filename}: length mismatch")
                continue

            y_pred = pred_df[label_col].tolist()

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')

            results.append({
                'file': filename,
                'accuracy': acc,
                'f1_weighted': f1,
            })

    return pd.DataFrame(results).sort_values(by='f1_weighted', ascending=False)

# Example usage:
if __name__ == "__main__":

    results_df = evaluate_folder(
        predictions_folder='predictions/ideology_binary',
        true_labels_path='true.csv',
        true_label='ideology_binary'
    )
    print(results_df)
    results_df.to_csv('evaluations/ideology_binary.csv', index=False)

    results_df = evaluate_folder(
        predictions_folder='predictions/ideology_multi',
        true_labels_path='true.csv',
        true_label='ideology_multiclass'
    )
    print(results_df)
    results_df.to_csv('evaluations/ideology_multi.csv', index=False)

    results_df = evaluate_folder(
        predictions_folder='predictions/profession',
        true_labels_path='true.csv',
        true_label='profession'
    )
    print(results_df)
    results_df.to_csv('evaluations/profession.csv', index=False)