import pandas as pd

# 1. Load the files
true_labels = pd.read_csv('true_labels_LARGE.csv') # The answer key
my_predictions = pd.read_csv('my_system_predictions.csv') # Your system's output

# 2. Merge them on the 'transaction_id' column
comparison_df = true_labels.merge(my_predictions, on='transaction_id')

# 3. (Optional) Save this merged file to inspect manually
comparison_df.to_csv('comparison_results.csv', index=False)

# 4. Create the Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report

conf_matrix = confusion_matrix(comparison_df['is_fraud'], comparison_df['Anomaly'])
print("Confusion Matrix:")
print(conf_matrix)

# 5. Print a full performance report (This is the most important part!)
print("\nDetailed Performance Report:")
print(classification_report(comparison_df['is_fraud'], comparison_df['Anomaly']))