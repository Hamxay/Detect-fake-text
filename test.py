import pandas as pd
from sklearn.metrics import accuracy_score
from model import GPT2PPL


model = GPT2PPL()

# Specify the row number to skip
row_to_skip = 5371

# Read the CSV file excluding the problematic row
data = pd.read_csv("Training_Essay_Data.csv", skiprows=lambda x: x == row_to_skip)

data['generated'] = data['generated'].map({1: 0, 0: 1})
test_texts = data['text']
test_labels = data['generated']
predictions = []

for i, text in enumerate(test_texts):
    try:
        # Assuming you have a function 'detect' to get predictions
        results, out = model(text)
        predictions.append(results['label'])
    except Exception as e:
        print("Error occurred at index:", i)
        # Remove corresponding label from test_lab<els
        test_labels = test_labels.drop(i)
        pass

# Calculate accuracy
accuracy = accuracy_score(test_labels, predictions)
print("Accuracy:", accuracy)