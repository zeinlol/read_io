import pandas as pd
import torch
from transformers import RobertaTokenizer

from classes import TextClassifier

# Load your pre-trained model
model_path = "text_classifier_model.pth"  # Replace with the path to your pre-trained model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = TextClassifier()
model.load_state_dict(torch.load(model_path))
model.eval()

# Load your test data from a CSV file
test_data = pd.read_csv("test.csv")

# Initialize a list to store predicted values
predicted_values = []

# Iterate through the test data
for index, row in test_data.iterrows():
    excerpt = row["excerpt"]

    # Tokenize the excerpt
    inputs = tokenizer(excerpt, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Forward pass to get predictions
    with torch.no_grad():
        output = model(**inputs)

    # Convert logits to a predicted readability score (assuming regression)
    predicted_readability = output.logit()

    # Append the predicted score to the list
    predicted_values.append(predicted_readability)

# Add the predicted values to the test data
test_data["predicted_target"] = predicted_values

# Save the results to a new CSV file
test_data.to_csv("test_results.csv", index=False)
