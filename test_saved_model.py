
import joblib
import pandas as pd

# Load the saved model
loaded_model = joblib.load("fraud_model.pkl")

# Load the original dataset
df = pd.read_csv("data/creditcard.csv")

# Separate features and actual labels
X = df.drop("Class", axis=1)
y = df["Class"]

# Use the same test split as before
from sklearn.model_selection import train_test_split

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Select 3 legitimate and 3 fraudulent test transactions
legitimate_rows = X_test[y_test == 0].head(3)
fraud_rows = X_test[y_test == 1].head(3)

sample_transactions = pd.concat([legitimate_rows, fraud_rows])

# Get the corresponding actual labels
actual_labels = y_test.loc[sample_transactions.index]

# Predict using the saved model
predictions = loaded_model.predict(sample_transactions)

# Compare actual and predicted results
results = pd.DataFrame({
    "Actual Class": actual_labels.to_numpy(),
    "Predicted Class": predictions
})

results["Actual Meaning"] = results["Actual Class"].map({
    0: "Legitimate",
    1: "Fraud"
})

results["Predicted Meaning"] = results["Predicted Class"].map({
    0: "Legitimate",
    1: "Potential Fraud"
})

results["Correct"] = (
    results["Actual Class"] == results["Predicted Class"]
)

print("\nPREDICTIONS: LEGITIMATE AND FRAUDULENT TRANSACTIONS")
print(results.to_string(index=False))
