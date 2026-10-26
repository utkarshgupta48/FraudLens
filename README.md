# FraudLens — Credit Card Fraud Detection

A machine learning-powered dashboard for detecting potentially fraudulent credit card transactions, analyzing transaction patterns, and comparing machine learning models.

## Overview

FraudLens is a Python and Streamlit project that explores how machine learning can help identify suspicious credit card transactions. It combines multiple classification and anomaly detection techniques with an interactive dashboard for batch predictions and model analysis.

## Features

- Interactive dashboard built with Streamlit
- CSV upload for batch transaction predictions
- Multiple machine learning models for comparison
- Transaction analysis and visualizations
- Model evaluation using precision, recall, F1-score, and confusion matrices when actual labels are available
- Downloadable prediction results

## Machine Learning Models

- Logistic Regression
- Random Forest
- XGBoost
- Isolation Forest

## Technology Stack

- **Language:** Python
- **Dashboard:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Visualization:** Matplotlib, Seaborn

## Dataset

The project uses the Credit Card Fraud Detection dataset, which contains anonymized transaction features and a binary target indicating legitimate or fraudulent transactions.

The dataset is highly imbalanced, making fraud-specific metrics such as precision and recall important when evaluating model performance.

**Note:** The dataset is not included in this repository. Obtain it from the original dataset source and place the CSV at the path expected by the project.

## Installation and Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/utkarshgupta48/FraudLens.git
   cd FraudLens
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   ```

   Windows PowerShell:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Prepare the required dataset and trained model files. These are excluded from the repository and must be generated or obtained separately.

5. Launch the dashboard:

   ```bash
   python -m streamlit run dashboard.py
   ```

## Project Structure

```text
FraudLens/
├── dashboard.py
├── project.py
├── test_saved_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Limitations

- Predictions depend on the quality and format of the input data.
- Trained models and the dataset are not included in the repository.
- This project is intended for learning and demonstration, not production banking decisions.

## Future Improvements

- Simplify model setup for new users
- Add more model evaluation visualizations
- Improve dashboard usability
- Explore deployment as a public web application

## Author

Developed as a machine learning and data science portfolio project.

## License

No license has been specified yet.