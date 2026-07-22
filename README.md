# CreditWise Loan Approval System

CreditWise is a machine learning project that predicts whether a loan application should be approved or rejected. The project compares multiple classification models and adjusts the final model's decision threshold to reduce incorrect loan approvals.

## Project Objective

The goal is to support the initial loan-screening process by identifying applicants who are more likely to qualify for approval before manual review.

In this project:

- `0` represents **No / Rejected**
- `1` represents **Yes / Approved**

Because approving an applicant who should be rejected may create financial risk, the project focuses on improving **precision for the approved class** and reducing false approvals.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Jupyter Notebook

## Project Workflow

1. Loaded and inspected the loan application dataset
2. Handled missing numerical and categorical values
3. Performed exploratory data analysis
4. Removed the applicant identifier
5. Encoded categorical features
6. Scaled numerical features
7. Created financial ratio features
8. Split the data into training, validation, and test sets
9. Trained and compared classification models
10. Tuned the final model's classification threshold
11. Evaluated the precision-recall tradeoff

## Feature Engineering

The following features were created to capture the applicant's financial position more clearly:

- **Total Income**  
  Applicant income plus coapplicant income

- **Loan-to-Income Ratio**  
  Loan amount relative to total income

- **Collateral Coverage**  
  Collateral value relative to loan amount

- **Savings-to-Loan Ratio**  
  Savings relative to loan amount

## Models Evaluated

- Logistic Regression
- K-Nearest Neighbors
- Gaussian Naive Bayes

Gaussian Naive Bayes produced the strongest approval precision after threshold tuning.

## Final Results

| Model | Threshold | Precision | Recall | F1 Score | Accuracy | False Approvals | False Rejections |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gaussian Naive Bayes | 0.50 | 81.0% | 79.7% | 80.3% | 88.5% | 11 | 12 |
| Gaussian Naive Bayes | 0.65 | **85.7%** | 61.0% | 71.3% | 85.5% | **6** | 23 |

Increasing the threshold from `0.50` to `0.65`:

- Increased approval precision from **81.0% to 85.7%**
- Reduced false approvals from **11 to 6**
- Reduced recall from **79.7% to 61.0%**
- Increased false rejections from **12 to 23**

The `0.65` threshold is the risk-focused option because it reduces incorrect approvals. The default `0.50` threshold provides better overall balance across precision, recall, F1 score, and accuracy.

## Confusion Matrix at the Selected Threshold

At the `0.65` threshold, the final confusion matrix was:

```text
[[135   6]
 [ 23  36]]
```

This represents:

- 135 correctly rejected applications
- 6 incorrectly approved applications
- 23 incorrectly rejected applications
- 36 correctly approved applications

## Repository Structure

```text
CreditWise-Loan-System/
├── CreditWise.ipynb
├── loan_approval_data.csv
├── README.md
└── .gitignore
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/BhushanSah/CreditWise-Loan-System.git
cd CreditWise-Loan-System
```

### 2. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

Open `CreditWise.ipynb` and run the cells from top to bottom.

## Key Takeaway

Increasing precision made the model more conservative. It approved fewer risky applications, but it also rejected more applicants who should have been approved. This demonstrates why a loan approval system should select its classification threshold based on the institution's risk policy rather than relying on accuracy alone.

## Future Improvements

- Move preprocessing into a scikit-learn pipeline
- Tune model hyperparameters with cross-validation
- Evaluate Random Forest and Gradient Boosting models
- Add probability calibration
- Compare thresholds using business costs
- Build a web interface for individual loan predictions

## Author

**Bhushan Sah**

- [GitHub](https://github.com/BhushanSah)
- [LinkedIn](https://www.linkedin.com/in/bhushan-sah)
