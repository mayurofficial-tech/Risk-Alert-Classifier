[![Watch Project Video](https://img.shields.io/badge/Google%20Drive-Watch%20Project%20Video-red?style=for-the-badge&logo=google-drive)](https://drive.google.com/file/d/16GbdUE1tJ5E10fvuWhKMAEFd14VIcgdI/view?usp=drive_link)

# 🛡️ Risk Alert Classifier

### Supervised Machine Learning \| Imbalanced Classification \| Streamlit

A machine learning project that estimates whether a banking customer is
**Low Risk (0)** or **High Risk (1)** based on customer, credit, and
transaction-related information. The project focuses on identifying
high-risk customers while reducing **false negatives**---cases where a
genuinely high-risk customer is incorrectly predicted as low risk.

> **Important:** This is an educational project. Predictions should not
> be used on their own to make real lending, credit, fraud, or other
> financial decisions.

------------------------------------------------------------------------

## 📌 Project Overview

Digital banking platforms need early-warning systems to identify
customers who may default on payments or show potentially risky
behavior. Because high-risk customers may represent a small portion of
the dataset, a classifier can appear accurate while missing important
high-risk cases.

This project builds and compares classification models, tests methods
for handling class imbalance, tunes model hyperparameters, evaluates
ROC-AUC and other metrics, and provides an interactive Streamlit
interface for entering customer details and viewing a risk prediction.

### 🎯 Objectives

-   Explore and prepare the customer risk dataset.
-   Handle missing numerical and categorical values.
-   Maintain the target-class distribution during train-test splitting.
-   Build a Logistic Regression baseline.
-   Compare under-sampling, over-sampling, SMOTE, and ADASYN.
-   Train Decision Tree and Random Forest classifiers.
-   Tune tree-based models using `RandomizedSearchCV` and
    `GridSearchCV`.
-   Evaluate models using classification metrics, confusion matrices,
    and ROC curves.
-   Prioritize reducing false negatives when selecting a model.
-   Build a Streamlit app that displays predicted risk and class
    probabilities.

------------------------------------------------------------------------

## 🧰 Tech Stack

-   **Python**
-   **Pandas, NumPy** --- data manipulation
-   **Matplotlib, Seaborn** --- visualizations
-   **Scikit-learn** --- preprocessing, models, tuning, and evaluation
-   **imbalanced-learn** --- sampling techniques and imbalanced
    pipelines
-   **Joblib** --- saving and loading the trained model
-   **Streamlit** --- interactive web application
-   **Jupyter Notebook** --- experimentation and analysis

------------------------------------------------------------------------

## 📂 Project Structure

Organize the repository like this so that the notebook and app can find
their files:

``` text
Risk-Alert-Classifier/
│
├── data/
│   └── Risk_Alert_Classifier_Dataset.csv
│
├── notebooks/
│   └── train_model.ipynb
│
├── model/
│   └── model.joblib
│
├── outputs/
│   ├── input_schema.json
│   └── Evaluation.csv
│
├── screenshots/
│   ├── app-home.png
│   ├── prediction-result.png
│   └── model-evaluation.png
│
├── app.py
├── requirements.txt
└── README.md
```

**Notes** - The notebook currently expects the dataset at
`../data/Risk_Alert_Classifier_Dataset.csv` when run from the
`notebooks/` directory. - The Streamlit app expects `model/model.joblib`
and `outputs/input_schema.json` relative to the project root. - The
dataset and generated model/schema files were not included among the
files provided for this README. Add them to the repository as
appropriate before running the complete project. - Do not commit private
or sensitive customer data.

------------------------------------------------------------------------

## 🗃️ Dataset

The project brief describes data containing customer demographic
information, transaction activity indicators, and credit behavior
indicators.

-   **Target column:** `risk_status`
-   **Class `0`:** Low Risk
-   **Class `1`:** High Risk
-   **Identifier excluded from model features:** `customer_id`
-   **Date feature engineering:** `last_transaction_date` is converted
    into `day_since_last_transaction`, then the original date column is
    removed.

The notebook uses an 80/20 train-test split with `stratify=y` and
`random_state=42` to preserve class distribution in the split.

------------------------------------------------------------------------

## ⚙️ Methodology

### 1. Data preparation

-   Inspect rows, columns, data types, missing values, and class
    distribution.
-   Separate predictors from the target.
-   Convert the last transaction date into a numeric recency feature.
-   Split the dataset into training and testing sets.

### 2. Preprocessing

-   **Numerical features:** `KNNImputer(n_neighbors=5)`;
    `StandardScaler` for the Logistic Regression and sampling pipelines.
-   **Categorical features:** most-frequent imputation and
    `OneHotEncoder(handle_unknown="ignore")`.
-   Tree-based models use imputation and one-hot encoding without
    standard scaling.

Preprocessing is placed in pipelines so transformations are fitted on
the training data as part of model training.

### 3. Models and imbalance handling

  -----------------------------------------------------------------------
  Approach                            Purpose
  ----------------------------------- -----------------------------------
  Logistic Regression                 Baseline classification model

  Random Under-Sampling               Reduce the majority class

  Random Over-Sampling                Increase representation of the
                                      minority class by resampling

  SMOTE                               Generate synthetic minority-class
                                      samples

  ADASYN                              Generate synthetic samples with
                                      emphasis on harder-to-learn
                                      minority examples

  Decision Tree                       Learn non-linear decision rules

  Random Forest                       Combine multiple decision trees

  RandomizedSearchCV                  Search sampled hyperparameter
                                      combinations

  GridSearchCV                        Search a defined parameter grid for
                                      further tuning
  -----------------------------------------------------------------------

Sampling is applied within an imbalanced-learn pipeline during model
fitting, rather than to the held-out test set.

### 4. Evaluation

The notebook evaluates models with:

-   Accuracy
-   High-risk precision
-   High-risk recall
-   High-risk F1-score
-   Confusion matrix
-   ROC curve and ROC-AUC
-   True negatives (TN), false positives (FP), false negatives (FN), and
    true positives (TP)

**Business priority:** false negatives matter because a high-risk
customer predicted as low risk may go undetected. The notebook ranks
candidate models by fewer false negatives, then higher high-risk recall,
then higher ROC-AUC.

------------------------------------------------------------------------

## 🚀 Getting Started

### 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Risk-Alert-Classifier
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your actual repository URL.

### 2. Create and activate a virtual environment (recommended)

**Windows**

``` bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Create a `requirements.txt` file with:

``` text
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
joblib
streamlit
jupyter
```

Then install:

``` bash
python -m pip install -r requirements.txt
```

### 4. Add the dataset and generate model artifacts

1.  Place `Risk_Alert_Classifier_Dataset.csv` in the `data/` directory.
2.  Open `notebooks/train_model.ipynb`.
3.  Run the notebook from top to bottom after checking the path and
    syntax notes below.
4.  Confirm the notebook creates:
    -   `model/model.joblib`
    -   `outputs/input_schema.json`
    -   `outputs/Evaluation.csv`

### 5. Launch the Streamlit app

From the project root, run:

``` bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) to
open in your browser.

------------------------------------------------------------------------

## 🖥️ Streamlit App

The app: 1. Loads the trained model from `model/model.joblib`. 2. Loads
feature definitions and input choices from `outputs/input_schema.json`.
3. Creates numeric inputs and categorical dropdowns dynamically. 4.
Predicts the high-risk probability when **Predict Risk** is submitted.
5. Displays the high-risk and low-risk probabilities and the predicted
risk class.

### App screenshots

Add screenshots to the `screenshots/` folder and uncomment/update these
examples:

``` markdown
![Streamlit app home screen](screenshots/app-home.png)
![Risk prediction result](screenshots/prediction-result.png)
![Model evaluation](screenshots/model-evaluation.png)
```

------------------------------------------------------------------------

## 📊 Results and Interpretation

The notebook includes comparison tables and ROC analysis. Use the
generated `outputs/Evaluation.csv` and notebook plots to document the
final test results.

A previously saved notebook output shows the baseline Logistic
Regression with 1.00 for accuracy, high-risk precision, recall,
F1-score, and ROC-AUC, with zero false positives and false negatives on
its displayed test split. Treat this as a result to **verify**, not a
guarantee of general performance: unusually perfect results should
prompt checks for target leakage, duplicated or synthetic patterns,
feature construction, and the integrity of the train-test separation.

The notebook's final model selection is designed to prioritize: 1. Lower
false-negative count 2. Higher high-risk recall 3. Higher ROC-AUC

Do not claim a model is production-ready solely because a test score is
high. Validate on independent, representative data and review threshold
choices before real-world use.

------------------------------------------------------------------------

## ⚠️ Before Running: Notebook Check

The uploaded notebook's Logistic Regression pipeline cell appears to
contain a syntax typo in the estimator line:

``` python
(" on(max_iter=1000,random_state=42))
```

It should be checked and corrected to:

``` python
("model", LogisticRegression(max_iter=1000, random_state=42))
```

Run **Restart Kernel and Run All** after correcting it to ensure the
notebook executes cleanly and regenerates the model and schema required
by the app.

Also confirm that the final model's class ordering is `[0, 1]` before
interpreting `predict_proba(...)[1]` as the high-risk probability. The
current app assumes the second probability belongs to class `1`.

------------------------------------------------------------------------

## 🔐 Limitations and Responsible Use

-   Predictions depend on the quality, representativeness, and freshness
    of the training data.
-   A probability is a model estimate, not a certainty or proof of
    fraud/default.
-   A fixed threshold of `0.5` is used in the app; business owners
    should validate an appropriate threshold using validation data and
    the costs of false positives and false negatives.
-   Check for data leakage, class imbalance, fairness concerns, and
    performance drift before any real-world deployment.
-   Avoid entering personal or confidential customer information into a
    public demo.

------------------------------------------------------------------------

## 🔮 Future Improvements

-   Add a validated, configurable risk threshold.
-   Display feature-level explanations where technically appropriate.
-   Add robust model monitoring and data-drift checks.
-   Validate performance with an independent dataset and report
    confidence intervals where suitable.
-   Add automated tests for input schema, model loading, and prediction
    output.
-   Deploy the app and add the verified public URL below.

------------------------------------------------------------------------

## 🌐 Project Links

-   **GitHub Repository:** [Add your repository
    URL](YOUR_GITHUB_REPOSITORY_URL)
-   **Live Streamlit App:** [Add your deployed app
    URL](YOUR_STREAMLIT_APP_URL)
-   **Project Explanation Video:** [Add your Google Drive or unlisted
    YouTube link](YOUR_VIDEO_URL)

------------------------------------------------------------------------

## 👨‍💻 Author

**Mayur Makwana**\
Data Science & Machine Learning Project

------------------------------------------------------------------------

## 📄 License

Add a license (for example, MIT) only if you intend to publish the
project under those terms.
