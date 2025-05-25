# FraudGuard | Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using various machine learning models. It explores the challenges posed by imbalanced datasets, a common characteristic of fraud detection problems, and demonstrates the impact of data balancing techniques on model performance.

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Dataset](#dataset)
3.  [Setup and Installation](#setup-and-installation)
4.  [Workflow](#workflow)
5.  [Methodology](#methodology)
    *   [Data Loading and Initial Exploration](#data-loading-and-initial-exploration)
    *   [Data Preprocessing](#data-preprocessing)
    *   [Modeling with Imbalanced Data](#modeling-with-imbalanced-data)
    *   [Handling Class Imbalance](#handling-class-imbalance)
    *   [Modeling with Balanced Data](#modeling-with-balanced-data)
6.  [Results and Observations](#results-and-observations)
7.  [Model Selection Considerations](#model-selection-considerations)
8.  [Conclusion](#conclusion)
9.  [How to Run](#how-to-run)

## 1. Project Overview

The primary goal of this project is to build and evaluate machine learning models capable of distinguishing fraudulent credit card transactions from legitimate ones. The notebook covers:
*   Setting up the environment and downloading the dataset from Kaggle.
*   Exploratory Data Analysis (EDA) to understand data characteristics, including class imbalance.
*   Feature scaling and data shuffling.
*   Training and evaluating several classification models on the original, imbalanced dataset.
*   Implementing a strategy to balance the dataset.
*   Training and evaluating the same models on the balanced dataset.
*   Comparing model performance and discussing the trade-offs between precision and recall.

## 2. Dataset

*   **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by mlg-ulb.
*   **Description:** The dataset contains transactions made by European cardholders in September 2013. It presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
*   **Features:**
    *   `Time`: Number of seconds elapsed between this transaction and the first transaction in the dataset.
    *   `V1` to `V28`: Principal components obtained with PCA. Due to confidentiality issues, original features are not provided.
    *   `Amount`: Transaction amount.
    *   `Class`: The target variable (1 in case of fraud, 0 otherwise).

## 3. Setup and Installation

This project is implemented in a Jupyter Notebook environment, likely run on Google Colab.

**Prerequisites:**
*   Python 3.x
*   Jupyter Notebook or Google Colab

**Key Libraries:**
*   `pandas`
*   `scikit-learn`
*   `tensorflow` (for Keras)
*   `xgboost`
*   `matplotlib` (implicitly used by `ConfusionMatrixDisplay`)
*   `zipfile`

**Kaggle API Setup (for dataset download):**
1.  Create a Kaggle account and obtain an API token (`kaggle.json`).
2.  Create a directory named `.kaggle` in your home directory (e.g., `/root/.kaggle` in Colab or `~/.kaggle/` locally).
    ```bash
    mkdir ~/.kaggle
    ```
3.  Place the `kaggle.json` file into this directory.
    ```bash
    cp kaggle.json ~/.kaggle/
    ```
4.  Set appropriate permissions for the API key file:
    ```bash
    chmod 600 ~/.kaggle/kaggle.json
    ```


## 4. Workflow

The notebook follows a structured approach:

1.  **Environment Setup:**
    *   Mounts Google Drive (if running in Colab).
    *   Configures the Kaggle API for dataset download.
2.  **Dataset Download & Loading:**
    *   Downloads the "creditcardfraud" dataset from Kaggle.
    *   Unzips and loads the `creditcard.csv` file into a pandas DataFrame.
3.  **Exploratory Data Analysis (EDA):**
    *   Displays the first few rows of the DataFrame.
    *   Checks class distribution (`Class` column), revealing significant imbalance.
    *   Generates histograms for all numerical features.
    *   Computes descriptive statistics.
4.  **Data Preprocessing:**
    *   Scales the `Amount` feature using `RobustScaler` (robust to outliers).
    *   Scales the `Time` feature using Min-Max scaling.
    *   Shuffles the DataFrame rows randomly.
5.  **Modeling on Imbalanced Data:**
    *   Splits the preprocessed (but still imbalanced) data into training, testing, and validation sets.
    *   Trains and evaluates the following models:
        *   Logistic Regression
        *   Shallow Neural Network (1 hidden layer with Batch Normalization)
        *   Random Forest Classifier
        *   Gradient Boosting Classifier
        *   Linear Support Vector Machine (LinearSVC)
        *   XGBoost Classifier
    *   Evaluation is done using classification reports and confusion matrices.
6.  **Handling Class Imbalance:**
    *   Separates fraudulent and non-fraudulent transactions.
    *   Creates a balanced dataset by undersampling the majority class (non-fraudulent transactions) to match the number of fraudulent transactions.
    *   Shuffles the balanced dataset.
7.  **Modeling on Balanced Data:**
    *   Splits the balanced dataset into new training, testing, and validation sets.
    *   Retrains and re-evaluates all the models listed above on this balanced data.
    *   Performance is again assessed using classification reports and confusion matrices on both validation and test sets.
8.  **Analysis and Conclusion:**
    *   Compares model performance on imbalanced vs. balanced datasets.
    *   Discusses the trade-offs between precision and recall, crucial for fraud detection.

## 5. Methodology

### Data Loading and Initial Exploration
The dataset `creditcard.csv` is loaded using pandas. Initial exploration includes:
*   Viewing the head of the DataFrame.
*   `df['Class'].value_counts()`: Revealed a significant class imbalance (284,315 non-fraudulent vs. 492 fraudulent transactions).
*   `df.hist()`: Visualized distributions of numerical features.
*   `df.describe()`: Provided summary statistics.

### Data Preprocessing
1.  **Feature Scaling:**
    *   The `Amount` column was scaled using `RobustScaler` to handle potential outliers effectively.
    *   The `Time` column was normalized using Min-Max scaling to bring its values within a [0, 1] range.
2.  **Data Shuffling:**
    *   The entire dataset was shuffled randomly (`df.sample(frac=1, random_state=1)`) before splitting to ensure unbiased distribution of data into train/test/validation sets.

### Modeling with Imbalanced Data
*   **Train-Test-Validation Split:** The shuffled (but still imbalanced) dataset was split:
    *   Training set: First 240,000 samples.
    *   Testing set: Next 22,000 samples.
    *   Validation set: Remaining samples.
*   **Models Trained:**
    1.  Logistic Regression
    2.  Shallow Neural Network (TensorFlow/Keras: Input -> Dense(2, relu) -> BatchNorm -> Dense(1, sigmoid))
    3.  Random Forest Classifier (`max_depth=2`)
    4.  Gradient Boosting Classifier (`n_estimators=50, learning_rate=1.0, max_depth=1`)
    5.  Linear Support Vector Machine (LinearSVC, `class_weight='balanced'`)
    6.  XGBoost Classifier (`n_estimators=1000, max_depth=10, learning_rate=0.1`)
*   **Evaluation:** Primarily using `classification_report` (precision, recall, F1-score) and `ConfusionMatrixDisplay` on the validation set.

### Handling Class Imbalance
The extreme class imbalance was addressed by creating a balanced dataset:
1.  Fraudulent transactions were isolated.
2.  An equal number of non-fraudulent transactions were randomly sampled from the majority class.
3.  These two sets were concatenated to form a balanced DataFrame (`balanced_df`).
4.  The `balanced_df` was then shuffled.

### Modeling with Balanced Data
*   **Train-Test-Validation Split (Balanced):** The shuffled balanced dataset (984 samples in total) was split:
    *   Training set (`_b` suffix): First 700 samples.
    *   Testing set (`_b` suffix): Next 142 samples.
    *   Validation set (`_b` suffix): Remaining 142 samples.
*   **Models Retrained:** All models from the imbalanced data phase were retrained on the balanced training data (`x_train_b`, `y_train_b`).
*   **Evaluation:** Performance was evaluated using `classification_report` and `ConfusionMatrixDisplay` on both the balanced validation set (`x_val_b`, `y_val_b`) and the balanced test set (`x_test_b`, `y_test_b`). The Shallow Neural Network was trained for more epochs (40) on the balanced data.

## 6. Results and Observations

*   **Imbalanced Data:**
    *   All models achieved high accuracy, largely due to correctly classifying the majority "Not Fraud" class.
    *   Performance on the "Fraud" class was generally poor, with low recall across most models. This means many fraudulent transactions were missed.
    *   LinearSVC with `class_weight='balanced'` showed improved recall for fraud but at the cost of precision.
*   **Balanced Data:**
    *   There was a significant improvement in detecting the "Fraud" class (higher recall and F1-scores for fraud).
    *   **Logistic Regression (Balanced):** Achieved an F1-score of ~0.94 for fraud on the validation set and ~0.93 on the test set.
    *   **Shallow Neural Net (Balanced):** Achieved an F1-score of ~0.90 for fraud on the validation set and ~0.89 on the test set after 40 epochs.
    *   **Random Forest (Balanced):** Achieved an F1-score of ~0.95 for fraud on the validation set and ~0.94 on the test set.
    *   **Gradient Boosting (Balanced):** Achieved an F1-score of ~0.93 for fraud on the validation set and ~0.91 on the test set.
    *   **Linear SVC (Balanced):** Achieved an F1-score of ~0.94 for fraud on the validation set and ~0.93 on the test set.
    *   **XGBoost (Balanced):** Achieved an F1-score of ~0.81 for fraud on the validation set and ~0.87 on the test set.

**Overall:**
*   Working with balanced data dramatically improves the models' ability to detect fraudulent transactions.
*   On the balanced test set, Random Forest, LinearSVC, and Logistic Regression showed very strong and balanced performance for both classes. XGBoost also performed well.
*   The Shallow Neural Network also showed good balanced performance.

## 7. Model Selection Considerations

The choice of the "best" model depends on the specific business context:

*   **Minimizing False Positives (High Precision for Fraud):** If incorrectly flagging a legitimate transaction as fraud is very costly (e.g., customer dissatisfaction), models with higher precision for the fraud class are preferred.
*   **Minimizing False Negatives (High Recall for Fraud):** If failing to detect a fraudulent transaction is very costly (e.g., financial loss), models with higher recall for the fraud class are preferred. This is often the primary concern in fraud detection.
*   **Balanced Performance (F1-Score):** The F1-score provides a balance between precision and recall. Models with high F1-scores for the fraud class are generally desirable.

For this project, after balancing the data, models like Random Forest, LinearSVC, and Logistic Regression provided excellent F1-scores for the fraud class on the test set, indicating a good balance in catching fraudulent transactions while not excessively flagging legitimate ones.

## 8. Conclusion

This project demonstrates the critical impact of class imbalance on fraud detection models. While models trained on imbalanced data achieve high overall accuracy, they often fail to effectively identify the minority (fraudulent) class. By employing a simple undersampling technique to balance the dataset, the performance of all tested models in detecting fraud improved significantly, particularly in terms of recall and F1-score for the fraud class.

For this dataset, after balancing, **Random Forest, LinearSVC, and Logistic Regression** emerged as strong-performing models, offering a good trade-off between identifying fraudulent transactions and minimizing false alarms on the balanced test data.

## 9. How to Run

1.  **Clone the repository (if applicable) or download the `.ipynb` file.**
2.  **Set up Kaggle API:**
    *   Ensure you have your `kaggle.json` API token.
    *   If using Google Colab, upload `kaggle.json`. The notebook contains commands to move it to the correct location (`/root/.kaggle/`).
    *   If running locally, place `kaggle.json` in `~/.kaggle/` (Linux/macOS) or `C:\Users\<Your-Username>\.kaggle\` (Windows).
3.  **Modify Paths (if necessary):**
    *   The notebook uses Google Drive paths (e.g., `/content/drive/MyDrive/...`). If you are running this locally or in a different Colab setup, you will need to adjust these paths for:
        *   Copying `kaggle.json` from Drive (or upload directly).
        *   The download path for the Kaggle dataset (`-p /content/drive/MyDrive/Data Analyst Projects/Credit Card Analysis`).
        *   The path for unzipping and reading `creditcard.csv`.
4.  **Install Dependencies:**
    ```bash
    pip install pandas scikit-learn tensorflow xgboost matplotlib
    ```
5.  **Run the Jupyter Notebook:** Execute the cells sequentially. The notebook handles data download, preprocessing, model training, and evaluation for both imbalanced and balanced scenarios.
