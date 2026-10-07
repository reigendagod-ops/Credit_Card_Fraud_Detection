# 🛡️ GuardianAI: Credit Card Fraud Detection System

GuardianAI is an end-to-end Machine Learning credit card fraud detection system designed as an AI/ML educational and practical laboratory project. The project demonstrates the entire machine learning pipeline: synthetic data generation, exploratory data analysis (EDA), model training (handling extreme class imbalance), real-time interactive prediction, and streaming simulation of live financial transactions.

---

## 🚀 Quick Start Guide

### Prerequisites
Make sure you have Python 3.8+ installed. You can check your version by running:
```bash
py --version
```

### Installation & Setup

1. **Navigate to the Project Directory**:
   ```bash
   cd C:/Users/44611152/.gemini/antigravity/scratch/credit-card-fraud-detection
   ```

2. **Create a Virtual Environment**:
   ```bash
   py -m venv venv
   ```

3. **Activate the Virtual Environment**:
   *   **Windows (PowerShell)**:
       ```powershell
       .\venv\Scripts\activate
       ```
   *   **Windows (Command Prompt)**:
       ```cmd
       .\venv\Scripts\activate.bat
       ```
   *   **macOS/Linux**:
       ```bash
       source venv/bin/activate
       ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Train the Machine Learning Model**:
   Execute the training pipeline to generate the synthetic transaction dataset, preprocess the features, train the Random Forest classifier, and serialize model assets.
   ```bash
   python train_model.py
   ```

6. **Launch the Interactive Web Dashboard**:
   Run the Streamlit application to explore the dashboard.
   ```bash
   python -m streamlit run app.py
   ```
   A browser window should open automatically at `http://localhost:8501`.

---

## 📁 Project Structure

```
credit-card-fraud-detection/
├── requirements.txt         # Project package dependencies
├── data_generator.py        # Generates realistic synthetic transaction logs
├── train_model.py          # Preprocessing and ML model training pipeline
├── app.py                  # Streamlit web dashboard application
├── README.md               # Setup and documentation
├── credit_card_transactions.csv  # Generated dataset (created upon training)
└── model_assets.pkl        # Serialized model, preprocessor, and metrics
```

---

## 🛠️ How It Works (AI/ML Pipeline Explainer)

### 1. Synthetic Data Generation (`data_generator.py`)
Credit card transaction logs are highly sensitive, making public datasets scarce and hard to use. GuardianAI solves this by generating **realistic synthetic transactions** based on human spending patterns:
*   **Log-Normal Distributions**: Transaction amounts and distances from home are generated using log-normal curves (the majority are small, everyday expenses near home; a few are large purchases or travel).
*   **Behavioral Rules**: We model logical dependencies (e.g., online orders don't use card chips or PIN codes; physical transactions at grocery stores/gas stations heavily utilize PINs).
*   **Fraud Probabilities**: Fraud label `is_fraud` is assigned based on combinations of risk factors (e.g., physical transaction very far from home, high-value online order, midnight transactions without a PIN, high-risk categories like electronics/travel). Random noise is added to prevent simple deterministic classification.

### 2. Feature Engineering & Preprocessing (`train_model.py`)
Features must be scaled and encoded before model training:
*   **Standard Scaling**: Numerical columns (`purchase_amount`, `distance_from_home`, `hour_of_day`) are scaled to have a mean of 0 and a variance of 1.
*   **One-Hot Encoding**: The categorical column (`category`) is converted into separate binary columns (e.g., `category_grocery`, `category_travel_lodging`).
*   **Binary Pass-through**: Columns like `online_order`, `use_chip`, and `use_pin` bypass processing since they are already in binary format (0/1).

```
Raw Data  ──►  Column Transformer  ──►  StandardScaler (Numerical)
                                    ──►  OneHotEncoder (Categorical)  ──► Processed Matrix
                                    ──►  Pass-through (Binary)
```

### 3. Model Training & Class Imbalance
*   **The Problem**: Real-world fraud is extremely rare (e.g., 0.15% fraud rate). An ML algorithm trying to maximize accuracy would default to classifying every transaction as legitimate, achieving 99.85% accuracy while catching 0% of the actual fraud.
*   **The Classifier**: We train a **Random Forest Classifier** containing 100 Decision Trees.
*   **Class Weighting**: The model is trained with `class_weight='balanced'`. This assigns a higher weight to minority (fraudulent) errors, forcing the model to optimize heavily on detecting fraud.

### 4. Metrics: Precision vs. Recall
We evaluate model success using two critical metrics:
*   **Precision (Legitimate Flagging)**: "Out of all flagged transactions, how many were actually fraud?" Low precision causes high false alarms, annoying customers.
*   **Recall (Fraud Detection Rate)**: "Out of all actual fraud transactions, how many did the system catch?" Low recall allows real fraud to slip past the security gate.

---

## 🎨 Web App Features

1.  **System Overview Dashboard**:
    Provides high-level KPIs showing total transactions, fraud rate, and average amounts. Includes interactive scatter plots and bar charts to visually analyze fraud correlations.
2.  **Real-Time Predictor**:
    Lets you input transaction variables manually via sliders and toggle switches to test the model's response instantly on individual transactions.
3.  **Live Transaction Stream Simulator**:
    Simulates a live credit card gateway. It screens transactions in real-time, displays an updating ledger showing flagged vs approved orders, and plots cumulative trends.
4.  **AI/ML Explainer**:
    An educational module explaining feature engineering, class imbalance, precision, recall, and features importance rankings.
