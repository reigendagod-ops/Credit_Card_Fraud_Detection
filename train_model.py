import os
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc

# Import the data generator
from data_generator import generate_synthetic_data

def train_fraud_detector(data_path='credit_card_transactions.csv', model_output_path='model_assets.pkl'):
    # 1. Load or Generate Dataset
    if not os.path.exists(data_path):
        print(f"Dataset {data_path} not found. Generating new synthetic dataset...")
        df = generate_synthetic_data(50000)
        df.to_csv(data_path, index=False)
    else:
        print(f"Loading existing dataset from {data_path}...")
        df = pd.read_csv(data_path)

    # 2. Separate Features and Target
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']

    # 3. Define Preprocessing Pipeline
    numerical_features = ['distance_from_home', 'purchase_amount', 'hour_of_day']
    categorical_features = ['category']
    binary_features = ['online_order', 'use_chip', 'use_pin']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough' # Leaves binary_features as is
    )

    # 4. Split Data (Stratified split to maintain fraud ratio in train and test sets)
    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Fit Preprocessor and Transform Features
    print("Preprocessing features...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_categories = list(cat_encoder.get_feature_names_out(categorical_features))
    feature_names = numerical_features + encoded_categories + binary_features

    # Convert processed arrays back to DataFrame for easier inspection
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

    # 6. Train Random Forest Classifier
    # Since fraud is highly imbalanced, we use class_weight='balanced'
    print("Training Random Forest Classifier (handling class imbalance)...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_processed, y_train)

    # 7. Evaluate Model Performance
    print("Evaluating model...")
    y_pred = model.predict(X_test_processed)
    y_prob = model.predict_proba(X_test_processed)[:, 1]

    # Calculate metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    auc_roc = roc_auc_score(y_test, y_prob)

    # Precision-Recall AUC
    precisions, recalls, _ = precision_recall_curve(y_test, y_prob)
    auc_pr = auc(recalls, precisions)

    print("\n--- Model Evaluation Results ---")
    print(f"ROC-AUC: {auc_roc:.4f}")
    print(f"PR-AUC:  {auc_pr:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 8. Extract Feature Importance
    importances = model.feature_importances_
    feature_importance_dict = dict(zip(feature_names, [float(val) for val in importances]))
    # Sort feature importances
    sorted_importance = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))

    # 9. Save all assets into a single file for deployment
    print(f"Saving model assets to {model_output_path}...")
    assets = {
        'model': model,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'binary_features': binary_features,
        'metrics': {
            'accuracy': report['accuracy'],
            'precision_fraud': report['1']['precision'],
            'recall_fraud': report['1']['recall'],
            'f1_fraud': report['1']['f1-score'],
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'confusion_matrix': conf_matrix
        },
        'feature_importances': sorted_importance
    }

    with open(model_output_path, 'wb') as f:
        pickle.dump(assets, f)

    print("Model assets saved successfully.")

if __name__ == '__main__':
    train_fraud_detector()
