import pandas as pd
import joblib
import json
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import utils
import os

def train():
    # 1. Process Data
    print("Applying data integrity fixes and feature engineering...")
    data = utils.full_pipeline('delagua_stove_data.csv')
    
    # Define features used in notebook
    features = [
        'district', 'latitude', 'longitude', 'household_size', 
        'baseline_fuel_kg_person_week', 'distance_to_market_km', 
        'elevation_m', 'market_climb_index', 'relative_remoteness'
    ]
    
    X = data[features].copy()
    y = data['low_adoption']
    
    # 2. Encode
    le = LabelEncoder()
    X['district'] = le.fit_transform(X['district'])
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 4. Train (Params matched to version2_final)
    model = HistGradientBoostingClassifier(
        max_iter=300, 
        learning_rate=0.05, 
        max_depth=5, 
        class_weight='balanced', 
        random_state=42
    )
    
    print("Fitting model...")
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc_score = roc_auc_score(y_test, y_prob)
    acc_score = accuracy_score(y_test, y_pred)
    
    # 6. Save
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    joblib.dump(model, 'models/stove_adoption_model.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    
    metadata = {
        'training_date': str(datetime.now().date()),
        'roc_auc': auc_score,
        'accuracy': acc_score,
        'features': features
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f)
    # Save the FULL cleaned dataset for the Dashboard
    data.to_csv('data/processed_stove_data.csv', index=False)
    
    # Save just a small sample for the user to use to TEST the Batch Upload tab
    data.head(20).to_csv('data/test_households_sample.csv', index=False)
    
    print(f"Success! Full processed data ({len(data)} rows) and test sample saved.")

    
    print(f"Success! Model saved with ROC-AUC: {auc_score:.4f}")

if __name__ == "__main__":
    train()