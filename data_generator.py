import pandas as pd
import numpy as np

def generate_synthetic_data(n_samples=50000, random_state=42):
    """
    Generates a realistic synthetic credit card transaction dataset.
    Features model human behavior patterns, while fraud is generated using
    probabilistic risk rules combined with random noise.
    """
    np.random.seed(random_state)
    
    # 1. Distance from home (km) - log-normal distribution (mostly close, few travel)
    distance_from_home = np.random.lognormal(mean=2.0, sigma=1.2, size=n_samples)
    distance_from_home = np.clip(distance_from_home, 0.1, 2000.0) # clip outliers
    
    # 2. Purchase amount ($) - log-normal distribution (mostly small, few large purchases)
    purchase_amount = np.random.lognormal(mean=3.5, sigma=1.0, size=n_samples)
    purchase_amount = np.clip(purchase_amount, 1.0, 5000.0)
    
    # 3. Categories and their relative probabilities
    categories = ['grocery', 'entertainment', 'food_dining', 'gas_transport', 'online_shopping', 'travel_lodging', 'electronics']
    category_probs = [0.30, 0.15, 0.20, 0.15, 0.10, 0.05, 0.05]
    category = np.random.choice(categories, size=n_samples, p=category_probs)
    
    # 4. Online order (binary)
    # Online shopping and travel lodging are online by default, others are mostly in-store
    online_order = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if category[i] in ['online_shopping', 'travel_lodging']:
            online_order[i] = np.random.choice([0, 1], p=[0.05, 0.95])
        else:
            online_order[i] = np.random.choice([0, 1], p=[0.85, 0.15])
            
    # 5. Chip and PIN usage (only for in-store transactions)
    use_chip = np.zeros(n_samples, dtype=int)
    use_pin = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        if online_order[i] == 0:
            # Chip vs magnetic stripe (85% chip, 15% stripe)
            use_chip[i] = np.random.choice([0, 1], p=[0.15, 0.85])
            # PIN usage (common at gas and grocery, less common at restaurants/retail)
            if category[i] in ['grocery', 'gas_transport']:
                use_pin[i] = np.random.choice([0, 1], p=[0.10, 0.90])
            else:
                use_pin[i] = np.random.choice([0, 1], p=[0.60, 0.40])
                
    # 6. Hour of day (0-23) - peak during day, low at night
    hour_probs = np.ones(24)
    hour_probs[0:6] = 0.15  # low activity at night (midnight - 6 AM)
    hour_probs[6:8] = 0.60
    hour_probs[8:22] = 1.30 # peak activity (8 AM - 10 PM)
    hour_probs[22:24] = 0.40
    hour_probs /= hour_probs.sum()
    hour_of_day = np.random.choice(range(24), size=n_samples, p=hour_probs)
    
    # 7. Generate Fraud Label (is_fraud) based on risk factors
    is_fraud = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        # Base fraud rate (0.15% - real-world credit card fraud is highly imbalanced)
        fraud_prob = 0.0015
        
        # Risk Factor 1: Physical card used very far from home (extremely suspicious)
        if online_order[i] == 0 and distance_from_home[i] > 80:
            if distance_from_home[i] > 300:
                fraud_prob += 0.80
            else:
                fraud_prob += 0.45
                
        # Risk Factor 2: High value online purchase
        if online_order[i] == 1 and purchase_amount[i] > 300:
            if purchase_amount[i] > 1000:
                fraud_prob += 0.75
            else:
                fraud_prob += 0.30
                
        # Risk Factor 3: High value transactions at night (1 AM to 5 AM) without PIN
        if (hour_of_day[i] >= 1 and hour_of_day[i] <= 5) and use_pin[i] == 0:
            if purchase_amount[i] > 150:
                fraud_prob += 0.55
            else:
                fraud_prob += 0.20
                
        # Risk Factor 4: High risk categories (electronics and travel are hot targets for fraud)
        if category[i] in ['travel_lodging', 'electronics'] and purchase_amount[i] > 400:
            fraud_prob += 0.35
            
        # Risk Factor 5: Online purchase combined with moderate distance (billing vs shipping mismatch)
        if online_order[i] == 1 and distance_from_home[i] > 250:
            fraud_prob += 0.25
            
        # Risk Factor 6: Physical transaction with magnetic stripe (no chip) and moderate amount
        if online_order[i] == 0 and use_chip[i] == 0 and purchase_amount[i] > 100:
            fraud_prob += 0.15
            
        # Apply random variation to simulate noise/irregularities in fraud patterns
        final_prob = min(max(fraud_prob, 0.0), 0.99)
        is_fraud[i] = np.random.choice([0, 1], p=[1.0 - final_prob, final_prob])
        
    df = pd.DataFrame({
        'distance_from_home': np.round(distance_from_home, 2),
        'purchase_amount': np.round(purchase_amount, 2),
        'category': category,
        'online_order': online_order,
        'use_chip': use_chip,
        'use_pin': use_pin,
        'hour_of_day': hour_of_day,
        'is_fraud': is_fraud
    })
    
    return df

if __name__ == '__main__':
    print("Generating synthetic credit card transactions...")
    df = generate_synthetic_data(50000)
    df.to_csv('credit_card_transactions.csv', index=False)
    print(f"Generated {len(df)} transactions.")
    fraud_count = df['is_fraud'].sum()
    fraud_percent = (fraud_count / len(df)) * 100
    print(f"Fraudulent transactions: {fraud_count} ({fraud_percent:.3f}%)")
    print("Saved dataset to credit_card_transactions.csv")
