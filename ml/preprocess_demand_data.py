"""preprocess_demand_data.py"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create ml/ directory if it doesn't exist
os.makedirs("ml", exist_ok=True)

def preprocess_demand_data(scenario="urban", output_dir="ml"):
    """
    Preprocess demand log data for demand prediction model.
    
    Args:
        scenario (str): Simulation scenario (e.g., "urban", "weekend", "shortage").
        output_dir (str): Directory to save processed data.
    
    Returns:
        None. Saves processed data to CSV files in output_dir.
    """
    # Load demand log
    demand_log_path = f"data/demand_log_{scenario}.csv"
    if not os.path.exists(demand_log_path):
        raise FileNotFoundError(f"Demand log file {demand_log_path} not found. Run the simulation first.")
    
    df = pd.read_csv(demand_log_path)
    
    # Convert timestamp to datetime
    df['demand_timestamp'] = pd.to_datetime(df['demand_timestamp'])
    
    # Feature engineering
    # Time-based features
    df['hour'] = df['demand_timestamp'].dt.hour
    df['day_of_week'] = df['demand_timestamp'].dt.dayofweek
    df['step_count'] = df['step_count'].astype(int)
    
    # Encode categorical variables
    df['recipient_type'] = df['recipient_type'].astype('category').cat.codes
    df['food_type_preference'] = df['food_type_preference'].astype('category').cat.codes
    df['priority'] = df['priority'].map({'High': 2, 'Medium': 1, 'Low': 0})
    
    # Spatial features
    df['location_x'] = df['location_x'].astype(float)
    df['location_y'] = df['location_y'].astype(float)
    
    # Target variable: Next demand level (shifted demand_level_kg for each recipient)
    df['next_demand'] = df.groupby('recipient_id')['demand_level_kg'].shift(-1)
    
    # Drop rows with missing target (last record for each recipient)
    df = df.dropna(subset=['next_demand'])
    
    # Select features and target
    features = [
        'recipient_type', 'location_x', 'location_y', 'food_type_preference',
        'priority', 'hour', 'day_of_week', 'step_count', 'fulfilled_demand'
    ]
    X = df[features]
    y = df['next_demand']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save processed data
    X_train.to_csv(f"{output_dir}/X_train_{scenario}.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test_{scenario}.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train_{scenario}.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test_{scenario}.csv", index=False)
    
    print(f"Processed demand data for {scenario} scenario. Saved to {output_dir}/")
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    print(f"Features: {features}")

if __name__ == "__main__":
    preprocess_demand_data(scenario="shortage")