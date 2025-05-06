"""train_demand_model.py"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
np.random.seed(42)

def train_demand_model(scenario="urban", input_dir="ml", output_dir="ml"):
    """
    Train a Random Forest model to predict recipient demand.
    
    Args:
        scenario (str): Simulation scenario (e.g., "urban", "weekend", "shortage").
        input_dir (str): Directory containing preprocessed data.
        output_dir (str): Directory to save model and plot.
    
    Returns:
        None. Saves trained model and evaluation plot to output_dir.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load preprocessed data
    try:
        X_train = pd.read_csv(f"{input_dir}/X_train_{scenario}.csv")
        X_test = pd.read_csv(f"{input_dir}/X_test_{scenario}.csv")
        y_train = pd.read_csv(f"{input_dir}/y_train_{scenario}.csv")
        y_test = pd.read_csv(f"{input_dir}/y_test_{scenario}.csv")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Preprocessed data files not found in {input_dir}. Run preprocess_demand_data.py first.")
    
    # Convert y_train and y_test to 1D arrays
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    
    # Initialize and train Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate model
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"Training MAE: {train_mae:.2f} kg")
    print(f"Test MAE: {test_mae:.2f} kg")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Save the trained model
    model_path = f"{output_dir}/random_forest_demand_model_{scenario}.joblib"
    joblib.dump(model, model_path)
    print(f"Trained model saved to {model_path}")
    
    # Plot actual vs predicted demand
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5, label="Test Data")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Prediction")
    plt.xlabel("Actual Demand (kg)")
    plt.ylabel("Predicted Demand (kg)")
    plt.title(f"Demand Prediction: Random Forest ({scenario.capitalize()} Scenario)")
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_path = f"{output_dir}/demand_prediction_plot_{scenario}.png"
    plt.savefig(plot_path)
    print(f"Prediction plot saved to {plot_path}")
    plt.close()

if __name__ == "__main__":
    train_demand_model(scenario="shortage")