import mesa
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

class RecipientAgent(mesa.Agent):
    def __init__(self, unique_id, model, data):
        super().__init__(unique_id, model)
        self.recipient_id = data["recipient_id"]
        self.recipient_type = data["recipient_type"]
        self.location = (data["location_x"], data["location_y"])
        self.demand_level_kg = data["demand_level_kg"]
        self.food_type_preference = data["food_type_preference"]
        self.priority = data["priority"]
        self.demand_timestamp = data["demand_timestamp"]
        
        self.current_demand = self.demand_level_kg
        self.fulfilled_demand = 0
        self.demand_update_interval = 2  # Update every 2 hours
        
        # Load Random Forest model if available
        self.rf_model = None
        model_path = f"ml/random_forest_demand_model_{model.scenario}.joblib"
        try:
            self.rf_model = joblib.load(model_path)
            print(f"Loaded demand prediction model for {self.recipient_id} from {model_path}")
        except FileNotFoundError:
            print(f"Warning: Demand prediction model not found at {model_path}. Using random demand updates.")
    
    def prepare_features(self, current_time):
        """Prepare features for demand prediction matching preprocess_demand_data.py."""
        timestamp = pd.to_datetime(current_time)
        features = {
            'recipient_type': pd.Categorical([self.recipient_type], categories=['Food Bank', 'Shelter', 'Community Center']).codes[0],
            'location_x': self.location[0],
            'location_y': self.location[1],
            'food_type_preference': pd.Categorical([self.food_type_preference], categories=['Vegetables', 'Dairy', 'Canned', 'Bread']).codes[0],
            'priority': {'High': 2, 'Medium': 1, 'Low': 0}[self.priority],
            'hour': timestamp.hour,
            'day_of_week': timestamp.dayofweek,
            'step_count': self.model.step_count,
            'fulfilled_demand': self.fulfilled_demand
        }
        return pd.DataFrame([features])
    
    def update_demand(self, current_time):
        if self.rf_model:
            try:
                # Prepare features and predict next demand
                X = self.prepare_features(current_time)
                predicted_demand = self.rf_model.predict(X)[0]
                
                # Validate prediction
                if predicted_demand >= 0:
                    self.current_demand = min(predicted_demand, 50)  # Cap at 50 kg
                else:
                    raise ValueError("Negative demand predicted.")
                
                print(f"Step {self.model.step_count}: {self.recipient_id} predicted demand: {self.current_demand:.2f} kg")
            except Exception as e:
                print(f"Warning: Prediction failed for {self.recipient_id} ({e}). Using random demand update.")
                # Fallback to random increase
                demand_increase = np.random.uniform(5, 15)
                self.current_demand += demand_increase
                self.current_demand = min(self.current_demand, 50)
        else:
            # No model available, use random increase
            demand_increase = np.random.uniform(5, 15)
            self.current_demand += demand_increase
            self.current_demand = min(self.current_demand, 50)
        
        # Log demand update
        self.model.demand_log.append({
            "recipient_id": self.recipient_id,
            "recipient_type": self.recipient_type,
            "location_x": self.location[0],
            "location_y": self.location[1],
            "demand_level_kg": self.current_demand,
            "fulfilled_demand": self.fulfilled_demand,
            "food_type_preference": self.food_type_preference,
            "priority": self.priority,
            "demand_timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": self.model.scenario,
            "step_count": self.model.step_count
        })
    
    def demand_process(self, env):
        while True:
            self.update_demand(self.model.current_time)
            yield env.timeout(self.demand_update_interval)
    
    def step(self):
        pass