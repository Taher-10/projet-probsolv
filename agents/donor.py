
import mesa
import numpy as np
from agents.food_item import FoodItem
from datetime import datetime, timedelta

class DonorAgent(mesa.Agent):
    def __init__(self, unique_id, model, donor_data, food_items_data):
        super().__init__(unique_id, model)
        self.donor_id = donor_data["donor_id"]
        self.donor_type = donor_data["donor_type"]
        self.location = (donor_data["location_x"], donor_data["location_y"])
        self.donation_timestamp = donor_data["donation_timestamp"]
        
        self.available_donations = []
        for _, row in food_items_data.iterrows():
            food_item = FoodItem(
                food_id=row["food_id"],
                food_type=row["food_type"],
                quantity_kg=row["quantity_kg"],
                expiry_date=row["expiry_date"],
                perishability_hours=row["perishability_hours"],
                nutritional_value=row["nutritional_value"]
            )
            self.available_donations.append(food_item)
        
        self.food_types = ["Vegetables", "Dairy", "Canned", "Bread"]
        self.nutritional_values = ["High", "Medium", "Low"]
        self.donation_interval = 2 if model.scenario == "weekend" else 4 if model.scenario == "urban" else 8
    
    def generate_donation(self, current_time):
        if self.model.scenario == "shortage" and self.location[0] > 30:
            return
        
        food_id = f"F{self.donor_id}{len(self.available_donations):03d}"
        food_type = np.random.choice(self.food_types, p=[0.3, 0.3, 0.2, 0.2] if self.model.scenario == "weekend" else [0.25, 0.25, 0.25, 0.25])
        quantity = np.random.uniform(1, 10)
        perishability = 24 if food_type in ["Dairy", "Vegetables"] else 168
        # Fix: Handle perishability <= 24 separately to avoid invalid randint range
        if perishability <= 24:
            expiry_date = current_time + timedelta(hours=perishability)
        else:
            expiry_date = current_time + timedelta(hours=np.random.randint(24, perishability))
        nutritional_value = np.random.choice(self.nutritional_values)
        
        food_item = FoodItem(
            food_id=food_id,
            food_type=food_type,
            quantity_kg=quantity,
            expiry_date=expiry_date.strftime("%Y-%m-%d %H:%M:%S"),
            perishability_hours=perishability,
            nutritional_value=nutritional_value
        )
        self.available_donations.append(food_item)
        self.model.donations.append(food_item.to_dict() | {"donor_id": self.donor_id, "donation_timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S")})
    
    def donation_process(self, env):
        while True:
            self.generate_donation(self.model.current_time)
            yield env.timeout(self.donation_interval)
    
    def check_undelivered(self, current_time):
        for food_item in self.available_donations:
            self.model.waste.append({
                "food_id": food_item.food_id,
                "donor_id": self.donor_id,
                "quantity_kg": food_item.quantity_kg,
                "reason": "undelivered",
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S")
            })
    
    def step(self):
        current_time = self.model.current_time
        remaining_donations = []
        for food_item in self.available_donations:
            if food_item.is_expired(current_time) and not food_item.reserved:
                self.model.waste.append({
                    "food_id": food_item.food_id,
                    "donor_id": self.donor_id,
                    "quantity_kg": food_item.quantity_kg,
                    "reason": "expired",
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S")
                })
            else:
                remaining_donations.append(food_item)
        self.available_donations = remaining_donations