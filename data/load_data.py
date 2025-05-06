import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Create data/ directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Parameters (configurable for scenarios)
scenario = "shortage"  # Options: "urban", "weekend", "shortage"
num_donors = 20 if scenario == "urban" else 30 if scenario == "weekend" else 15
num_recipients = 10 if scenario == "urban" else 10 if scenario == "weekend" else 15
num_transports = 5 if scenario == "urban" else 5 if scenario == "weekend" else 3
start_date = datetime(2025, 4, 30)
area_size = 10 if scenario == "urban" else 50  # Tighter area for urban scenario
donation_frequency = 4 if scenario == "weekend" else 2  # More donations in weekend scenario

# 1. Donors Dataset
donors = []
food_types = ["Vegetables", "Dairy", "Canned", "Bread"]
nutritional_values = ["High", "Medium", "Low"]
for i in range(num_donors):
    # Disable donors in Region C for shortage scenario (x > 30)
    location_x = np.random.uniform(0, area_size)
    if scenario == "shortage" and location_x > 30:
        continue
    donor_type = np.random.choice(["Supermarket", "Restaurant", "Household"], p=[0.4, 0.4, 0.2])
    location_y = np.random.uniform(0, area_size)
    donation_time = start_date + timedelta(hours=np.random.randint(0, 24))
    
    # Adjust donation frequency for scenarios
    num_donations = np.random.randint(1, donation_frequency + 1)
    for j in range(num_donations):
        food_id = f"F{i:03d}{j:03d}"
        food_type = np.random.choice(food_types, p=[0.3, 0.3, 0.2, 0.2] if scenario == "weekend" else [0.25, 0.25, 0.25, 0.25])
        quantity = np.random.uniform(1, 10)  # kg
        perishability = 24 if food_type in ["Dairy", "Vegetables"] else 168  # hours
        expiry_date = donation_time + timedelta(hours=np.random.randint(12, perishability))
        
        # Validate data
        if quantity <= 0:
            quantity = 1.0
        if expiry_date <= donation_time:
            expiry_date = donation_time + timedelta(hours=12)
        
        donors.append({
            "donor_id": f"D{i:03d}",
            "donor_type": donor_type,
            "location_x": location_x,
            "location_y": location_y,
            "food_id": food_id,
            "food_type": food_type,
            "quantity_kg": quantity,
            "expiry_date": expiry_date.strftime("%Y-%m-%d %H:%M:%S"),
            "perishability_hours": perishability,
            "nutritional_value": np.random.choice(nutritional_values),
            "donation_timestamp": donation_time.strftime("%Y-%m-%d %H:%M:%S")
        })

donors_df = pd.DataFrame(donors)
donors_df.to_csv(f"data/donors_{scenario}.csv", index=False)
with open(f"data/donors_{scenario}.json", "w") as f:
    json.dump(donors, f, indent=2)

# 2. Recipients Dataset (unchanged)
recipients = []
recipient_types = ["Food Bank", "Shelter", "Community Center"]
priorities = ["High", "Medium", "Low"]
for i in range(num_recipients):
    recipient_type = np.random.choice(recipient_types)
    location = (np.random.uniform(30, area_size), np.random.uniform(0, area_size)) if scenario == "shortage" else (np.random.uniform(0, area_size), np.random.uniform(0, area_size))
    demand_level = np.random.uniform(10, 30) if scenario == "shortage" else np.random.uniform(5, 20)
    food_preference = np.random.choice(food_types)
    priority = np.random.choice(priorities, p=[0.5, 0.3, 0.2] if scenario == "shortage" else [0.3, 0.4, 0.3])
    demand_time = start_date + timedelta(hours=np.random.randint(0, 24))
    recipients.append({
        "recipient_id": f"R{i:03d}",
        "recipient_type": recipient_type,
        "location_x": location[0],
        "location_y": location[1],
        "demand_level_kg": demand_level,
        "food_type_preference": food_preference,
        "priority": priority,
        "demand_timestamp": demand_time.strftime("%Y-%m-%d %H:%M:%S")
    })

recipients_df = pd.DataFrame(recipients)
recipients_df.to_csv(f"data/recipients_{scenario}.csv", index=False)
with open(f"data/recipients_{scenario}.json", "w") as f:
    json.dump(recipients, f, indent=2)

# 3. Transports Dataset (unchanged)
transports = []
vehicle_types = ["Van", "Truck"]
for i in range(num_transports):
    vehicle_type = np.random.choice(vehicle_types)
    capacity = 100 if vehicle_type == "Truck" else 50
    location = (np.random.uniform(0, area_size), np.random.uniform(0, area_size))
    fuel_efficiency = np.random.uniform(8, 12)
    speed = np.random.uniform(40, 60)
    availability_time = start_date + timedelta(hours=np.random.randint(0, 12))
    transports.append({
        "transport_id": f"T{i:03d}",
        "vehicle_type": vehicle_type,
        "capacity_kg": capacity,
        "location_x": location[0],
        "location_y": location[1],
        "fuel_efficiency_km_per_l": fuel_efficiency,
        "speed_km_per_h": speed,
        "availability_timestamp": availability_time.strftime("%Y-%m-%d %H:%M:%S")
    })

transports_df = pd.DataFrame(transports)
transports_df.to_csv(f"data/transports_{scenario}.csv", index=False)
with open(f"data/transports_{scenario}.json", "w") as f:
    json.dump(transports, f, indent=2)

print(f"Datasets generated for {scenario} scenario in data/ directory: donors_{scenario}.csv/json, recipients_{scenario}.csv/json, transports_{scenario}.csv/json")