"""import mesa
import pandas as pd
import simpy
import numpy as np
from math import sqrt
from datetime import datetime, timedelta
from agents.donor import DonorAgent
from agents.recipient import RecipientAgent
from agents.transport import TransportAgent
from optimizer.ga_optimizer import optimize_routes_ga
import os

class FoodRedistributionModel(mesa.Model):
    def __init__(self, scenario="urban", max_steps=72):
        super().__init__()
        self.scenario = scenario
        self.max_steps = max_steps
        self.current_time = datetime(2025, 4, 30)
        self.step_count = 0
        self.schedule = mesa.time.RandomActivation(self)
        self.donations = []
        self.demands = []
        self.deliveries = []
        self.waste = []
        self.demand_log = []  # Store demand history
        self.vehicle_assignments = {}
        
        self.env = simpy.Environment()
        
        # Ensure data/ and ml/ directories exist
        os.makedirs("data", exist_ok=True)
        os.makedirs("ml", exist_ok=True)
        
        donors_df = pd.read_csv(f"data/donors_{scenario}.csv")
        recipients_df = pd.read_csv(f"data/recipients_{scenario}.csv")
        transports_df = pd.read_csv(f"data/transports_{scenario}.csv")
        
        for donor_id, group in donors_df.groupby("donor_id"):
            donor_data = group.iloc[0]
            donor = DonorAgent(donor_id, self, donor_data, group)
            self.schedule.add(donor)
            self.env.process(donor.donation_process(self.env))
            for _, row in group.iterrows():
                self.donations.append(row.to_dict())
        
        for _, row in recipients_df.iterrows():
            recipient = RecipientAgent(row["recipient_id"], self, row)
            self.schedule.add(recipient)
            self.env.process(recipient.demand_process(self.env))
            self.demands.append(row.to_dict())
        
        for _, row in transports_df.iterrows():
            transport = TransportAgent(row["transport_id"], self, row)
            self.schedule.add(transport)
    
    def calculate_distance(self, loc1, loc2):
        return sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    def save_demand_log(self):
       """ #the demand log to a CSV file.
"""
       if self.demand_log:
            demand_log_df = pd.DataFrame(self.demand_log)
            demand_log_df.to_csv(f"data/demand_log_{self.scenario}.csv", index=False)
            print(f"Step {self.step_count}: Saved demand log to data/demand_log_{self.scenario}.csv")

    def optimize_routes(self):
        food_items = []
        donors = []
        recipients = []
        vehicles = []
        expired_items = 0
        
        for agent in self.schedule.agents:
            if isinstance(agent, DonorAgent):
                donors.append(agent)
                for item in agent.available_donations:
                    if not item.reserved:
                        if item.is_expired(self.current_time):
                            expired_items += 1
                        else:
                            food_items.append((item, agent))
            elif isinstance(agent, RecipientAgent) and agent.current_demand > 0:
                recipients.append(agent)
            elif isinstance(agent, TransportAgent) and not agent.is_busy:
                vehicles.append(agent)
        
        total_demand = sum(r.current_demand for r in recipients)
        print(f"Step {self.step_count}: {len(food_items)} food items, {len(recipients)} recipients, {len(vehicles)} vehicles, {expired_items} expired items, total demand: {total_demand:.2f} kg")
        
        if not food_items or not recipients or not vehicles:
            self.vehicle_assignments = {v.transport_id: [] for v in vehicles}
            print(f"Step {self.step_count}: No assignments (food_items={len(food_items)}, recipients={len(recipients)}, vehicles={len(vehicles)})")
            return
        
        # Use GA optimizer
        self.vehicle_assignments, total_assigned_kg = optimize_routes_ga(
            food_items,
            recipients,
            vehicles,
            self.calculate_distance,
            self.current_time
        )
        
        print(f"Step {self.step_count}: Assignments: {[(tid, len(assignments)) for tid, assignments in self.vehicle_assignments.items()]}")
        print(f"Step {self.step_count}: Total assigned kg: {total_assigned_kg:.2f}")
        print(f"Step {self.step_count}: Recipient demands: {[(r.recipient_id, r.current_demand) for r in recipients]}")
    
    def get_vehicle_assignments(self, transport_id):
        return self.vehicle_assignments.get(transport_id, [])
    
    def step(self):
        self.step_count += 1
        self.env.run(until=self.step_count)
        self.current_time += timedelta(hours=1)
        
        self.optimize_routes()
        self.schedule.step()
        
        # Save demand log every 6 steps (6 hours)
        if self.step_count % 6 == 0:
            self.save_demand_log()
        
        if self.step_count >= self.max_steps:
            self.end_simulation()
    
    def end_simulation(self):
        for agent in self.schedule.agents:
            if isinstance(agent, DonorAgent):
                agent.check_undelivered(self.current_time)
        
        # Save final demand log
        self.save_demand_log()
        
        total_donated_kg = sum(d["quantity_kg"] for d in self.donations)
        total_wasted_kg = sum(w["quantity_kg"] for w in self.waste)
        waste_rate = (total_wasted_kg / total_donated_kg) * 100 if total_donated_kg > 0 else 0
        print(f"Waste Rate: {waste_rate:.2f}% ({total_wasted_kg:.2f} kg wasted / {total_donated_kg:.2f} kg donated)")

if __name__ == "__main__":
    model = FoodRedistributionModel(scenario="shortage")
    for _ in range(72):
        model.step()
    print(f"Simulation complete. {len(model.donations)} donations, {len(model.demands)} demands, {len(model.deliveries)} deliveries, {len(model.waste)} waste events.")
    print("\nSample Waste Events:", model.waste[:2])
    print("\nSample Deliveries:", model.deliveries[:2])
"""
import mesa
import pandas as pd
import simpy
import numpy as np
from math import sqrt
from datetime import datetime, timedelta
from agents.donor import DonorAgent
from agents.recipient import RecipientAgent
from agents.transport import TransportAgent
from optimizer.ga_optimizer import optimize_routes_ga
from optimizer.aco_optimizer import optimize_routes_aco
import os

class FoodRedistributionModel(mesa.Model):
    def __init__(self, scenario="urban", max_steps=24, optimizer_type="ga"):
        super().__init__()
        self.scenario = scenario
        self.max_steps = max_steps
        self.optimizer_type = optimizer_type  # "ga" or "aco"
        self.current_time = datetime(2025, 4, 30)
        self.step_count = 0
        self.schedule = mesa.time.RandomActivation(self)
        self.donations = []
        self.demands = []
        self.deliveries = []
        self.waste = []
        self.demand_log = []
        self.vehicle_assignments = {}
        
        self.env = simpy.Environment()
        
        os.makedirs("data", exist_ok=True)
        os.makedirs("ml", exist_ok=True)
        
        donors_df = pd.read_csv(f"data/donors_{scenario}.csv")
        recipients_df = pd.read_csv(f"data/recipients_{scenario}.csv")
        transports_df = pd.read_csv(f"data/transports_{scenario}.csv")
        
        for donor_id, group in donors_df.groupby("donor_id"):
            donor_data = group.iloc[0]
            donor = DonorAgent(donor_id, self, donor_data, group)
            self.schedule.add(donor)
            self.env.process(donor.donation_process(self.env))
            for _, row in group.iterrows():
                self.donations.append(row.to_dict())
        
        for _, row in recipients_df.iterrows():
            recipient = RecipientAgent(row["recipient_id"], self, row)
            self.schedule.add(recipient)
            self.env.process(recipient.demand_process(self.env))
            self.demands.append(row.to_dict())
        
        for _, row in transports_df.iterrows():
            transport = TransportAgent(row["transport_id"], self, row)
            self.schedule.add(transport)
    
    def calculate_distance(self, loc1, loc2):
        return sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    def save_demand_log(self):
        if self.demand_log:
            demand_log_df = pd.DataFrame(self.demand_log)
            demand_log_df.to_csv(f"data/demand_log_{self.scenario}.csv", index=False)
            print(f"Step {self.step_count}: Saved demand log to data/demand_log_{self.scenario}.csv")
    
    def optimize_routes(self):
        food_items = []
        donors = []
        recipients = []
        vehicles = []
        expired_items = 0
        
        for agent in self.schedule.agents:
            if isinstance(agent, DonorAgent):
                donors.append(agent)
                for item in agent.available_donations:
                    if not item.reserved:
                        if item.is_expired(self.current_time):
                            expired_items += 1
                        else:
                            food_items.append((item, agent))
            elif isinstance(agent, RecipientAgent) and agent.current_demand > 0:
                recipients.append(agent)
            elif isinstance(agent, TransportAgent) and not agent.is_busy:
                vehicles.append(agent)
        
        total_demand = sum(r.current_demand for r in recipients)
        print(f"Step {self.step_count}: {len(food_items)} food items, {len(recipients)} recipients, {len(vehicles)} vehicles, {expired_items} expired items, total demand: {total_demand:.2f} kg")
        
        if not food_items or not recipients or not vehicles:
            self.vehicle_assignments = {v.transport_id: [] for v in vehicles}
            print(f"Step {self.step_count}: No assignments (food_items={len(food_items)}, recipients={len(recipients)}, vehicles={len(vehicles)})")
            return
        
        if self.optimizer_type == "aco":
            self.vehicle_assignments, total_assigned_kg = optimize_routes_aco(
                food_items,
                recipients,
                vehicles,
                self.calculate_distance,
                self.current_time
            )
            print(f"Step {self.step_count}: ACO assigned {total_assigned_kg:.2f} kg")
        else:
            self.vehicle_assignments, total_assigned_kg = optimize_routes_ga(
                food_items,
                recipients,
                vehicles,
                self.calculate_distance,
                self.current_time
            )
            print(f"Step {self.step_count}: GA assigned {total_assigned_kg:.2f} kg")
        
        print(f"Step {self.step_count}: Assignments: {[(tid, len(assignments)) for tid, assignments in self.vehicle_assignments.items()]}")
        print(f"Step {self.step_count}: Recipient demands: {[(r.recipient_id, r.current_demand) for r in recipients]}")
    
    def get_vehicle_assignments(self, transport_id):
        return self.vehicle_assignments.get(transport_id, [])
    
    def step(self):
        self.step_count += 1
        self.env.run(until=self.step_count)
        self.current_time += timedelta(hours=1)
        
        self.optimize_routes()
        self.schedule.step()
        
        if self.step_count % 6 == 0:
            self.save_demand_log()
        
        if self.step_count >= self.max_steps:
            self.end_simulation()
    
    def end_simulation(self):
        for agent in self.schedule.agents:
            if isinstance(agent, DonorAgent):
                agent.check_undelivered(self.current_time)
        
        self.save_demand_log()
        
        total_donated_kg = sum(d["quantity_kg"] for d in self.donations)
        total_wasted_kg = sum(w["quantity_kg"] for w in self.waste)
        waste_rate = (total_wasted_kg / total_donated_kg) * 100 if total_donated_kg > 0 else 0
        print(f"Waste Rate: {waste_rate:.2f}% ({total_wasted_kg:.2f} kg wasted / {total_donated_kg:.2f} kg donated)")

if __name__ == "__main__":
    model = FoodRedistributionModel(scenario="urban", optimizer_type="aco")
    for _ in range(24):
        model.step()
    print(f"Simulation complete. {len(model.donations)} donations, {len(model.demands)} demands, {len(model.deliveries)} deliveries, {len(model.waste)} waste events.")
    print("\nSample Waste Events:", model.waste[:2])
    print("\nSample Deliveries:", model.deliveries[:2])