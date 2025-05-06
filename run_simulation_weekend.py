import mesa
import pandas as pd
import simpy
import numpy as np
from math import sqrt
from datetime import datetime, timedelta
from agents.donor import DonorAgent
from agents.recipient import RecipientAgent
from agents.transport import TransportAgent
from optimizer.dp_optimizer import knapsack_optimize_food_allocation
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
        self.demand_log = []
        self.vehicle_assignments = {}  # {transport_id: [(donor, recipient, food_item)]}
        
        self.env = simpy.Environment()
        os.makedirs("data", exist_ok=True)
        os.makedirs("ml", exist_ok=True)
        
        # Load data
        donors_df = pd.read_csv(f"data/donors_{scenario}.csv")
        recipients_df = pd.read_csv(f"data/recipients_{scenario}.csv")
        transports_df = pd.read_csv(f"data/transports_{scenario}.csv")
        
        # Initialize agents
        for donor_id, group in donors_df.groupby("donor_id"):
            donor_data = group.iloc[0]
            donor = DonorAgent(donor_id, self, donor_data, group)
            self.schedule.add(donor)
            self.env.process(donor.donation_process(self.env))
        
        for _, row in recipients_df.iterrows():
            recipient = RecipientAgent(row["recipient_id"], self, row)
            self.schedule.add(recipient)
            self.env.process(recipient.demand_process(self.env))
        
        for _, row in transports_df.iterrows():
            transport = TransportAgent(row["transport_id"], self, row)
            self.schedule.add(transport)
    
    def calculate_distance(self, loc1, loc2):
        return sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    def save_demand_log(self):
        if self.demand_log:
            pd.DataFrame(self.demand_log).to_csv(f"data/demand_log_{self.scenario}.csv", index=False)

    def get_vehicle_assignments(self, transport_id):
        """Get assignments for a specific transport vehicle"""
        return self.vehicle_assignments.get(transport_id, [])

    def optimize_routes(self):
        print(f"\n=== DP Optimizer Step {self.step_count} ===")
        print(f"Current Time: {self.current_time.strftime('%Y-%m-%d %H:%M')}")
        
        # Collect available resources
        food_items = []
        donors = []
        recipients = []
        vehicles = []
        
        # Gather data
        for agent in self.schedule.agents:
            if isinstance(agent, DonorAgent):
                donors.append(agent)
                for item in agent.available_donations:
                    if not item.reserved and not item.is_expired(self.current_time):
                        food_items.append((item, agent))
            elif isinstance(agent, RecipientAgent) and agent.current_demand > 0:
                recipients.append(agent)
            elif isinstance(agent, TransportAgent) and not agent.is_busy:
                vehicles.append(agent)
        
        # Print initial state
        print("\n[Initial State]")
        print(f"- Available food items: {len(food_items)}")
        print(f"- Recipients with demand: {len(recipients)}")
        print(f"- Available vehicles: {len(vehicles)}")
        
        if not food_items or not recipients or not vehicles:
            print("No optimization possible - missing resources")
            self.vehicle_assignments = {}
            return
        
        # DP Optimization
        print("\n[DP Optimization Starting]")
        knapsack_assignments = knapsack_optimize_food_allocation(
            food_items, recipients, self.current_time
        )
        
        # Print DP results
        print("\n[DP Allocation Results]")
        total_allocated = 0
        for recipient_id, assignments in knapsack_assignments.items():
            allocated = sum(item[2].quantity_kg for item in assignments)
            total_allocated += allocated
            recipient = next(r for r in recipients if r.recipient_id == recipient_id)
            print(f"- {recipient_id} ({recipient.recipient_type}): {allocated:.2f}kg allocated")
        
        print(f"\nTotal allocated: {total_allocated:.2f}kg")
        
        # Vehicle assignment
        print("\n[Vehicle Assignment]")
        self.vehicle_assignments = {v.transport_id: [] for v in vehicles}
        assignment_counts = {v.transport_id: 0 for v in vehicles}
        
        for recipient_id, assignments in knapsack_assignments.items():
            for donor, recipient, food_item in assignments:
                closest_vehicle = min(
                    [v for v in vehicles if not v.is_busy],
                    key=lambda v: self.calculate_distance(v.location, donor.location),
                    default=None
                )
                if closest_vehicle:
                    self.vehicle_assignments[closest_vehicle.transport_id].append(
                        (donor, recipient, food_item)
                    )
                    assignment_counts[closest_vehicle.transport_id] += 1
                    food_item.reserve()
        
        # Print vehicle assignments
        for vehicle_id, count in assignment_counts.items():
            vehicle = next(v for v in vehicles if v.transport_id == vehicle_id)
            print(f"- {vehicle_id} ({vehicle.vehicle_type}): {count} assignments")
        
        print("\n=== DP Optimization Complete ===\n")
        # Collect available resources
        food_items = []
        donors = []
        recipients = []
        vehicles = []
        
        for agent in self.schedule.agents:
            if isinstance(agent, DonorAgent):
                donors.append(agent)
                for item in agent.available_donations:
                    if not item.reserved and not item.is_expired(self.current_time):
                        food_items.append((item, agent))
            elif isinstance(agent, RecipientAgent) and agent.current_demand > 0:
                recipients.append(agent)
            elif isinstance(agent, TransportAgent) and not agent.is_busy:
                vehicles.append(agent)
        
        if not food_items or not recipients or not vehicles:
            self.vehicle_assignments = {}
            return
        
        # Knapsack optimization for food allocation
        knapsack_assignments = knapsack_optimize_food_allocation(
            food_items, recipients, self.current_time
        )
        
        # Assign to nearest available vehicles
        self.vehicle_assignments = {v.transport_id: [] for v in vehicles}
        for recipient_id, assignments in knapsack_assignments.items():
            for donor, recipient, food_item in assignments:
                # Find nearest available vehicle
                closest_vehicle = min(
                    [v for v in vehicles if not v.is_busy],
                    key=lambda v: self.calculate_distance(v.location, donor.location),
                    default=None
                )
                if closest_vehicle:
                    self.vehicle_assignments[closest_vehicle.transport_id].append(
                        (donor, recipient, food_item)
                    )
                    food_item.reserve()
    
    def step(self):
        self.step_count += 1
        self.current_time += timedelta(hours=1)
        self.env.run(until=self.step_count)
        
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
        total_donated = sum(d["quantity_kg"] for d in self.donations)
        total_wasted = sum(w["quantity_kg"] for w in self.waste)
        print(f"Final Waste Rate: {(total_wasted/total_donated)*100:.2f}%")

if __name__ == "__main__":
    model = FoodRedistributionModel(scenario="weekend")
    for _ in range(72):
        model.step()