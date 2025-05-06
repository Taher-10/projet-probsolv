# simulation/model.py
import mesa
import simpy
import pandas as pd
from datetime import datetime, timedelta
from agents.donor import DonorAgent
from agents.recipient import RecipientAgent
from agents.transport import TransportAgent

class FoodRedistributionModel(mesa.Model):
    def __init__(self, scenario="urban", step_duration=3600):  # 1 step = 1 hour
        super().__init__()
        self.schedule = mesa.time.RandomActivation(self)
        self.env = simpy.Environment()
        self.donations = []
        self.demands = []
        self.current_time = datetime(2025, 4, 30)
        self.step_duration = step_duration  # Seconds per step
        
        # Load datasets
        donors_df = pd.read_csv(f"data/donors_{scenario}.csv")
        recipients_df = pd.read_csv(f"data/recipients_{scenario}.csv")
        transports_df = pd.read_csv(f"data/transports_{scenario}.csv")
        
        # Initialize donors
        for _, row in donors_df.iterrows():
            donor = DonorAgent(row["donor_id"], self, row)
            self.schedule.add(donor)
        
        # Initialize recipients
        for _, row in recipients_df.iterrows():
            recipient = RecipientAgent(row["recipient_id"], self, row)
            self.schedule.add(recipient)
        
        # Initialize transports
        for _, row in transports_df.iterrows():
            transport = TransportAgent(row["transport_id"], self, row)
            self.schedule.add(transport)
    
    def step(self):
        """Advance the simulation by one step."""
        self.current_time += timedelta(seconds=self.step_duration)
        self.schedule.step()
        self.env.run(until=self.env.now + self.step_duration)