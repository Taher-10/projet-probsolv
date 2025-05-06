import mesa
from math import sqrt

class TransportAgent(mesa.Agent):
    def __init__(self, unique_id, model, data):
        super().__init__(unique_id, model)
        self.transport_id = data["transport_id"]
        self.vehicle_type = data["vehicle_type"]
        self.capacity_kg = data["capacity_kg"]
        self.location = (data["location_x"], data["location_y"])
        self.fuel_efficiency = data["fuel_efficiency_km_per_l"]
        self.speed = data["speed_km_per_h"]
        self.availability_timestamp = data["availability_timestamp"]
        
        self.current_load_kg = 0
        self.current_route = []
        self.is_busy = False
    
    def calculate_distance(self, loc1, loc2):
        return sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    def delivery_process(self, env, donor, recipient, food_item):
        current_time = self.model.current_time
        
        if food_item not in donor.available_donations or food_item.is_expired(current_time):
            if food_item.is_expired(current_time):
                self.model.waste.append({
                    "food_id": food_item.food_id,
                    "donor_id": donor.donor_id,
                    "quantity_kg": food_item.quantity_kg,
                    "reason": "expired_during_delivery",
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S")
                })
            food_item.unreserve()
            return
        
        pickup_distance = self.calculate_distance(self.location, donor.location)
        delivery_distance = self.calculate_distance(donor.location, recipient.location)
        total_distance = pickup_distance + delivery_distance
        fuel_used = total_distance / self.fuel_efficiency
        
        self.current_load_kg += food_item.quantity_kg
        self.current_route = [donor.location, recipient.location]
        self.is_busy = True
        
        donor.available_donations.remove(food_item)
        recipient.current_demand -= food_item.quantity_kg
        recipient.fulfilled_demand += food_item.quantity_kg
        if recipient.current_demand < 0:
            recipient.current_demand = 0
        
        is_timely = not food_item.is_expired(current_time)
        self.model.deliveries.append({
            "transport_id": self.transport_id,
            "donor_id": donor.donor_id,
            "recipient_id": recipient.recipient_id,
            "food_id": food_item.food_id,
            "quantity_kg": food_item.quantity_kg,
            "delivery_timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "distance_km": total_distance,
            "fuel_used_l": fuel_used,
            "is_timely": is_timely
        })
        
        travel_time_hours = total_distance / self.speed
        yield env.timeout(travel_time_hours)
        
        self.complete_delivery()
        food_item.unreserve()
    
    def complete_delivery(self):
        self.current_load_kg = 0
        self.current_route = []
        self.is_busy = False
        self.location = self.current_route[-1] if self.current_route else self.location
    """
    def step(self):
        if self.is_busy:
            return
        
        assignments = self.model.get_vehicle_assignments(self.transport_id)
        for donor, recipient, food_item in assignments:
            self.model.env.process(self.delivery_process(self.model.env, donor, recipient, food_item))"""
    def step(self):
        if self.is_busy:
            return
        
        # Get assignments for this specific transport
        assignments = self.model.get_vehicle_assignments(self.transport_id)
        
        # Process each assignment
        for donor, recipient, food_item in assignments:
            if not food_item.reserved:
                continue
                
            # Start the delivery process
            self.model.env.process(
                self.delivery_process(
                    self.model.env,
                    donor,
                    recipient,
                    food_item
                )
            )