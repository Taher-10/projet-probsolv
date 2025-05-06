from datetime import datetime

from datetime import datetime

class FoodItem:
    """Class representing a food item donated by a donor."""
    
    def __init__(self, food_id, food_type, quantity_kg, expiry_date, perishability_hours, nutritional_value):
        self.food_id = food_id
        self.food_type = food_type
        self.quantity_kg = quantity_kg
        self.expiry_date = datetime.strptime(expiry_date, "%Y-%m-%d %H:%M:%S")
        self.perishability_hours = perishability_hours
        self.nutritional_value = nutritional_value
        self.reserved = False  # Track if item is reserved for delivery
    
    def is_expired(self, current_time):
        """Check if the food item has expired based on the current simulation time."""
        return current_time >= self.expiry_date
    
    def reserve(self):
        """Mark the food item as reserved."""
        self.reserved = True
    
    def unreserve(self):
        """Mark the food item as unreserved."""
        self.reserved = False
    
    def to_dict(self):
        """Convert food item to dictionary for serialization or visualization."""
        return {
            "food_id": self.food_id,
            "food_type": self.food_type,
            "quantity_kg": self.quantity_kg,
            "expiry_date": self.expiry_date.strftime("%Y-%m-%d %H:%M:%S"),
            "perishability_hours": self.perishability_hours,
            "nutritional_value": self.nutritional_value,
            "reserved": self.reserved
        }