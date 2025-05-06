import pytest

from datetime import datetime, timedelta
from agents.food_item import FoodItem
from agents.recipient import RecipientAgent
import mesa
import numpy as np

from optimizer.dp_optimizer import knapsack_optimize_food_allocation

class MockModel(mesa.Model):
    """Mock model for agent initialization"""
    def __init__(self):
        super().__init__()
        self.current_time = datetime.now()
        self.scenario = "test"
        self.step_count = 0
        self.demand_log = []

class TestDPOptimizer:
    @pytest.fixture
    def current_time(self):
        return datetime.now()

    @pytest.fixture
    def mock_model(self):
        return MockModel()

    def create_recipient(self, model, recipient_data):
        """Helper to create recipients matching your actual class signature"""
        data = {
            "recipient_id": recipient_data.get("recipient_id", "R1"),
            "recipient_type": recipient_data.get("recipient_type", "Food Bank"),
            "location_x": 0,
            "location_y": 0,
            "demand_level_kg": recipient_data.get("current_demand", 10),
            "food_type_preference": recipient_data.get("food_type_preference", "Canned"),
            "priority": recipient_data.get("priority", "Medium"),
            "demand_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return RecipientAgent(recipient_data.get("recipient_id", "R1"), model, data)

    def test_high_demand_low_supply(self, current_time, mock_model):
        """Test allocation when demand far exceeds supply"""
        recipients = [
            self.create_recipient(mock_model, {
                "recipient_id": "R1",
                "recipient_type": "Food Bank",
                "priority": "High",
                "current_demand": 100,
                "food_type_preference": "Canned"
            })
        ]
        
        food_items = [
            (FoodItem(
                food_id="F1",
                food_type="Canned",
                quantity_kg=5,
                expiry_date=(current_time + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
                perishability_hours=2,
                nutritional_value="High"
            ), None)
        ]
        
        result = knapsack_optimize_food_allocation(food_items, recipients, current_time)
        allocations = result.get("R1", [])
        total_allocated = sum(item[2].quantity_kg for item in allocations)
        
        assert total_allocated > 0, "Failed to allocate scarce resources"
        assert total_allocated == 5, "Should allocate full amount when only one recipient"

    def test_priority_guarantee(self, current_time, mock_model):
        """Verify high-priority recipients get minimum allocation"""
        recipients = [
            self.create_recipient(mock_model, {
                "recipient_id": "R1",
                "recipient_type": "Food Bank",
                "priority": "High",
                "current_demand": 10,
                "food_type_preference": "Vegetables"
            }),
            self.create_recipient(mock_model, {
                "recipient_id": "R2",
                "recipient_type": "Community Center",
                "priority": "Low",
                "current_demand": 100,
                "food_type_preference": "Dairy"
            })
        ]
        
        food_items = [
            (FoodItem(
                food_id="F1",
                food_type="Vegetables",
                quantity_kg=8,
                expiry_date=(current_time + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
                perishability_hours=24,
                nutritional_value="Medium"
            ), None)
        ]
        
        result = knapsack_optimize_food_allocation(food_items, recipients, current_time)
        high_pri_alloc = sum(item[2].quantity_kg for item in result.get("R1", []))
        
        assert high_pri_alloc >= 3, "High-priority recipient should get minimum 30%"
        assert high_pri_alloc <= 8, "Should not exceed available supply"

    def test_perishable_priority(self, current_time, mock_model):
        recipients = [
            self.create_recipient(mock_model, {
                "recipient_id": "R1",
                "recipient_type": "Shelter",
                "priority": "Medium",
                "current_demand": 20,
                "food_type_preference": "Dairy"
            })
        ]

        # Create foods with different perishability
        soon_expiring = FoodItem(
            food_id="F1",
            food_type="Dairy",
            quantity_kg=5,
            expiry_date=(current_time + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
            perishability_hours=6,  # Will expire soon (low hours)
            nutritional_value="High"
        )
        
        long_expiring = FoodItem(
            food_id="F2",
            food_type="Canned",
            quantity_kg=10,
            expiry_date=(current_time + timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S"),
            perishability_hours=168,  # Expires later (high hours)
            nutritional_value="Medium"
        )

        # Put long-expiring first to test sorting
        food_items = [
            (long_expiring, None),
            (soon_expiring, None)
        ]

        result = knapsack_optimize_food_allocation(food_items, recipients, current_time)
        allocations = result.get("R1", [])

        assert len(allocations) > 0, "No allocations made"
        
        # Get all allocated food types in order
        allocated_types = [item[2].food_type for item in allocations]
        
        # Dairy should be allocated (regardless of position)
        assert "Dairy" in allocated_types, "Perishable item not allocated at all"
        
        # If both were allocated, perishable (Dairy) should come first
        if len(allocations) > 1:
            # Verify Dairy is before Canned in allocation order
            dairy_index = allocated_types.index("Dairy")
            canned_index = allocated_types.index("Canned")
            assert dairy_index < canned_index, "Perishable food not prioritized"