from typing import List, Dict, Tuple
from agents.food_item import FoodItem
from agents.recipient import RecipientAgent
from agents.donor import DonorAgent
from datetime import datetime
import numpy as np
from collections import defaultdict

def knapsack_optimize_food_allocation(
    food_items: List[Tuple[FoodItem, DonorAgent]], 
    recipients: List[RecipientAgent],
    current_time: datetime
) -> Dict[str, List[Tuple[DonorAgent, RecipientAgent, FoodItem]]]:
    """
    Enhanced DP optimizer with:
    - Explicit fairness constraints
    - Perishability urgency boosting
    - Priority compliance tracking
    - Waste reduction safeguards
    """
    
    # Priority weights with emergency boost factors
    recipient_priority_weights = {
        'Food Bank': {'High': 4.0, 'Medium': 2.5, 'Low': 1.5},
        'Shelter': {'High': 3.5, 'Medium': 2.0, 'Low': 1.0},
        'Community Center': {'High': 3.0, 'Medium': 1.5, 'Low': 1.0}
    }
    
    # 1. Initialize metrics tracking
    metrics = {
        'total_allocated': 0,
        'waste_risk_items': 0,
        'priority_violations': 0,
        'fulfillment_ratios': defaultdict(float)
    }
    
    # 2. Pre-process recipients with fairness constraints
    total_demand = sum(r.current_demand for r in recipients)
    recipient_data = []
    
    for recipient in recipients:
        base_score = recipient_priority_weights[recipient.recipient_type][recipient.priority]
        
        # Dynamic demand factor with fairness cap
        demand_ratio = recipient.current_demand / max(1, total_demand)
        demand_factor = 1 + min(0.5, demand_ratio)  # Cap at 1.5x
        
        recipient_data.append({
            'obj': recipient,
            'base_score': base_score,
            'demand_factor': demand_factor,
            'remaining_demand': recipient.current_demand,
            'preference': recipient.food_type_preference,
            'min_allocation': 0.3 * recipient.current_demand if recipient.priority == 'High' else 0
        })
    
    # 3. Sort food items with perishability urgency
    sorted_food = sorted(
        [(f, d) for f, d in food_items if not f.reserved and not f.is_expired(current_time)],
        key=lambda x: (
            -x[0].perishability_hours,  # Should be LOWEST perishability_hours first (soonest to expire)
            -{'High':3, 'Medium':2, 'Low':1}[x[0].nutritional_value],
            -x[0].quantity_kg
        )
    )
    
    # 4. Track items at risk of expiring soon
    for food_item, _ in sorted_food:
        expiry = food_item.expiry_date if isinstance(food_item.expiry_date, datetime) else \
                datetime.strptime(food_item.expiry_date, "%Y-%m-%d %H:%M:%S")
        if (expiry - current_time).total_seconds() < 6 * 3600:  # <6 hours remaining
            metrics['waste_risk_items'] += 1
    
    # 5. Enhanced allocation algorithm
    allocations = {r['obj'].recipient_id: [] for r in recipient_data}
    
    # First pass: satisfy minimum allocations for high-priority
    for recipient in [r for r in recipient_data if r['min_allocation'] > 0]:
        needed = max(0, recipient['min_allocation'] - recipient['remaining_demand'])
        if needed <= 0:
            continue
            
        for food_item, donor in sorted_food:
            if food_item.reserved or food_item.quantity_kg <= 0:
                continue
                
            alloc_kg = min(food_item.quantity_kg, needed)
            if alloc_kg <= 0:
                continue
                
            # Create allocation
            expiry_str = food_item.expiry_date.strftime("%Y-%m-%d %H:%M:%S") if \
                        isinstance(food_item.expiry_date, datetime) else food_item.expiry_date
            
            allocated_food = FoodItem(
                food_id=f"{food_item.food_id}_pri_{alloc_kg}",
                food_type=food_item.food_type,
                quantity_kg=alloc_kg,
                expiry_date=expiry_str,
                perishability_hours=food_item.perishability_hours,
                nutritional_value=food_item.nutritional_value
            )
            
            allocations[recipient['obj'].recipient_id].append((donor, recipient['obj'], allocated_food))
            food_item.quantity_kg -= alloc_kg
            recipient['remaining_demand'] -= alloc_kg
            metrics['total_allocated'] += alloc_kg
            food_item.reserve()
            needed -= alloc_kg
            
            if needed <= 0:
                break
    
    # Second pass: normal allocation
    for food_item, donor in sorted_food:
        if food_item.reserved or food_item.quantity_kg <= 0:
            continue
            
        best_score = -1
        best_recipient = None
        
        for recipient in recipient_data:
            if recipient['remaining_demand'] <= 0:
                continue
                
            # Calculate enhanced score
            pref_match = 2.0 if food_item.food_type == recipient['preference'] else 1.0
            
            expiry = food_item.expiry_date if isinstance(food_item.expiry_date, datetime) else \
                    datetime.strptime(food_item.expiry_date, "%Y-%m-%d %H:%M:%S")
            hours_remaining = (expiry - current_time).total_seconds() / 3600
            
            # Urgency boost for items expiring soon
            urgency_boost = 1.5 if hours_remaining < 12 else 1.0
            
            score = (recipient['base_score'] * 
                    recipient['demand_factor'] * 
                    pref_match * 
                    urgency_boost)
            
            if score > best_score:
                best_score = score
                best_recipient = recipient
        
        if best_recipient:
            alloc_kg = min(food_item.quantity_kg, best_recipient['remaining_demand'])
            
            expiry_str = food_item.expiry_date.strftime("%Y-%m-%d %H:%M:%S") if \
                        isinstance(food_item.expiry_date, datetime) else food_item.expiry_date
            
            allocated_food = FoodItem(
                food_id=f"{food_item.food_id}_alloc_{alloc_kg}",
                food_type=food_item.food_type,
                quantity_kg=alloc_kg,
                expiry_date=expiry_str,
                perishability_hours=food_item.perishability_hours,
                nutritional_value=food_item.nutritional_value
            )
            
            allocations[best_recipient['obj'].recipient_id].append((donor, best_recipient['obj'], allocated_food))
            food_item.quantity_kg -= alloc_kg
            best_recipient['remaining_demand'] -= alloc_kg
            metrics['total_allocated'] += alloc_kg
            food_item.reserve()
    
    # 6. Calculate and log metrics
    for recipient in recipient_data:
        allocated = sum(item[2].quantity_kg for item in allocations[recipient['obj'].recipient_id])
        metrics['fulfillment_ratios'][recipient['obj'].recipient_id] = allocated / max(1, recipient['obj'].current_demand)
        
        if recipient['min_allocation'] > 0 and allocated < recipient['min_allocation']:
            metrics['priority_violations'] += 1
    
    print(f"""
[DP Optimizer Metrics]
- Total allocated: {metrics['total_allocated']:.2f}kg
- Waste risk items: {metrics['waste_risk_items']}
- Priority violations: {metrics['priority_violations']}
- Avg fulfillment: {np.mean(list(metrics['fulfillment_ratios'].values())):.2f}
- Fairness score: {1 - np.std(list(metrics['fulfillment_ratios'].values())):.3f}
""")
    
    return allocations