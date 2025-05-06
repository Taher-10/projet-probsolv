import pulp
from agents.donor import DonorAgent
from agents.recipient import RecipientAgent
from agents.transport import TransportAgent

def build_variables(food_items, recipients, vehicles):
    # Variables for quantity assigned (continuous, 0 to food item quantity)
    return pulp.LpVariable.dicts(
        "x", [(i, j, k) for i in range(len(food_items)) 
                        for j in range(len(recipients)) 
                        for k in range(len(vehicles))],
        lowBound=0,
        cat="Continuous"
    )

def objective_function(prob, x, food_items, recipients, vehicles, calc_dist):
    # Minimize total distance, prioritize perishability, maximize assigned quantity
    prob += pulp.lpSum(
        x[(i, j, k)] * (
            calc_dist(vehicles[k].location, food_items[i][1].location) +  # Vehicle to donor
            calc_dist(food_items[i][1].location, recipients[j].location)  # Donor to recipient
        ) + x[(i, j, k)] * (1.0 / max(food_items[i][0].perishability_hours, 1e-6)) * 100.0  # Perishability penalty
        - x[(i, j, k)] * 5.0  # Encourage assignment (scaled quantity)
        for i in range(len(food_items)) for j in range(len(recipients)) for k in range(len(vehicles))
    )

def add_constraints(prob, x, food_items, recipients, vehicles):
    # Constraint 1: Assign at most the available quantity of each food item
    for i in range(len(food_items)):
        prob += pulp.lpSum(x[(i, j, k)] for j in range(len(recipients)) for k in range(len(vehicles))) <= food_items[i][0].quantity_kg

    # Constraint 2: Respect vehicle capacity
    for k in range(len(vehicles)):
        prob += pulp.lpSum(
            x[(i, j, k)]
            for i in range(len(food_items)) for j in range(len(recipients))
        ) <= vehicles[k].capacity_kg

    # Constraint 3: Respect recipient demand
    for j in range(len(recipients)):
        prob += pulp.lpSum(
            x[(i, j, k)]
            for i in range(len(food_items)) for k in range(len(vehicles))
        ) <= recipients[j].current_demand

def extract_solution(x, food_items, recipients, vehicles, step_count, current_time):
    assignments = {v.transport_id: [] for v in vehicles}
    total_assigned_kg = 0

    for i in range(len(food_items)):
        for j in range(len(recipients)):
            for k in range(len(vehicles)):
                assigned_qty = pulp.value(x[(i, j, k)])
                if assigned_qty is not None and assigned_qty > 0.01:  # Non-zero assignment
                    food_item, donor = food_items[i]
                    recipient = recipients[j]
                    vehicle = vehicles[k]
                    
                    # Scale nutritional value proportionally to assigned quantity
                    nutritional_value = food_item.nutritional_value * (assigned_qty / food_item.quantity_kg) if food_item.quantity_kg > 0 else food_item.nutritional_value
                    
                    # Create a new FoodItem for the assigned quantity
                    from agents.food_item import FoodItem
                    new_food_item = FoodItem(
                        food_id=f"{food_item.food_id}_split_{step_count}_{i}_{j}_{k}",
                        food_type=food_item.food_type,
                        quantity_kg=assigned_qty,
                        perishability_hours=food_item.perishability_hours,
                        expiry_date=food_item.expiry_date,
                        nutritional_value=nutritional_value
                    )
                    new_food_item.reserve()
                    assignments[vehicle.transport_id].append((donor, recipient, new_food_item))
                    total_assigned_kg += assigned_qty

                    # Update original food item quantity
                    food_item.quantity_kg -= assigned_qty
                    if food_item.quantity_kg < 0.01:
                        food_item.quantity_kg = 0
                        food_item.reserve()  # Mark as fully assigned

    # Log assignments
    print(f"Step {step_count}: LP Assignments: {[(tid, len(assignments[tid])) for tid in assignments]}")
    return assignments, total_assigned_kg

def optimize_routes_lp(food_items, recipients, vehicles, calculate_distance, current_time, step_count=0):
    """
    Optimize food delivery routes using PuLP linear programming.
    
    Args:
        food_items (list): List of (FoodItem, DonorAgent) tuples.
        recipients (list): List of RecipientAgent objects.
        vehicles (list): List of TransportAgent objects.
        calculate_distance (function): Function to compute distance between locations.
        current_time (datetime): Current simulation time.
        step_count (int): Current simulation step.
    
    Returns:
        tuple: (assignments, total_assigned_kg)
            - assignments: Dict mapping vehicle transport_id to list of (donor, recipient, food_item).
            - total_assigned_kg: Total quantity assigned in kg.
    """
    if not food_items or not recipients or not vehicles:
        print(f"Step {step_count}: No assignments possible (food_items={len(food_items)}, recipients={len(recipients)}, vehicles={len(vehicles)})")
        return {v.transport_id: [] for v in vehicles}, 0

    prob = pulp.LpProblem("FoodDeliveryOptimization", pulp.LpMinimize)
    x = build_variables(food_items, recipients, vehicles)
    objective_function(prob, x, food_items, recipients, vehicles, calculate_distance)
    add_constraints(prob, x, food_items, recipients, vehicles)
    
    # Solve the LP problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    print(f"Step {step_count}: LP Solver status: {pulp.LpStatus[prob.status]}")
    if prob.status != pulp.LpStatusOptimal:
        print(f"Step {step_count}: No optimal solution found. Returning empty assignments.")
        return {v.transport_id: [] for v in vehicles}, 0
    
    print(f"Step {step_count}: LP Objective value: {pulp.value(prob.objective):.2f}")
    return extract_solution(x, food_items, recipients, vehicles, step_count, current_time)