import numpy as np
from deap import base, creator, tools, algorithms

def optimize_routes_ga(food_items, recipients, vehicles, calculate_distance, current_time):
    """
    Optimize vehicle assignments using a genetic algorithm.
    
    Args:
        food_items: List of (food_item, donor) tuples.
        recipients: List of RecipientAgent objects with positive demand.
        vehicles: List of TransportAgent objects that are not busy.
        calculate_distance: Function to compute distance between two locations.
        current_time: Current simulation time (datetime).
    
    Returns:
        vehicle_assignments: Dict mapping transport_id to list of (donor, recipient, food_item) tuples.
        total_assigned_kg: Total kilograms assigned in the step.
    """
    # DEAP setup
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    def create_assignment():
        if not food_items or not recipients or not vehicles:
            return None
        i = np.random.randint(0, len(food_items))
        total_demand = sum(r.current_demand for r in recipients)
        if total_demand < 30.0:  # Low-demand step: assign to any recipient with demand
            probs = [1.0 if r.current_demand > 0 else 0.0 for r in recipients]
            probs = [p / sum(probs) if sum(probs) > 0 else 1.0 / len(recipients) for p in probs]
        else:  # Normal demand: prioritize by demand
            probs = [r.current_demand / total_demand if total_demand > 0 else 1.0 / len(recipients) for r in recipients]
        j = np.random.choice(range(len(recipients)), p=probs)
        k = np.random.choice(range(len(vehicles)), p=[v.capacity_kg / sum(v.capacity_kg for v in vehicles) for v in vehicles])
        # Split quantity to fit vehicle/recipient constraints
        quantity_kg = min(food_items[i][0].quantity_kg, recipients[j].current_demand, vehicles[k].capacity_kg)
        return (i, j, k, quantity_kg)
    
    def init_individual():
        num_assignments = min(len(food_items), len(recipients) * len(vehicles), 20)
        assignments = []
        used_items = set()
        used_vehicles = set()
        # Ensure at least one valid assignment if possible
        if num_assignments > 0:
            for _ in range(10):
                assignment = create_assignment()
                if assignment is None:
                    break
                i, j, k, quantity_kg = assignment
                if (i not in used_items and 
                    k not in used_vehicles and 
                    quantity_kg <= recipients[j].current_demand and 
                    quantity_kg <= vehicles[k].capacity_kg):
                    assignments.append(assignment)
                    used_items.add(i)
                    used_vehicles.add(k)
                    break
        for _ in range(num_assignments - len(assignments)):
            for _ in range(10):
                assignment = create_assignment()
                if assignment is None:
                    break
                i, j, k, quantity_kg = assignment
                if (i not in used_items and 
                    k not in used_vehicles and 
                    quantity_kg <= recipients[j].current_demand and 
                    quantity_kg <= vehicles[k].capacity_kg):
                    assignments.append(assignment)
                    used_items.add(i)
                    used_vehicles.add(k)
                    break
        return creator.Individual(assignments)
    
    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate_individual(individual):
        total_cost = 0.0
        item_counts = {}
        vehicle_loads = {k: 0.0 for k in range(len(vehicles))}
        recipient_loads = {j: 0.0 for j in range(len(recipients))}
        vehicle_usage = set()
        
        for assignment in individual:
            if assignment is None:
                continue
            i, j, k, quantity_kg = assignment
            if not (0 <= i < len(food_items) and 0 <= j < len(recipients) and 0 <= k < len(vehicles)):
                return (10000,)  # Penalty for invalid indices
            
            item_counts[i] = item_counts.get(i, 0) + 1
            vehicle_loads[k] += quantity_kg
            recipient_loads[j] += quantity_kg
            vehicle_usage.add(k)
            
            # Cost components
            pickup_distance = calculate_distance(vehicles[k].location, food_items[i][1].location)
            delivery_distance = calculate_distance(food_items[i][1].location, recipients[j].location)
            total_cost += pickup_distance + delivery_distance
            total_cost += (10.0 / food_items[i][0].perishability_hours)
            total_cost -= quantity_kg * 5.0
        
        # Penalties for constraint violations
        penalty = 0
        for i, count in item_counts.items():
            if count > 1:
                penalty += 1000 * (count - 1)
        for k, load in vehicle_loads.items():
            if load > vehicles[k].capacity_kg:
                penalty += 1000 * (load - vehicles[k].capacity_kg)
            if load < vehicles[k].capacity_kg * 0.5:
                penalty += 200 * (vehicles[k].capacity_kg - load)
        for j, load in recipient_loads.items():
            if load > recipients[j].current_demand:
                penalty += 1000 * (load - recipients[j].current_demand) / (recipients[j].current_demand + 1e-6)
        # Penalty for unassigned items
        unassigned_items = len(food_items) - len(item_counts)
        penalty += 500 * unassigned_items
        return (total_cost + penalty,)
    
    toolbox.register("evaluate", evaluate_individual)
    
    def safe_crossover(ind1, ind2):
        if len(ind1) < 2 or len(ind2) < 2:
            return ind1, ind2
        return tools.cxTwoPoint(ind1, ind2)
    
    toolbox.register("mate", safe_crossover)
    
    def mutate_individual(individual):
        if not individual:
            return individual,
        if np.random.random() < 0.5:
            used_items = set(i for i, _, _, _ in individual)
            used_vehicles = set(k for _, _, k, _ in individual)
            for _ in range(10):
                assignment = create_assignment()
                if assignment is None:
                    break
                i, j, k, quantity_kg = assignment
                if (i not in used_items and 
                    k not in used_vehicles and 
                    quantity_kg <= recipients[j].current_demand and 
                    quantity_kg <= vehicles[k].capacity_kg):
                    individual.append(assignment)
                    break
        else:
            if len(individual) > 0:
                idx = np.random.randint(0, len(individual))
                if np.random.random() < 0.5:
                    individual.pop(idx)
                else:
                    for _ in range(10):
                        assignment = create_assignment()
                        if assignment is None:
                            break
                        i, j, k, quantity_kg = assignment
                        used_items = set(i for i, _, _, _ in individual if i != individual[idx][0])
                        used_vehicles = set(k for _, _, k, _ in individual if k != individual[idx][2])
                        if (i not in used_items and 
                            k not in used_vehicles and 
                            quantity_kg <= recipients[j].current_demand and 
                            quantity_kg <= vehicles[k].capacity_kg):
                            individual[idx] = assignment
                            break
        return individual,
    
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Adaptive GA parameters
    population_size = 300 if len(food_items) > 200 else 200 if len(food_items) > 100 else 100
    generations = 150 if len(food_items) > 200 else 100 if len(food_items) > 100 else 50
    cxpb = 0.9 if len(food_items) > 100 else 0.7
    population = toolbox.population(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    
    hof = tools.HallOfFame(1)
    population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=cxpb,
        mutpb=0.2,
        ngen=generations,
        stats=stats,
        halloffame=hof,
        verbose=False
    )
    
    best_individual = hof[0]
    
    # Apply assignments
    vehicle_assignments = {v.transport_id: [] for v in vehicles}
    total_assigned_kg = 0
    for assignment in best_individual:
        if assignment is None:
            continue
        i, j, k, quantity_kg = assignment
        if not (0 <= i < len(food_items) and 0 <= j < len(recipients) and 0 <= k < len(vehicles)):
            continue
        food_item, donor = food_items[i]
        recipient = recipients[j]
        vehicle = vehicles[k]
        food_item.quantity_kg = quantity_kg  # Adjust quantity for split
        food_item.reserve()
        vehicle_assignments[vehicle.transport_id].append((donor, recipient, food_item))
        total_assigned_kg += quantity_kg
    
    return vehicle_assignments, total_assigned_kg