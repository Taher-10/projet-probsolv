import numpy as np
from datetime import timedelta

def fuel_consumption(distance, load_kg, vehicle):
    """Calculate fuel consumption based on distance, load, and vehicle efficiency."""
    load_factor = min(load_kg / vehicle.capacity_kg, 1.0)
    fuel_rate = vehicle.fuel_efficiency * (1.0 + 0.1 * load_factor)
    return distance * fuel_rate

def optimize_routes_aco(food_items, recipients, vehicles, distance_fn, current_time, n_ants=5, n_iters=20, alpha=1.0, beta=3.0, gamma=2.0, rho=0.1, q=1.0, early_stopping_patience=5):
    """
    Optimized Ant Colony Optimization for vehicle routing, minimizing fuel consumption, distance, and expiry urgency.
    
    Args:
        food_items: List of (FoodItem, DonorAgent) tuples.
        recipients: List of RecipientAgent with positive demand.
        vehicles: List of available TransportAgent.
        distance_fn: Function to compute distance between two locations.
        current_time: Current simulation time (datetime).
        n_ants: Number of ants per iteration.
        n_iters: Number of iterations.
        alpha: Pheromone influence.
        beta: Distance heuristic influence.
        gamma: Expiry urgency influence.
        rho: Pheromone evaporation rate.
        q: Pheromone deposit constant.
        early_stopping_patience: Stop if best score doesn't improve for this many iterations.
    
    Returns:
        assignments: Dict {vehicle_id: [(donor, recipient, food_item), ...]}.
        total_assigned_kg: Total kilograms assigned.
    """
    # Filter food items to exclude near-expired items (less than 1 hour)
    filtered_food_items = [
        (item, donor) for item, donor in food_items
        if (item.expiry_date - current_time).total_seconds() / 3600.0 > 1.0
    ]
    if not filtered_food_items:
        print("No valid food items after filtering near-expired items")
        return {v.transport_id: [] for v in vehicles}, 0.0

    # Build nodes for pickups and drops
    nodes = []
    for item, donor in filtered_food_items:
        nodes.append({'type': 'pick', 'loc': donor.location, 'donor': donor, 'item': item})
    for r in recipients:
        nodes.append({'type': 'drop', 'loc': r.location, 'recipient': r})
    N = len(nodes)
    print(f"ACO: {N} nodes ({len(filtered_food_items)} pick, {len(recipients)} drop)")

    # Precompute distance matrix
    dist = np.full((N, N), np.inf)
    for i in range(N):
        for j in range(N):
            if i != j:
                dist[i, j] = distance_fn(nodes[i]['loc'], nodes[j]['loc'])

    # Pheromone matrix
    pher = np.ones((N, N))

    # Expiry urgency for pick nodes
    urgency = np.zeros(N)
    for idx, node in enumerate(nodes):
        if node['type'] == 'pick':
            expiry = node['item'].expiry_date
            hours_left = max((expiry - current_time).total_seconds() / 3600.0, 0)
            urgency[idx] = 1.0 / (hours_left + 1e-6)

    best_solution = None
    best_score = np.inf
    total_assigned_kg = 0.0
    no_improvement_count = 0

    # ACO iterations
    for iter_idx in range(n_iters):
        solutions = []
        scores = []
        assigned_kgs = []

        for _ in range(n_ants):
            remaining = np.array(list(range(N)))
            solution = []
            score = 0.0
            assigned_kg = 0.0

            for vehicle in vehicles:
                route = []
                current_loc = vehicle.location
                load = 0.0
                last_type = None
                while len(remaining) > 0:
                    # Force alternation: pick -> drop -> pick -> drop
                    valid_nodes = [
                        j for j in remaining
                        if (last_type is None or
                            (last_type == 'pick' and nodes[j]['type'] == 'drop') or
                            (last_type == 'drop' and nodes[j]['type'] == 'pick'))
                    ]
                    if not valid_nodes:
                        break

                    # Vectorized transition probabilities
                    valid_indices = np.array([j for j in remaining if j in valid_nodes])
                    last_idx = route[-1] if route else -1
                    pher_values = pher[last_idx, valid_indices] if last_idx >= 0 else np.ones(len(valid_indices))
                    distances = np.array([dist[last_idx, j] for j in valid_indices]) if last_idx >= 0 else np.array([distance_fn(current_loc, nodes[j]['loc']) for j in valid_indices])
                    eta = (1.0 / (distances + 1e-6)) ** beta
                    urg = urgency[valid_indices] ** gamma
                    weights = (pher_values ** alpha) * eta * urg

                    total = weights.sum()
                    if total <= 0 or np.isnan(total):
                        probs = np.ones(len(valid_indices)) / len(valid_indices)
                    else:
                        probs = weights / total

                    choice_idx = np.random.choice(len(valid_indices), p=probs)
                    j = valid_indices[choice_idx]

                    # Update score and load
                    d_leg = distances[choice_idx] if last_idx >= 0 else distance_fn(current_loc, nodes[j]['loc'])
                    score += fuel_consumption(d_leg, load, vehicle) + 0.5 * d_leg
                    if nodes[j]['type'] == 'pick':
                        score += urgency[j] * 100
                        load += nodes[j]['item'].quantity_kg
                        assigned_kg += nodes[j]['item'].quantity_kg
                    route.append(j)
                    last_type = nodes[j]['type']
                    remaining = np.delete(remaining, np.where(remaining == j)[0])
                    current_loc = nodes[j]['loc']
                    if load >= vehicle.capacity_kg:
                        break
                solution.append(route)
                solutions.append(solution)
                scores.append(score)
                assigned_kgs.append(assigned_kg)

        # Update best solution
        min_score_idx = np.argmin(scores)
        if scores[min_score_idx] < best_score:
            best_score = scores[min_score_idx]
            best_solution = solutions[min_score_idx]
            total_assigned_kg = assigned_kgs[min_score_idx]
            no_improvement_count = 0
            print(f"Iteration {iter_idx + 1}: Best score={best_score:.2f}, Assigned={total_assigned_kg:.2f} kg")
        else:
            no_improvement_count += 1

        # Early stopping
        if no_improvement_count >= early_stopping_patience:
            print(f"Early stopping at iteration {iter_idx + 1}: no improvement for {early_stopping_patience} iterations")
            break

        # Pheromone update (only for best solution)
        pher *= (1 - rho)
        for route in solutions[min_score_idx]:
            for u, v in zip(route, route[1:]):
                pher[u, v] += q / (scores[min_score_idx] + 1e-6)

    # Decode best_solution into assignments
    assignments = {v.transport_id: [] for v in vehicles}
    for vidx, route in enumerate(best_solution):
        vehicle = vehicles[vidx]
        current_load = 0.0
        pick_node = None
        for idx in route:
            node = nodes[idx]
            if node['type'] == 'pick':
                current_load += node['item'].quantity_kg
                if current_load > vehicle.capacity_kg:
                    print(f"Vehicle {vehicle.transport_id}: Skipped pick node {idx} (load={current_load:.2f} kg > capacity={vehicle.capacity_kg})")
                    pick_node = None
                    continue
                pick_node = node
            elif node['type'] == 'drop' and pick_node:
                assignments[vehicle.transport_id].append((pick_node['donor'], node['recipient'], pick_node['item']))
                print(f"Vehicle {vehicle.transport_id}: Assigned (donor={pick_node['donor'].donor_id}, recipient={node['recipient'].recipient_id}, item={pick_node['item'].food_id}, kg={pick_node['item'].quantity_kg:.2f})")
                pick_node = None

    print(f"ACO: Final assignments: {[(vid, len(assignments[vid])) for vid in assignments]}")
    return assignments, total_assigned_kg