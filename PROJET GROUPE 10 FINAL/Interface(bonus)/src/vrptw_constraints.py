"""
Module de gestion des contraintes VRPTW (Vehicle Routing Problem with Time Windows)

Ce module fournit des fonctions pour vérifier et gérer les contraintes de fenêtres
temporelles dans les solutions VRP.

Contraintes gérées:
- Fenêtres temporelles [ready_time, due_date] pour chaque client
- Temps de service à chaque client
- Temps de trajet entre clients
"""

import numpy as np
from copy import deepcopy


def calculate_route_time_schedule(route, instance):
    """
    Calcule le planning temporel complet d'une route.

    Pour chaque client dans la route, calcule:
    - Heure d'arrivée
    - Temps d'attente (si arrivée avant ready_time)
    - Heure de début de service
    - Heure de départ

    Args:
        route (list): Route [depot, c1, c2, ..., cn, depot]
        instance (dict): Instance avec 'edge_weight', 'time_window', 'service_time'

    Returns:
        dict: {
            'arrival_times': list,    # Heures d'arrivée à chaque nœud
            'waiting_times': list,    # Temps d'attente à chaque nœud
            'service_start': list,    # Heures de début de service
            'departure_times': list,  # Heures de départ de chaque nœud
            'total_time': float,      # Temps total de la route
            'is_feasible': bool       # Route faisable?
        }
    """
    distances = instance['edge_weight']
    time_windows = instance['time_window']
    service_times = instance['service_time']

    n = len(route)
    arrival_times = [0] * n
    waiting_times = [0] * n
    service_start = [0] * n
    departure_times = [0] * n

    is_feasible = True
    current_time = 0

    for i in range(n):
        node = route[i]

        # Arrivée au nœud
        if i == 0:
            # Départ du dépôt
            arrival_times[i] = 0
            ready, due = time_windows[node]
            current_time = max(0, ready)  # Commencer au plus tôt à ready_time du dépôt
        else:
            # Arrivée = départ du nœud précédent + temps de trajet
            prev_node = route[i - 1]
            travel_time = distances[prev_node][node]
            arrival_times[i] = current_time + travel_time

        # Fenêtre temporelle du nœud
        ready, due = time_windows[node]

        # Vérifier si arrivée avant due_date
        if arrival_times[i] > due:
            is_feasible = False

        # Attente si arrivée avant ready_time
        if arrival_times[i] < ready:
            waiting_times[i] = ready - arrival_times[i]
            service_start[i] = ready
        else:
            waiting_times[i] = 0
            service_start[i] = arrival_times[i]

        # Départ = début service + temps de service
        departure_times[i] = service_start[i] + service_times[node]
        current_time = departure_times[i]

    total_time = departure_times[-1]  # Retour au dépôt

    return {
        'arrival_times': arrival_times,
        'waiting_times': waiting_times,
        'service_start': service_start,
        'departure_times': departure_times,
        'total_time': total_time,
        'is_feasible': is_feasible
    }


def is_route_feasible(route, instance):
    """
    Vérifie si une route respecte les contraintes de fenêtres temporelles et de capacité.

    Args:
        route (list): Route [depot, c1, c2, ..., cn, depot]
        instance (dict): Instance VRP

    Returns:
        tuple: (is_feasible: bool, violations: dict)
            violations = {
                'capacity_exceeded': bool,
                'time_window_violations': list of (node_idx, arrival, due_date),
                'capacity_used': int,
                'capacity_max': int
            }
    """
    violations = {
        'capacity_exceeded': False,
        'time_window_violations': [],
        'capacity_used': 0,
        'capacity_max': instance['capacity']
    }

    # Vérifier capacité
    capacity_used = sum(instance['demand'][c] for c in route if c != 0)
    violations['capacity_used'] = capacity_used

    if capacity_used > instance['capacity']:
        violations['capacity_exceeded'] = True

    # Vérifier fenêtres temporelles
    schedule = calculate_route_time_schedule(route, instance)

    if not schedule['is_feasible']:
        # Identifier les violations
        time_windows = instance['time_window']
        for i, node in enumerate(route):
            arrival = schedule['arrival_times'][i]
            ready, due = time_windows[node]

            if arrival > due:
                violations['time_window_violations'].append((node, arrival, due))

    is_feasible = (not violations['capacity_exceeded'] and
                   len(violations['time_window_violations']) == 0)

    return is_feasible, violations


def is_solution_feasible(solution, instance):
    """
    Vérifie si une solution complète est faisable.

    Args:
        solution (list): Liste de routes
        instance (dict): Instance VRP

    Returns:
        tuple: (is_feasible: bool, route_violations: list)
            route_violations = [(route_idx, violations), ...]
    """
    route_violations = []

    for route_idx, route in enumerate(solution):
        is_feasible, violations = is_route_feasible(route, instance)

        if not is_feasible:
            route_violations.append((route_idx, violations))

    return len(route_violations) == 0, route_violations


def calculate_insertion_time_impact(route, customer, insert_pos, instance):
    """
    Calcule l'impact temporel de l'insertion d'un client dans une route.

    Args:
        route (list): Route actuelle
        customer (int): Client à insérer
        insert_pos (int): Position d'insertion (entre insert_pos-1 et insert_pos)
        instance (dict): Instance VRP

    Returns:
        dict: {
            'time_increase': float,  # Augmentation du temps total
            'is_feasible': bool,     # Insertion faisable?
            'new_total_time': float  # Nouveau temps total si insertion
        }
    """
    # Calcul du temps avant insertion
    schedule_before = calculate_route_time_schedule(route, instance)
    time_before = schedule_before['total_time']

    # Créer nouvelle route avec insertion
    new_route = route[:insert_pos] + [customer] + route[insert_pos:]

    # Calcul du temps après insertion
    schedule_after = calculate_route_time_schedule(new_route, instance)
    time_after = schedule_after['total_time']

    # Vérifier faisabilité (capacité + fenêtres temporelles)
    is_feasible, _ = is_route_feasible(new_route, instance)

    return {
        'time_increase': time_after - time_before,
        'is_feasible': is_feasible,
        'new_total_time': time_after,
        'old_total_time': time_before
    }


def calculate_solution_total_time(solution, instance):
    """
    Calcule le temps total d'une solution (somme des temps de toutes les routes).

    Args:
        solution (list): Liste de routes
        instance (dict): Instance VRP

    Returns:
        float: Temps total
    """
    total_time = 0

    for route in solution:
        schedule = calculate_route_time_schedule(route, instance)
        total_time += schedule['total_time']

    return total_time


def calculate_solution_total_distance(solution, instance):
    """
    Calcule la distance totale d'une solution (somme des distances de toutes les routes).

    Args:
        solution (list): Liste de routes
        instance (dict): Instance VRP

    Returns:
        float: Distance totale
    """
    distances = instance['edge_weight']
    total_distance = 0

    for route in solution:
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            total_distance += distances[from_node][to_node]

    return total_distance


def repair_time_window_violations(solution, instance, max_attempts=10):
    """
    Tente de réparer les violations de fenêtres temporelles dans une solution.

    Stratégie:
    1. Identifier les routes avec violations
    2. Pour chaque violation, essayer de réordonner les clients
    3. Si échec, essayer de déplacer les clients problématiques vers d'autres routes

    Args:
        solution (list): Solution à réparer
        instance (dict): Instance VRP
        max_attempts (int): Nombre maximum de tentatives

    Returns:
        tuple: (repaired_solution: list, success: bool)
    """
    solution = deepcopy(solution)

    for attempt in range(max_attempts):
        is_feasible, violations = is_solution_feasible(solution, instance)

        if is_feasible:
            return solution, True

        # Essayer de réparer la première route avec violation
        if violations:
            route_idx, route_violations = violations[0]
            route = solution[route_idx]

            # Stratégie 1: Réordonner les clients par ready_time
            if route_violations['time_window_violations']:
                customers = route[1:-1]  # Sans dépôt
                time_windows = instance['time_window']

                # Trier par ready_time
                customers_sorted = sorted(customers,
                                         key=lambda c: time_windows[c][0])

                new_route = [route[0]] + customers_sorted + [route[-1]]

                # Vérifier si la réparation améliore
                is_feas_new, _ = is_route_feasible(new_route, instance)

                if is_feas_new:
                    solution[route_idx] = new_route
                    continue

        # Si aucune réparation réussie, échec
        break

    # Retourner la meilleure solution trouvée (même si non faisable)
    return solution, False


def print_route_schedule(route, instance, route_name="Route"):
    """
    Affiche le planning détaillé d'une route.

    Args:
        route (list): Route à afficher
        instance (dict): Instance VRP
        route_name (str): Nom de la route pour l'affichage
    """
    schedule = calculate_route_time_schedule(route, instance)
    time_windows = instance['time_window']
    demands = instance['demand']

    print(f"\n=== {route_name} ===")
    print(f"Temps total: {schedule['total_time']:.2f}")
    print(f"Faisable: {'Oui' if schedule['is_feasible'] else 'Non'}")

    print(f"\n{'Node':<6} {'Arrival':<10} {'Ready':<8} {'Due':<8} {'Wait':<8} "
          f"{'Service':<10} {'Depart':<10} {'Demand':<8}")
    print("-" * 90)

    for i, node in enumerate(route):
        arrival = schedule['arrival_times'][i]
        waiting = schedule['waiting_times'][i]
        service_start = schedule['service_start'][i]
        departure = schedule['departure_times'][i]
        ready, due = time_windows[node]
        demand = demands[node]

        violation = "!" if arrival > due else ""

        print(f"{node:<6} {arrival:<10.2f} {ready:<8} {due:<8} {waiting:<8.2f} "
              f"{service_start:<10.2f} {departure:<10.2f} {demand:<8} {violation}")


# Exemple d'utilisation
if __name__ == "__main__":
    print("Module de contraintes VRPTW")
    print("Importez ce module pour utiliser les fonctions de vérification")
    print("\nExemple:")
    print("  from src.vrptw_constraints import is_route_feasible")
    print("  is_feasible, violations = is_route_feasible(route, instance)")
