import vrplib
import matplotlib.pyplot as plt
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
import sys

# Import de l'algorithme ALNS
sys.path.append('src')
from alns import ALNS


# ============================================================================
# SECTION 1: GESTION DE L'HISTORIQUE DES RÉSULTATS
# ============================================================================

def load_results_history():
    """Charge l'historique des résultats depuis un fichier JSON."""
    history_file = Path("results_history_vrptw.json")
    if history_file.exists():
        with open(history_file, 'r') as f:
            return json.load(f)
    return []


def save_result(instance_name, algorithm, initial_cost, final_cost,
                elapsed_time, num_routes, gap_vs_optimal=None):
    """Sauvegarde un résultat dans l'historique."""
    history = load_results_history()

    improvement = ((initial_cost - final_cost) / initial_cost) * 100

    result = {
        'timestamp': datetime.now().isoformat(),
        'instance': instance_name,
        'algorithm': algorithm,
        'initial_cost': float(initial_cost),
        'final_cost': float(final_cost),
        'improvement': float(improvement),
        'time_seconds': float(elapsed_time),
        'num_routes': int(num_routes),
        'gap_vs_optimal': float(gap_vs_optimal) if gap_vs_optimal is not None else None
    }

    history.append(result)

    # Sauvegarder
    with open("results_history_vrptw.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n[OK] Résultat sauvegardé dans l'historique ({len(history)} résultats au total)")

    return history


def plot_results_history(current_instance=None):
    """Affiche l'historique des résultats sous forme de graphiques."""
    history = load_results_history()

    if len(history) == 0:
        print("\n[INFO] Aucun historique disponible")
        return

    print(f"\n{'='*70}")
    print(f"HISTORIQUE DES RÉSULTATS - {len(history)} exécutions")
    print(f"{'='*70}")

    # Créer une figure avec 4 sous-graphiques
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Préparer les données
    instances = [r['instance'] for r in history]
    algorithms = [r['algorithm'] for r in history]
    improvements = [r['improvement'] for r in history]
    times = [r['time_seconds'] for r in history]
    final_costs = [r['final_cost'] for r in history]

    # Graphique 1: Évolution de l'amélioration au fil du temps
    ax1.plot(range(1, len(history) + 1), improvements,
             marker='o', linewidth=2, markersize=8, color='#2ecc71')

    # Mettre en évidence le résultat actuel
    if current_instance:
        current_indices = [i for i, r in enumerate(history) if r['instance'] == current_instance]
        if current_indices:
            last_idx = current_indices[-1]
            ax1.plot(last_idx + 1, improvements[last_idx],
                    marker='*', markersize=20, color='red',
                    label='Résultat actuel', zorder=10)

    ax1.set_xlabel('Numéro d\'exécution', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Amélioration (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Évolution des Améliorations', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Ajouter ligne de tendance
    if len(improvements) > 1:
        z = np.polyfit(range(len(improvements)), improvements, 1)
        p = np.poly1d(z)
        ax1.plot(range(1, len(history) + 1), p(range(len(improvements))),
                "--", alpha=0.5, color='red', linewidth=2, label='Tendance')

    # Graphique 2: Amélioration par instance
    unique_instances = list(set(instances))
    instance_improvements = {}

    for inst in unique_instances:
        inst_improvements = [r['improvement'] for r in history if r['instance'] == inst]
        instance_improvements[inst] = inst_improvements

    # Créer un box plot pour chaque instance
    data_to_plot = [instance_improvements[inst] for inst in unique_instances]
    positions = range(1, len(unique_instances) + 1)

    bp = ax2.boxplot(data_to_plot, positions=positions,
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

    # Colorer les boîtes
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_instances)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax2.set_xticks(positions)
    ax2.set_xticklabels(unique_instances, rotation=45, ha='right')
    ax2.set_xlabel('Instance', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Amélioration (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution des Améliorations par Instance',
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Graphique 3: Temps d'exécution par algorithme
    unique_algos = list(set(algorithms))
    algo_times = {algo: [r['time_seconds'] for r in history if r['algorithm'] == algo]
                  for algo in unique_algos}

    x_pos = np.arange(len(unique_algos))
    avg_times = [np.mean(algo_times[algo]) for algo in unique_algos]
    std_times = [np.std(algo_times[algo]) if len(algo_times[algo]) > 1 else 0
                 for algo in unique_algos]

    bars = ax3.bar(x_pos, avg_times, yerr=std_times,
                   color=['#3498db', '#e74c3c', '#f39c12'][:len(unique_algos)],
                   capsize=5, alpha=0.8)

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(unique_algos)
    ax3.set_xlabel('Algorithme', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Temps moyen (s)', fontsize=12, fontweight='bold')
    ax3.set_title('Temps d\'Exécution Moyen par Algorithme',
                 fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Ajouter les valeurs sur les barres
    for i, (bar, val) in enumerate(zip(bars, avg_times)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}s', ha='center', va='bottom', fontweight='bold')

    # Graphique 4: Coût final vs Instance (évolution temporelle)
    # Grouper par instance et montrer l'évolution
    ax4_data = {}
    for r in history:
        inst = r['instance']
        if inst not in ax4_data:
            ax4_data[inst] = []
        ax4_data[inst].append(r['final_cost'])

    for i, (inst, costs) in enumerate(ax4_data.items()):
        color = colors[i % len(colors)]
        ax4.plot(range(1, len(costs) + 1), costs,
                marker='o', linewidth=2, markersize=6,
                label=inst, color=color)

    ax4.set_xlabel('Exécution (pour cette instance)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Coût final', fontsize=12, fontweight='bold')
    ax4.set_title('Évolution du Coût Final par Instance',
                 fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()

    # Sauvegarder le graphique
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"history_plot_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"[OK] Graphique sauvegardé: {filename}")

    plt.show()

    # Afficher un tableau récapitulatif
    print(f"\n{'='*70}")
    print(f"TABLEAU RÉCAPITULATIF")
    print(f"{'='*70}")
    print(f"{'#':<4} {'Instance':<15} {'Algo':<10} {'Init':<8} {'Final':<8} {'Amél.':<8} {'Temps':<8}")
    print(f"{'-'*70}")

    for i, r in enumerate(history[-10:], start=max(1, len(history)-9)):  # 10 derniers
        print(f"{i:<4} {r['instance']:<15} {r['algorithm']:<10} "
              f"{r['initial_cost']:<8.0f} {r['final_cost']:<8.0f} "
              f"{r['improvement']:<8.1f}% {r['time_seconds']:<8.1f}s")

    if len(history) > 10:
        print(f"... ({len(history)-10} résultats plus anciens)")

    print(f"{'='*70}\n")


# ============================================================================
# SECTION 2: FONCTION DE VÉRIFICATION DES CONTRAINTES TEMPORELLES
# ============================================================================

def check_temporal_time(temporal_route, time_windows, service_time, distance):
    """
    Vérifie si une route respecte les contraintes de fenêtres temporelles.

    Args:
        temporal_route: Liste des indices de clients (sans dépôt)
        time_windows: Array des fenêtres temporelles
        service_time: Array des temps de service
        distance: Matrice des distances

    Returns:
        bool: True si faisable, False sinon
    """
    full_route = [0] + temporal_route + [0]
    current_time = 0.0
    is_feasible = True

    for i in range(len(full_route) - 1):
        node_i = full_route[i]
        node_j = full_route[i + 1]
        travel_time = distance[node_i, node_j]
        arrival_time_j = current_time + travel_time

        earliest_j = time_windows[node_j, 0]
        latest_j = time_windows[node_j, 1]

        if arrival_time_j > latest_j:
            is_feasible = False
            break

        start_service_j = max(arrival_time_j, earliest_j)
        end_service_j = start_service_j + service_time[node_j]
        current_time = end_service_j

    return is_feasible


# ============================================================================
# SECTION 3: GÉNÉRATION DE LA SOLUTION INITIALE
# ============================================================================

def generate_initial_solution_vrptw(instance):
    """
    Génère une solution initiale avec insertion séquentielle + contraintes VRPTW.

    Args:
        instance: Dictionnaire d'instance vrplib

    Returns:
        tuple: (solution, initial_cost)
    """
    print("\n" + "="*70)
    print("GÉNÉRATION DE LA SOLUTION INITIALE")
    print("="*70)

    coord = instance['node_coord']
    distance = instance['edge_weight']
    time_windows = instance['time_window']
    service_time = instance['service_time']
    demand = instance['demand']
    max_capacity = instance['capacity']

    # Coordonnées du dépôt
    depot_x, depot_y = coord[0][0], coord[0][1]

    # Coordonnées des clients
    client_coords = np.delete(coord, 0, axis=0)
    client_x = client_coords[:, 0]
    client_y = client_coords[:, 1]

    # Calcul des distances au dépôt
    distance_coords = []
    for i in range(len(client_coords)):
        abs_client_x = abs(client_x[i] - depot_x)
        abs_client_y = abs(client_y[i] - depot_y)
        distance_coords.append(abs_client_x + abs_client_y)

    clients_remaining = list(range(len(demand) - 1))
    initial_solution = []
    route_num = 1

    while clients_remaining:
        print(f"\nConstruction de la route n°{route_num}...")

        # Choisir le client le plus proche du dépôt
        distances_remaining = [distance_coords[i] for i in clients_remaining]
        index_min = np.argmin(distances_remaining)
        client_choosed = clients_remaining[index_min]
        client_choosed_vrplib = client_choosed + 1

        current_route = [client_choosed_vrplib]
        actual_charge = demand[client_choosed_vrplib]
        clients_remaining.remove(client_choosed)

        # Extension de la route
        continue_extension = True
        while clients_remaining and continue_extension:
            best_insertion_cost = np.inf
            min_index_cost = None
            best_index = -1

            for client_index in clients_remaining:
                true_client_index = client_index + 1
                client_demand = demand[true_client_index]

                if actual_charge + client_demand <= max_capacity:
                    test_route = [0] + current_route + [0]

                    for i in range(len(test_route) - 1):
                        client_i = test_route[i]
                        client_j = test_route[i + 1]

                        temporal_route = current_route[:]
                        temporal_route.insert(i, true_client_index)

                        time_feasible = check_temporal_time(temporal_route, time_windows,
                                                           service_time, distance)

                        if time_feasible:
                            insertion_distance = (distance[client_i, true_client_index] +
                                                distance[true_client_index, client_j] -
                                                distance[client_i, client_j])

                            if insertion_distance < best_insertion_cost:
                                best_insertion_cost = insertion_distance
                                min_index_cost = client_index
                                best_index = i

            if min_index_cost is not None:
                client_to_add = min_index_cost + 1
                actual_charge += demand[client_to_add]
                current_route.insert(best_index, client_to_add)
                clients_remaining.remove(min_index_cost)
            else:
                continue_extension = False

        initial_solution.append([0] + current_route + [0])
        print(f"  Route n°{route_num}: {initial_solution[-1]}")
        print(f"  Charge: {actual_charge}/{max_capacity}")
        route_num += 1

    # Calcul du coût
    initial_cost = 0.0
    for route in initial_solution:
        for i in range(len(route) - 1):
            initial_cost += distance[route[i], route[i + 1]]

    print(f"\n[OK] Solution initiale générée:")
    print(f"  - Nombre de routes: {len(initial_solution)}")
    print(f"  - Coût total: {initial_cost:.2f}")

    return initial_solution, initial_cost


# ============================================================================
# SECTION 4: OPTIMISATION AVEC ALNS
# ============================================================================

def optimize_with_alns(instance, initial_solution, time_limit=60):
    """
    Optimise une solution avec ALNS.

    Args:
        instance: Dictionnaire d'instance
        initial_solution: Solution initiale
        time_limit: Temps limite en secondes

    Returns:
        tuple: (best_solution, best_cost, cost_history)
    """
    print("\n" + "="*70)
    print("OPTIMISATION AVEC ALNS")
    print("="*70)
    print(f"Temps limite: {time_limit}s")

    # Créer le solver ALNS
    solver = ALNS(instance, time_limit=time_limit, temperature=100, cooling_rate=0.995)

    # Lancer l'optimisation
    start_time = time.time()
    best_solution, best_cost, cost_history = solver.solve(initial_solution)
    elapsed_time = time.time() - start_time

    print(f"\n[OK] Optimisation terminée:")
    print(f"  - Temps écoulé: {elapsed_time:.2f}s")
    print(f"  - Coût final: {best_cost:.2f}")
    print(f"  - Nombre d'itérations: {len(cost_history)}")

    return best_solution, best_cost, cost_history, elapsed_time


# ============================================================================
# SECTION 5: VISUALISATION
# ============================================================================

def plot_solution(instance, solution, cost, title="Solution VRP"):
    """Affiche graphiquement une solution."""
    coord = instance['node_coord']
    depot_x, depot_y = coord[0][0], coord[0][1]
    client_coords = np.delete(coord, 0, axis=0)
    client_x = client_coords[:, 0]
    client_y = client_coords[:, 1]

    plt.figure(figsize=(12, 9))
    plt.plot(depot_x, depot_y, 's', color='red', markersize=15,
            label='Dépôt', zorder=5)
    plt.plot(client_x, client_y, 'o', color='blue', markersize=7,
            label='Clients', alpha=0.6)

    # Numérotation
    dimension = len(coord)
    for i in range(dimension):
        x, y = coord[i]
        plt.text(x + 1, y + 1, str(i), fontsize=9, fontweight='bold')

    # Routes
    cmap = plt.colormaps.get_cmap('hsv')
    indices = np.linspace(0, 1, len(solution) + 1)
    custom_colors = cmap(indices)

    for route_index, route in enumerate(solution):
        route_coord_x = [coord[node_index][0] for node_index in route]
        route_coord_y = [coord[node_index][1] for node_index in route]

        plt.plot(route_coord_x, route_coord_y,
                color=custom_colors[route_index],
                linestyle='-',
                linewidth=2.5,
                alpha=0.8,
                label=f'Route {route_index + 1}')

    plt.title(f"{title}\nCoût total: {cost:.0f}",
             fontsize=16, fontweight='bold')
    plt.xlabel("Coordonnée X", fontsize=12)
    plt.ylabel("Coordonnée Y", fontsize=12)
    plt.legend(loc='best')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_convergence(cost_history, initial_cost, final_cost):
    """Affiche la courbe de convergence."""
    plt.figure(figsize=(12, 6))

    iterations = range(len(cost_history))
    plt.plot(iterations, cost_history, linewidth=2, color='#3498db', label='ALNS')
    plt.axhline(y=initial_cost, color='red', linestyle='--', linewidth=2,
               label=f'Coût initial: {initial_cost:.0f}')
    plt.axhline(y=final_cost, color='green', linestyle='--', linewidth=2,
               label=f'Coût final: {final_cost:.0f}')

    # Amélioration
    improvement = ((initial_cost - final_cost) / initial_cost) * 100
    plt.text(len(cost_history)*0.7, (initial_cost + final_cost)/2,
            f'Amélioration: {improvement:.1f}%',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.xlabel('Itération', fontsize=12)
    plt.ylabel('Coût', fontsize=12)
    plt.title('Convergence de l\'algorithme ALNS', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# SECTION 6: PROGRAMME PRINCIPAL
# ============================================================================

def main():
    """Programme principal."""
    print("\n" + "="*70)
    print(" "*15 + "TEST VRPTW AVEC ALNS ET SUIVI DES RÉSULTATS")
    print("="*70)

    # Lecture de l'instance
    path_file_instance = 'tests/data/cvrplib/Vrp-Set-Solomon/C201.txt'
    print(f"\nChargement de l'instance: {path_file_instance}")

    instance = vrplib.read_instance(path_file_instance, instance_format="solomon")

    # Ajouter 'dimension' si manquant (requis par ALNS)
    if 'dimension' not in instance:
        instance['dimension'] = len(instance['demand'])

    print(f"\n[OK] Instance chargée:")
    print(f"  - Nom: {instance['name']}")
    print(f"  - Clients: {len(instance['demand']) - 1}")
    print(f"  - Véhicules: {instance['vehicles']}")
    print(f"  - Capacité: {instance['capacity']}")

    # Génération de la solution initiale
    initial_solution, initial_cost = generate_initial_solution_vrptw(instance)

    # Optimisation avec ALNS
    best_solution, best_cost, cost_history, elapsed_time = optimize_with_alns(
        instance, initial_solution, time_limit=60
    )

    # Calcul de l'amélioration
    improvement = ((initial_cost - best_cost) / initial_cost) * 100

    print("\n" + "="*70)
    print("RÉSULTATS FINAUX")
    print("="*70)
    print(f"Coût initial:     {initial_cost:.2f}")
    print(f"Coût final:       {best_cost:.2f}")
    print(f"Amélioration:     {improvement:.2f}%")
    print(f"Temps:            {elapsed_time:.2f}s")
    print(f"Nombre de routes: {len(best_solution)}")

    # Calcul du gap vs optimal (si solution optimale disponible)
    try:
        path_sol = path_file_instance.replace('.txt', '.sol')
        optimal_sol = vrplib.read_solution(path_sol)
        optimal_cost = optimal_sol['cost']
        gap = abs((best_cost - optimal_cost) / optimal_cost * 100)
        print(f"Coût optimal:     {optimal_cost:.2f}")
        print(f"Gap:              {gap:.2f}%")
    except:
        gap = None
        print(f"Coût optimal:     Non disponible")

    # Calcul CO2
    CO2_factor = 0.18  # kg CO2/km
    initial_CO2 = initial_cost * CO2_factor
    final_CO2 = best_cost * CO2_factor
    CO2_saved = initial_CO2 - final_CO2

    print(f"\nImpact environnemental:")
    print(f"  CO2 initial:    {initial_CO2:.2f} kg")
    print(f"  CO2 final:      {final_CO2:.2f} kg")
    print(f"  CO2 économisé:  {CO2_saved:.2f} kg ({(CO2_saved/initial_CO2*100):.1f}%)")
    print("="*70)

    # Sauvegarder dans l'historique
    save_result(
        instance_name=instance['name'],
        algorithm='ALNS',
        initial_cost=initial_cost,
        final_cost=best_cost,
        elapsed_time=elapsed_time,
        num_routes=len(best_solution),
        gap_vs_optimal=gap
    )

    # Visualisations
    print("\nAffichage des graphiques...")

    # 1. Solution initiale
    plot_solution(instance, initial_solution, initial_cost,
                 f"Solution Initiale - {instance['name']}")

    # 2. Solution optimisée
    plot_solution(instance, best_solution, best_cost,
                 f"Solution Optimisée (ALNS) - {instance['name']}")

    # 3. Convergence
    plot_convergence(cost_history, initial_cost, best_cost)

    # 4. Historique des résultats
    plot_results_history(current_instance=instance['name'])

    print("\n[OK] Programme terminé avec succès!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
