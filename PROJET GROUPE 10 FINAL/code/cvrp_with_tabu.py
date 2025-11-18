"""
Implémentation complète CVRP avec Tabu Search
Script principal intégrant la solution de base + optimisation Tabu Search
"""
import vrplib
import matplotlib.pyplot as plt
import numpy as np
from tabu_search_cvrp import TabuSearch_CVRP
import sys

# Ajout du chemin si nécessaire
sys.path.insert(0, 'tests')

# ==================== LECTURE DE L'INSTANCE ====================
print("="*60)
print("CVRP avec optimisation Tabu Search")
print("="*60)

path_file_instance_vrplib = 'code/tests/data/A-n32-k5.vrp'
instance_vrplib = vrplib.read_instance(path_file_instance_vrplib)

print(f"\nInstance chargée: {instance_vrplib['name']}")
print(f"Nombre de clients: {instance_vrplib['dimension'] - 1}")
print(f"Capacité des véhicules: {instance_vrplib['capacity']}")

# ==================== GÉNÉRATION DE LA SOLUTION INITIALE ====================
print("\n" + "="*60)
print("GÉNÉRATION DE LA SOLUTION INITIALE")
print("="*60)

# Récupération des coordonnées (x,y)
coord = instance_vrplib['node_coord']

# Coordonnées du dépôt
depot_index = 0
depot_x, depot_y = coord[depot_index][0], coord[depot_index][1]

# Coordonnées des clients
client_coords = np.delete(coord, depot_index, axis=0)
client_x = client_coords[:, 0]
client_y = client_coords[:, 1]

# On recherche le client le plus proche du dépôt
distance_coords = []
for i in range(len(client_coords)):
    abs_client_x = abs(client_x[i] - depot_x)
    abs_client_y = abs(client_y[i] - depot_y)
    sum_abs_x_y = abs_client_x + abs_client_y
    distance_coords.append(sum_abs_x_y)

# Initialisation de la solution initiale
max_capacity = instance_vrplib['capacity']
demand = instance_vrplib['demand']
distance = instance_vrplib['edge_weight']
clients_remaining = list(range(len(demand) - 1))
initial_solution = []

print("Construction de la solution initiale avec l'heuristique d'insertion...")

index4 = 1
while clients_remaining:
    # Choix du client le plus proche du dépôt
    distances_remaining = [distance_coords[i] for i in clients_remaining]
    index_min_distance_remaining = np.argmin(distances_remaining)
    client_choosed = clients_remaining[index_min_distance_remaining]

    client_choosed_vrplib = client_choosed + 1
    current_route = [client_choosed_vrplib]
    actual_charge = demand[client_choosed_vrplib]

    clients_remaining.remove(client_choosed)

    # Pour chaque client restant, trouver la meilleure insertion
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

                    insertion_distance = (distance[client_i, true_client_index] +
                                        distance[true_client_index, client_j] -
                                        distance[client_i, client_j])

                    if insertion_distance < best_insertion_cost:
                        best_insertion_cost = insertion_distance
                        min_index_cost = client_index
                        best_index = i + 1

        if min_index_cost is not None:
            client_to_add = min_index_cost + 1
            actual_charge = actual_charge + demand[client_to_add]
            current_route.insert(best_index - 1, client_to_add)
            clients_remaining.remove(min_index_cost)
        else:
            continue_extension = False

    initial_solution.append([0] + current_route + [0])
    index4 = index4 + 1

# Calcul du coût initial
initial_cost = 0.0
for route in initial_solution:
    for i in range(len(route) - 1):
        initial_cost += distance[route[i], route[i + 1]]

print(f"\nSolution initiale construite!")
print(f"Nombre de routes: {len(initial_solution)}")
print(f"Coût de la solution initiale: {initial_cost:.2f}")

# ==================== OPTIMISATION AVEC TABU SEARCH ====================
print("\n" + "="*60)
print("OPTIMISATION AVEC TABU SEARCH")
print("="*60)

# Créer l'instance Tabu Search
tabu = TabuSearch_CVRP(instance_vrplib, initial_solution)

# Exécuter l'algorithme
results = tabu.run(
    max_iterations=3000,
    tabu_tenure=10,
    neighborhood_size=50,
    diversification_freq=100
)

# ==================== CALCUL DU GAP ====================
print("\n" + "="*60)
print("ÉVALUATION DE LA PERFORMANCE")
print("="*60)

# Lecture de la solution optimale
path_file_solution_vrplib = 'code/tests/data/A-n32-k5.sol'
solution_vrplib = vrplib.read_solution(path_file_solution_vrplib)
optimal_cost = solution_vrplib['cost']

gap = ((results['best_cost'] - optimal_cost) / optimal_cost) * 100

print(f"\nCoût de la solution optimale connue: {optimal_cost:.2f}")
print(f"Coût de notre meilleure solution:    {results['best_cost']:.2f}")
print(f"Gap par rapport à l'optimum:         {gap:.2f}%")

if gap < 5:
    print(f"\n✓ OBJECTIF ATTEINT! Gap < 5%")
else:
    print(f"\n✗ Gap supérieur à 5%, optimisation supplémentaire nécessaire")

# Calcul du CO2 émis
CO2_g_per_km = 900
total_CO2_emmited_g = results['best_cost'] * CO2_g_per_km
total_CO2_emmited_kg = total_CO2_emmited_g / 1000

print(f"\nCO2 émis (solution initiale):        {(initial_cost * CO2_g_per_km / 1000):.2f} kg")
print(f"CO2 émis (solution optimisée):       {total_CO2_emmited_kg:.2f} kg")
print(f"Réduction de CO2:                    {((initial_cost - results['best_cost']) * CO2_g_per_km / 1000):.2f} kg")

# ==================== VISUALISATION ====================
print("\n" + "="*60)
print("GÉNÉRATION DES GRAPHIQUES")
print("="*60)

# 1. Graphique de convergence
tabu.plot_convergence('convergence_cvrp_tabu.png')

# 2. Affichage de la solution finale
plt.figure(figsize=(10, 8))
plt.plot(depot_x, depot_y, 's', color='red', markersize=10, label='Dépôt', zorder=5)
plt.plot(client_x, client_y, 'o', color='blue', markersize=5, label='Clients')

dimension = instance_vrplib['dimension']
for i in range(dimension):
    x, y = coord[i]
    plt.text(x + 1, y + 1, str(i + 1), fontsize=8)

# Palette de couleurs
cmap = plt.colormaps.get_cmap('hsv')
indices = np.linspace(0, 1, len(results['best_solution']) + 1)
custom_colors = cmap(indices)

# Personnalisation des routes
for route_index, route in enumerate(results['best_solution']):
    route_coord_x = [coord[node_index][0] for node_index in route]
    route_coord_y = [coord[node_index][1] for node_index in route]

    plt.plot(route_coord_x, route_coord_y,
            color=custom_colors[route_index],
            linestyle='-',
            linewidth=2,
            alpha=0.8,
            label=f'Route {route_index + 1}')

plt.title(f"Solution optimisée Tabu Search - {instance_vrplib['name']}\nCoût: {results['best_cost']:.2f} | Gap: {gap:.2f}%")
plt.xlabel("Coordonnée X")
plt.ylabel("Coordonnée Y")
plt.legend()
plt.axis('equal')
plt.savefig('solution_cvrp_tabu.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Graphique de performance comparative
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Comparaison des coûts
costs = [initial_cost, results['best_cost'], optimal_cost]
labels = ['Initial', 'Tabu Search', 'Optimal']
colors = ['orange', 'purple', 'blue']

axes[0].bar(labels, costs, color=colors, alpha=0.7, edgecolor='black')
axes[0].set_ylabel('Coût de la solution')
axes[0].set_title('Comparaison des coûts')
axes[0].grid(axis='y', alpha=0.3)

# Ajouter les valeurs sur les barres
for i, (cost, label) in enumerate(zip(costs, labels)):
    axes[0].text(i, cost + 5, f'{cost:.0f}', ha='center', va='bottom', fontweight='bold')

# Amélioration au fil des itérations
improvements = [(initial_cost - cost) / initial_cost * 100 for cost in tabu.best_cost_history]
axes[1].plot(improvements, linewidth=2, color='purple')
axes[1].set_xlabel('Itération')
axes[1].set_ylabel('Amélioration (%)')
axes[1].set_title('Amélioration par rapport à la solution initiale')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=((initial_cost - optimal_cost) / initial_cost * 100),
                color='blue', linestyle='--', label='Optimum', linewidth=2)
axes[1].legend()

plt.tight_layout()
plt.savefig('performance_cvrp_tabu.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nGraphiques sauvegardés:")
print("  - convergence_cvrp_tabu.png")
print("  - solution_cvrp_tabu.png")
print("  - performance_cvrp_tabu.png")

print("\n" + "="*60)
print("EXÉCUTION TERMINÉE")
print("="*60)
