"""
Lecture du fichier d'instance CVRP avec VRPLIB.
Affichage au terminal.
"""
import vrplib
import matplotlib.pyplot as plt
import numpy as np
import time

# Lecture du fichier d'instance VRPLIB
path_file_instance_vrplib = 'code/tests/data/P-n16-k8.vrp'
instance_vrplib = vrplib.read_instance(path_file_instance_vrplib)

# Affichage (terminal) de l'instance VRPLIB
print("--------------------------------")
print("| Affichage de l'instance CVRP |")
print("--------------------------------")
print(f"Nom de l'instance : {instance_vrplib['name']}")
print(f"Commentaire : {instance_vrplib['comment']}")
print(f"Type de problème : {instance_vrplib['type']}")
print(f"Nombre total de sommets : {instance_vrplib['dimension']}")
print(f"Type de poids d'arête : {instance_vrplib['edge_weight_type']}")
print(f"Capacité d'un camion : {instance_vrplib['capacity']}")
print(f"Coordonnées des sommets (x,y) :")

index3 = 0

for i in instance_vrplib['node_coord'] :
    print(f"Sommet {index3 + 1} : {i}")
    index3 = index3 + 1

print("Commande des clients :")

index1 = 0

for i in instance_vrplib['demand'] :
    index1 = index1 + 1
    print(f"Client au sommet {index1} : {i} objet(s)")

print(f"Sommet(s) du/des dépôts : {instance_vrplib['depot']}")
print(f"Poids des arêtes : {instance_vrplib['edge_weight']}")
print(f"Dictionnaire de l'instance : {instance_vrplib.keys()}") # Pour vérifier la structure du dictionnaire



"""
Génération de la solution initiale par rapport au fichier d'instance CVRP récupéré.
Heuristique utilisé : insertion séquentielle gloutonne en fonction du coût de chaque arête
Contrainte : capacité maximale des camions (CVRP)
"""
start1 = time.time()
# Récupération des coordonnées et de la dimension
coord = instance_vrplib['node_coord']
dimension = instance_vrplib['dimension']

# Coordonnées du dépôt
depot_index = 0 # le dépôt est à l'index 0
depot_x, depot_y = coord[depot_index][0], coord[depot_index][1]

# Coordonnées des clients
client_coords = np.delete(coord, depot_index, axis = 0) # suppression de la ligne de coordonnée du dépôt
client_x = client_coords[:, 0] # toute la première colonne pris (coordonnée x)
client_y = client_coords[:, 1] # toute la deuxième colonne pris (coordonnée y)

# On recherche le client le plus proche du dépôt
distance_coords = [] # stocke les distances qui sépare le dépôt à chaque clients

for i in range(len(client_coords)) :
    abs_client_x = abs(client_x[i] - depot_x) # valeur absolue des coordonnées x de chaque client par rapport à la coordonnée x du dépôt
    abs_client_y = abs(client_y[i] - depot_y) # valeur absolue des coordonnées y de chaque client par rapport à la coordonnée y du dépôt
    sum_abs_x_y = abs_client_x + abs_client_y
    distance_coords.append(sum_abs_x_y)

print(f"Liste des distances : {distance_coords}")
print(f"Distance minimale : {min(distance_coords)}") # distance minimale entre le dépôt et le client le plus proche

# Initialisation de la solution initiale
max_capacity = instance_vrplib['capacity']
demand = instance_vrplib['demand']
print(f"Charge de chaque commande : {demand}")

distance = instance_vrplib['edge_weight']
clients_remaining = list(range(len(demand) - 1)) # liste d'index de clients pas encore visité (- 1 pour l'index python)
print(f"Clients restant : {clients_remaining}")

initial_solution = [] # stock une ou plusieurs routes de la solution finale
initial_solution1 = [] # pour la solution initiale sans dépôt

print("Début de la construction de la solution initiale...")

index4 = 1

while clients_remaining :
    print(f"Construction de la route n°{index4}...")

    # Choix du client le plus proche du dépôt parmi les clients restants pour être le meilleur point de départ
    distances_remaining = [distance_coords[i] for i in clients_remaining] # liste de distances de chaque clients restant
    index_min_distance_remaining = np.argmin(distances_remaining)
    client_choosed = clients_remaining[index_min_distance_remaining]
    print(f"Client restant choisi en premier pour la route : {client_choosed}")

    client_choosed_vrplib = client_choosed + 1 # pour l'index VRPLIB
    print(f"Client restant choisi (après indexage) : {client_choosed_vrplib}")

    current_route = [client_choosed_vrplib] # nouvelle route temporaire en construction
    actual_charge = demand[client_choosed_vrplib] # initialisation de la charge d'un nouveau camion actuelle

    clients_remaining.remove(client_choosed)

    # Pour chaque client restant, trouver la meilleure insertion vers la solution initiale
    continue_extension = True
    
    while clients_remaining and continue_extension :
        
        best_insertion_cost = np.inf # coût le plus bas trouvé (démarre avec un nombre infini)
        min_index_cost = None
        best_index = -1 # index de la meilleure insertion
        
        # Recherche des meilleurs client restant à insérer
        for client_index in clients_remaining :
            true_client_index = client_index + 1 # index par rapport au vrplib
            client_demand = demand[true_client_index]   
            
            # Si la demande du client n'est pas trop lourd
            if actual_charge + client_demand <= max_capacity :
                test_route = [0] + current_route + [0] # ajout du dépôt au début et à la fin de la route en construction
                
                for i in range(len(test_route) - 1) :
                    client_i = test_route[i] # index de l'arête de départ
                    client_j = test_route[i + 1] # index de l'arête d'arrivée
                    
                    # Calcul du coût d'insertion : (coût(i -> client) + coût(client -> j)) - coût(i -> j)
                    insertion_distance = distance[client_i, true_client_index] + distance[true_client_index, client_j] - distance[client_i, client_j]
                    
                    # Si le coût trouver est meilleur que celle que l'on a actuellement
                    if insertion_distance < best_insertion_cost :
                        best_insertion_cost = insertion_distance
                        min_index_cost = client_index # index local pour la liste restante
                        best_index = i + 1 # position pour la route en construction 'current_route'

        # Si on a trouver un meilleur coût
        if min_index_cost is not None :
            client_to_add = min_index_cost + 1 # index par rapport au vrplib
            actual_charge = actual_charge + demand[client_to_add] # mise à jour de la charge actuelle du camion
            current_route.insert(best_index - 1, client_to_add) # mise à jour de la route
            clients_remaining.remove(min_index_cost)

        # On ne peut plus rien insérer car il n'y a soit plus de client, 
        # soit les clients restant ont des commandes trop lourdes
        else :
            continue_extension = False 
            print("fin de la construction de la route actuelle")
            
    initial_solution.append([0] + current_route + [0]) # ajout de la route actuelle (avec dépôt au début et à la fin du chemin) à la solution initiale
    initial_solution1.append(current_route) # ajout de la route actuelle (sans dépôt) à la solution initiale
    print(f"Charge finale occupé par le camion : {actual_charge}")
    print(f"Solution initiale actuelle (après construction de la route n°{index4}) : {initial_solution}")
    index4 = index4 + 1

print(f"Routes trouvé pour la solution initiale :")

# Affichage des routes avec les indices VRPLIB
for route_index, route in enumerate(initial_solution) :
    route_corrected = [client_index + 1 for client_index in route] # pour l'index VRPLIB
    print(f"Route n°{route_index + 1}: {route_corrected}")

# Calcul du coût total de la solution initiale
initial_final_cost = 0.0 # coût total de la solution initiale
index5 = 1

print(f"Calcul du coût total de la solution initiale...")

# Parcour de chaque route dans la solution initiale
for route in initial_solution :
    route_cost = 0.0 # coût de la route actuelle
    
    # Parcour les arêtes de la route (de i à j)
    # -1 car le dernier sommet est l'arrivée (le dépôt).
    for i in range(len(route) - 1) :
        start_node_index = route[i] # sommet de départ (i)
        end_node_index = route[i + 1] # sommet d'arrivée (j)
        vertice_cost = distance[start_node_index, end_node_index] # coût direct entre les deux sommets
        route_cost = route_cost + vertice_cost

    initial_final_cost = initial_final_cost + route_cost
    
    print(f"Coût de la route n°{index5} : {route_cost:.0f}")
    index5 = index5 + 1

print(f"Coût total de la solution initiale : {initial_final_cost:.0f}")

# Calcul du CO2 émis
initial_CO2_g_per_km = 900 # gramme de CO2 par km
initial_total_CO2_emmited_g = initial_final_cost * initial_CO2_g_per_km
initial_total_CO2_emmited_kg = initial_total_CO2_emmited_g / 1000 # conversion kg
print(f"Total de CO2 (en kg) émis pour la solution initiale : {initial_total_CO2_emmited_kg:.2f} kg")



"""
Affichage graphique de la solution initiale CVRP.
"""
# Affichage graphique des coordonnées du dépôt et des clients
plt.figure(figsize = (10, 8))
plt.plot(depot_x, depot_y, 's', color = 'red', markersize = 10, label = 'Dépôt', zorder = 5)
plt.plot(client_x, client_y, 'o', color = 'blue', markersize = 5, label = 'Clients')

# Numérotation des sommets
for i in range(dimension) :
    x, y = coord[i]
    plt.text(x + 1, y + 1, str(i + 1), fontsize = 8) # str(i + 1) pour s'adapter au format VRPLIB (l'index démarre à 1)

# Personalisation d'une palette de couleur
cmap = plt.colormaps.get_cmap('hsv') # choix de la colormap 'hsv'
indices = np.linspace(0, 1, len(initial_solution) + 1) # choix d'un nombre de couleurs aléatoire pour chaque route
custom_colors = cmap(indices)

# Personnalisation des routes
for route_index, route in enumerate(initial_solution) :

    # Récupération des coordonnées (x, y) de chaque sommets
    route_coord_x = [coord[node_index][0] for node_index in route]
    route_coord_y = [coord[node_index][1] for node_index in route]

    # Affichage graphique de chaque route
    plt.plot(route_coord_x, route_coord_y, 
            color = custom_colors[route_index], # coloration unique de chaque route 
            linestyle = '-', 
            linewidth = 2, 
            alpha = 0.8, 
            label = f'Route {route_index + 1}') # Légende pour une route

# Personnalisation des légendes
plt.title(f"Solution de l'instance VRP : {instance_vrplib['name']} au coût totale {initial_final_cost:.0f}")
plt.xlabel("Coordonnée X")
plt.ylabel("Coordonnée Y")
plt.legend()
plt.axis('equal') # assure une échelle correcte
plt.show()



"""
Génération de la solution finale optimisé de la solution initiale CVRP.
Algorithme utilisé : !!! à remplir (Mickaël) !!!
"""
# !!! Implémentez chacun vos algorithmes ici mes chères collègues :-) (Mickaël) !!!

# recherche tabou.py
import random
import numpy as np

#liste des meilleur cout
test_coût = []
N = 0

def compute_distance_matrix(coords):
    """Calcule la matrice de distances euclidiennes."""
    # Convertir en numpy array pour permettre les opérations vectorielles
    coords_np = np.array(coords, dtype=float)
    n = len(coords_np)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = np.linalg.norm(coords_np[i] - coords_np[j])
    return dist

# Évaluation de la solution
def evaluate_solution(routes, demands, capacity, dist_matrix):
    global N
    """Évalue une solution CVRP (coût = distance totale)."""
    total_dist = 0.0
    for route in routes:
        load = sum(demands[i] for i in route)
        
        N = N + 1
        if load > capacity:
            return float('inf')  # Solution non faisable
        prev = 0  # Dépôt
        for node in route:
            total_dist += dist_matrix[prev][node]
            prev = node
        total_dist += dist_matrix[prev][0]  # Retour au dépôt
    return total_dist

# Génération des voisins
def get_neighbors(solution, instance, tabu_list):
    """Génère des voisins en évitant les mouvements tabous."""
    coords = instance["node_coord"]
    demands = instance["demand"]
    capacity = instance["capacity"]
    dist_matrix = compute_distance_matrix(coords)
    neighbors = []
    n = len(coords)

    # Opérateur : swap (échanger deux clients)
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            for a in range(len(solution[i])):
                for b in range(len(solution[j])):
                    new_sol = [route[:] for route in solution]
                    # Échanger les clients
                    new_sol[i][a], new_sol[j][b] = new_sol[j][b], new_sol[i][a]
                    move = ("swap", i, a, j, b)
                    if move not in tabu_list:
                        cost = evaluate_solution(new_sol, demands, capacity, dist_matrix)
                        if cost < float('inf'):
                            neighbors.append((new_sol, cost, move))

    # Opérateur : relocate (déplacer un client)
    for i in range(len(solution)):
        for j in range(len(solution)):
            if i == j:
                continue
            for a in range(len(solution[i])):
                for b in range(len(solution[j]) + 1):
                    new_sol = [route[:] for route in solution]
                    client = new_sol[i].pop(a)
                    new_sol[j].insert(b, client)
                    move = ("relocate", i, a, j, b)
                    if move not in tabu_list:
                        cost = evaluate_solution(new_sol, demands, capacity, dist_matrix)
                        if cost < float('inf'):
                            neighbors.append((new_sol, cost, move))

    return neighbors

# Résolution du CVRP avec la Recherche Tabou
def solve_vrp(instance, time_limit=45, seed=42):
    """
    Résout le CVRP avec la Recherche Tabou.
    Retourne un dictionnaire {"routes": [...], "cost": float}
    """
    random.seed(seed)
    coords = instance["node_coord"]
    demands = instance["demand"]
    capacity = instance["capacity"]
    dist_matrix = compute_distance_matrix(coords)

    # Solution initiale
    current_sol = initial_solution1
    current_cost = evaluate_solution(current_sol, demands, capacity, dist_matrix)
    best_sol = [r[:] for r in current_sol]
    best_cost = current_cost
    test_coût.append(best_cost)
    # Liste tabou (stocke les derniers mouvements interdits)
    tabu_list = []
    tabu_tenure = 20000  # Durée d'interdiction

    import time
    start = time.time()

    while time.time() - start < time_limit:
        neighbors = get_neighbors(current_sol, instance, tabu_list)
        if not neighbors:
            break

        # Choisir le meilleur voisin (même s'il est pire que current)
        neighbors.sort(key=lambda x: x[1])
        best_neighbor, best_neighbor_cost, move = neighbors[0]

        # Mettre à jour la meilleure solution globale
        if best_neighbor_cost < best_cost:
            best_sol = [r[:] for r in best_neighbor]
            best_cost = best_neighbor_cost
            test_coût.append(best_cost)
        
        # Accepter le voisin
        current_sol = best_neighbor
        current_cost = best_neighbor_cost

        # Ajouter le mouvement à la liste tabou
        tabu_list.append(move)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

    return {"routes": best_sol, "cost": best_cost}
final_solution_optimized = solve_vrp(instance_vrplib, time_limit= 30, seed=42)


"""
Lecture de la solution finale CVRP avec VRPLIB.
Affichage au terminal.
"""
# Lecture de la solution VRPLIB
path_file_solution_vrplib = 'code/tests/data/P-n16-k8.sol'
solution_vrplib = vrplib.read_solution(path_file_solution_vrplib)

# Afichage (terminal) de la solution VRPLIB
print("---------------------------------")
print("| Affichage de la solution CVRP |")
print("---------------------------------")

final_solution = final_solution_optimized['routes']

final_solution_with_added_depot_python = [] # pour stocker les routes avec les vrais indices Python + dépot
final_solution_with_added_depot_vrplib = [] # pour stocker les routes avec les vrais indices VRPLIB + dépot

print(f"Solution finale actuelle : {final_solution}")
print(f"Routes trouvé pour la solution finale :")

# Affichage des routes avec les indices VRPLIB
for route_index, route in enumerate(final_solution) :

    # Avec index Python
    route_corrected_python = [client_index + 0 for client_index in route] # pour l'index Python
    route_with_added_depot_python = [0] + route_corrected_python + [0] # ajout du dépôt au début et à la fin
    final_solution_with_added_depot_python.append(route_with_added_depot_python)
    
    # Avec index VRPLIB
    route_corrected_vrplib = [client_index + 1 for client_index in route] # pour l'index VRPLIB
    route_with_added_depot_vrplib = [1] + route_corrected_vrplib + [1] # ajout du dépôt au début et à la fin
    final_solution_with_added_depot_vrplib.append(route_with_added_depot_vrplib)
    print(f"Route n°{route_index + 1}: {route_with_added_depot_vrplib}")

print(f"Dictionnaire de la solution : {final_solution_optimized.keys()}") # pour vérifier la" structure du dictionnaire
print(f"Routes standard : {final_solution_with_added_depot_python}")
print(f"Routes corrigé : {final_solution_with_added_depot_vrplib}")

# Calcul du coût total de la solution finale
final_final_cost = 0.0 # coût total de la solution finale
final_final_cost = final_solution_optimized['cost']
index6 = 1
"""
print(f"Calcul du coût total de la solution finale...")

# Parcour de chaque route dans la solution finale
for route in final_solution_with_added_depot_python :
    route_cost = 0.0 # coût de la route actuelle
    
    # Parcour les arêtes de la route (de i à j)
    # -1 car le dernier sommet est l'arrivée (le dépôt).
    for i in range(len(route) - 1) :
        start_node_index = route[i] # sommet de départ (i)
        end_node_index = route[i + 1] # sommet d'arrivée (j)
        vertice_cost = distance[start_node_index, end_node_index] # coût direct entre les deux sommets
        route_cost = route_cost+ vertice_cost

    final_final_cost = final_final_cost + route_cost
    
    print(f"Coût de la route n°{index6} : {route_cost:.0f}")
    index6 = index6 + 1
"""
print(f"Coût total de la solution finale : {final_final_cost:.0f}")

# Calcul du CO2 émis
final_CO2_g_per_km = 900 # gramme de CO2 par km
final_total_CO2_emmited_g = final_final_cost * final_CO2_g_per_km
final_total_CO2_emmited_kg = final_total_CO2_emmited_g / 1000 # conversion kg
print(f"Total de CO2 (en kg) émis pour la solution finale : {final_total_CO2_emmited_kg:.2f} kg")

# Calcul de l'écart entre le coût de la solution finale et la solution la plus optimal
true_final_cost = solution_vrplib['cost']
cost_gap = abs((( final_final_cost - true_final_cost) / final_final_cost) * 100)
print(f"Écart de coût entre la solution finale trouvé et celle qui est la plus optimisé : {cost_gap:.2f} %")

"""
Affichage graphique de la solution CVRP.
"""

# Affichage graphique des coordonnées du dépôt et des clients
plt.figure(figsize = (10, 8))
plt.plot(depot_x, depot_y, 's', color = 'red', markersize = 10, label = 'Dépôt', zorder = 5)
plt.plot(client_x, client_y, 'o', color = 'blue', markersize = 5, label = 'Clients')

# Numérotation des sommets
for i in range(dimension) :
    x, y = coord[i]
    plt.text(x + 1, y + 1, str(i + 1), fontsize = 8) # str(i + 1) pour s'adapter au format VRPLIB (l'index démarre à 1)

# Personalisation d'une palette de couleur
cmap = plt.colormaps.get_cmap('hsv') # choix de la colormap 'hsv'
indices = np.linspace(0, 1, len(final_solution_optimized['routes']) + 1) # choix d'un nombre de couleurs aléatoire pour chaque route
custom_colors = cmap(indices)

# Personnalisation des routes
for route_index, route in enumerate(final_solution_optimized['routes']):
    full_route = [0] + route + [0] # route complet (coordonnées)

    # Récupération des coordonnées (x, y) de chaque sommets
    route_coord_x = [coord[node_index][0] for node_index in full_route]
    route_coord_y = [coord[node_index][1] for node_index in full_route]

    # Affichage graphique de chaque route
    plt.plot(route_coord_x, route_coord_y, 
            color = custom_colors[route_index], # coloration unique de chaque route 
            linestyle = '-', 
            linewidth = 2, 
            alpha = 0.8, 
            label = f'Route {route_index + 1}') # Légende pour une route

# Personnalisation des légendes
cout_total = f"{final_solution_optimized['cost']:.2f}"
plt.title(f"Solution de l'instance VRP : {instance_vrplib['name']} au coût totale {cout_total}")
plt.xlabel("Coordonnée X")
plt.ylabel("Coordonnée Y")
plt.legend()
plt.axis('equal') # assure une échelle correcte
plt.show()

#pour afficher la convergence
def plot_convergence(cost_history, initial_cost, final_cost):
    """Affiche la courbe de convergence."""
    plt.figure(figsize=(12, 6))

    iterations = range(len(cost_history))
    plt.plot(iterations, cost_history, linewidth=2, color='#3498db', label='Tabou')
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
    plt.title('Convergence de l\'algorithme recherche tabou', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

#  Convergence
print(N)


cost_history = test_coût # Exemple de données
initial_cost = initial_final_cost
print(initial_cost)
plot_convergence(cost_history, initial_cost, final_final_cost)    