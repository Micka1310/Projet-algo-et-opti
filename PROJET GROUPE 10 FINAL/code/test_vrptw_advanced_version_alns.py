"""
Lecture du fichier d'instance VRPTW avec VRPLIB.
Affichage au terminal.
"""
import vrplib
import matplotlib.pyplot as plt
import numpy as np
import time

# Lecture du fichier d'instance Solomon
path_file_instance_vrplib = 'code/tests/data/cvrplib/Vrp-Set-Solomon/C208.txt'
instance_vrplib = vrplib.read_instance(path_file_instance_vrplib, instance_format = "solomon")

# Affichage (terminal) de l'instance Solomon
print("---------------------------------")
print("| Affichage de l'instance VRPTW |")
print("---------------------------------")
print(f"Nom de l'instance : {instance_vrplib['name']}")
print(f"Nombre de véhicule : {instance_vrplib['vehicles']}")
print(f"Capacité d'un camion : {instance_vrplib['capacity']}")
print(f"Coordonnées des sommets (x,y) :")

temps_debut = time.time()
index3 = 0

for i in instance_vrplib['node_coord'] :
    print(f"Sommet {index3} : {i}")
    index3 = index3 + 1

print("Commande des clients :")

index1 = 0

for i in instance_vrplib['demand'] :
    print(f"Client au sommet {index1} : {i} objet(s)")
    index1 = index1 + 1

index7 = 0

print(f"Fenêtre temporelle : ")

for i in instance_vrplib['time_window'] :
    print(f"Client au sommet {index7} : {i} minutes")
    index7 = index7 + 1

print(f"Temps de service pour chaque client (minutes) : {instance_vrplib['service_time']}")
print(f"Poids des arêtes : {instance_vrplib['edge_weight']}")
print(f"Dictionnaire de l'instance : {instance_vrplib.keys()}")



"""
Génération de la solution initiale par rapport au fichier d'instance VRPTW récupéré.
Heuristique utilisé : insertion séquentielle gloutonne en fonction du coût de chaque arête
Contrainte avancé : Fenêtre temporelle (VRPTW)
"""
# Récupération des coordonnées (x,y)
coord = instance_vrplib['node_coord']

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

clients_remaining = list(range(len(demand) - 1)) # liste d'index de clients pas encore visité (- 1 pour l'index python)
print(f"Clients restant (index Python) : {clients_remaining}")

initial_solution = [] # stock une ou plusieurs routes de la solution finale

print("Début de la construction de la solution initiale...")

# Fonction qui permet de vérifier la contrainte temporelle
# Les paramètres suivant sont : 
# - 'temporal_route' : tableau d'indice (index VRPLIB) de sommet d'une route sans dépôt
# - 'time_windows' : tableau de fenêtre de temps de l'instance de donnée
# - 'time_service' : tableau des temps de service pour chaque client de l'instance de donnée
# - 'distance' : tableau des poids d'arête de l'instance de donnée
def check_temporal_time(temporal_route, time_windows, service_time, distance) :
    full_route = [0] + temporal_route + [0] # ajout du dépôt
    current_time = 0.0 # heure de départ du dépôt
    is_feasible = True # True si la contrainte est respecter, sinon ce sera False

    # pour chaque sommet de la route
    for i in range(len(full_route) - 1):
        
        # définition de l'arête
        node_i = full_route[i]
        node_j = full_route[i + 1]
        travel_time = distance[node_i, node_j] # poid de l'arête
        arrival_time_j = current_time + travel_time # heure d'arrivé à j
        
        # Fenêtre temporelle au sommet j
        earliest_j = time_windows[node_j, 0] # première colonne
        latest_j = time_windows[node_j, 1] # deuxième colonne

        # Si on arrive après la fenêtre maximale
        if arrival_time_j > latest_j: 
            is_feasible = False

        # Définition du début de service autorisé au sommet j
        start_service_j = max(arrival_time_j, earliest_j) 

        # Temps après le service effectuer au sommet j
        end_service_j = start_service_j + service_time[node_j]
        
        # Le temps de départ à partir du sommet j et égale au temps après service depuis celui-ci
        current_time = end_service_j

    #return True
    return is_feasible

distance = instance_vrplib['edge_weight']
time_windows = instance_vrplib['time_window']
service_time = instance_vrplib['service_time']
index4 = 1

while clients_remaining :
    print(f"Construction de la route n°{index4}...")

    # Choix du client le plus proche du dépôt parmi les clients restants pour être le meilleur point de départ
    distances_remaining = [distance_coords[i] for i in clients_remaining] # liste de distances de chaque clients restant
    index_min_distance_remaining = np.argmin(distances_remaining)
    client_choosed = clients_remaining[index_min_distance_remaining]
    print(f"Client restant choisi en premier pour la route : {client_choosed}")

    client_choosed_vrplib = client_choosed + 1 # pour l'index VRPLIB
    print(f"Client restant choisi (index VRPLIB) : {client_choosed_vrplib}")

    current_route = [client_choosed_vrplib] # nouvelle route temporaire en construction
    actual_charge = demand[client_choosed_vrplib] # initialisation de la charge d'un nouveau camion actuel

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

                    # Route temporaire pour tester la contrainte de temps
                    temporal_route = current_route[:] # Copie de la route actuelle (sans dépôt)
                    temporal_route.insert(i, true_client_index) # Insertion du client à la position i

                    # Vérification de la contrainte temporelle
                    time_feasible = check_temporal_time(temporal_route, time_windows, service_time, distance)

                    # La fenêtre temporelle est respecter
                    if time_feasible :
                    
                        # Calcul du coût d'insertion : (coût(i -> client) + coût(client -> j)) - coût(i -> j)
                        insertion_distance = distance[client_i, true_client_index] + distance[true_client_index, client_j] - distance[client_i, client_j]
                        
                        # Si le coût trouver est meilleur que celle que l'on a actuellement
                        if insertion_distance < best_insertion_cost :
                            best_insertion_cost = insertion_distance
                            min_index_cost = client_index # index local pour la liste restante
                            best_index = i # position pour la route en construction 'current_route'

        # Si on a trouver un meilleur coût
        if min_index_cost is not None :
            client_to_add = min_index_cost + 1 # index par rapport au vrplib
            actual_charge = actual_charge + demand[client_to_add] # mise à jour de la charge actuelle du camion
            current_route.insert(best_index, client_to_add) # mise à jour de la route
            clients_remaining.remove(min_index_cost)

        # On ne peut plus rien insérer car il n'y a soit plus de client, 
        # soit les clients restant ont des commandes trop lourdes
        else :
            continue_extension = False 
            print("fin de la construction de la route actuelle")
            
    initial_solution.append([0] + current_route + [0]) # ajout de la route actuelle (avec dépôt au début et à la fin du chemin) à la solution initiale
    print(f"Charge finale occupé par le camion : {actual_charge}")
    print(f"Solution initiale actuelle après construction de la route n°{index4} (index Python) :")

    for i in range (len(initial_solution)) :
        print(f"Route n°{i + 1} : {initial_solution[i]}")
    index4 = index4 + 1

print(f"Routes trouvé pour la solution initiale (index VRPLIB) :")

# Affichage des routes avec les indices VRPLIB
for route_index, route in enumerate(initial_solution) :
    route_corrected = [client_index for client_index in route] # pour l'index VRPLIB
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
Affichage graphique de la solution initiale VRPTW.
"""
# Affichage graphique des coordonnées du dépôt et des clients
plt.figure(figsize = (10, 8))
plt.plot(depot_x, depot_y, 's', color = 'red', markersize = 10, label = 'Dépôt', zorder = 5)
plt.plot(client_x, client_y, 'o', color = 'blue', markersize = 5, label = 'Clients')

dimension = len(instance_vrplib['node_coord'])

# Numérotation des sommets
for i in range(dimension) :
    x, y = coord[i]
    plt.text(x + 1, y + 1, str(i), fontsize = 8)

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
plt.title(f"Solution de l'instance VRPTW : {instance_vrplib['name']} au coût totale {initial_final_cost:.0f}")
plt.xlabel("Coordonnée X")
plt.ylabel("Coordonnée Y")
plt.legend()
plt.axis('equal') # assure une échelle correcte
plt.show()



"""
Génération de la solution finale optimisé de la solution initiale VRPTW.
Algorithme utilisé : ALNS (Adaptive Large Neighborhood Search)
"""
import random
import math
import copy
import time as time_module

print("\n" + "="*60)
print("OPTIMISATION AVEC ALNS (Adaptive Large Neighborhood Search)")
print("="*60)

# ==================== CLASSE ALNS POUR VRPTW ====================
class ALNS_VRPTW:
    """Adaptive Large Neighborhood Search pour VRPTW"""

    def __init__(self, instance, initial_solution):
        self.instance = instance
        self.distance = instance['edge_weight']
        self.demand = instance['demand']
        self.capacity = instance['capacity']
        self.time_windows = instance['time_window']
        self.service_time = instance['service_time']
        self.initial_solution = copy.deepcopy(initial_solution)

        # Paramètres ALNS
        self.destroy_weights = {'random': 1.0, 'worst': 1.0, 'shaw': 1.0, 'time_oriented': 1.0}
        self.repair_weights = {'greedy': 1.0, 'regret': 1.0}

        # Scores d'adaptation
        self.sigma1 = 20  # Nouvelle meilleure solution
        self.sigma2 = 10  # Solution acceptée améliorant la solution actuelle
        self.sigma3 = 5   # Solution acceptée sans amélioration

        # Historique
        self.best_cost_history = []
        self.current_cost_history = []

    def calculate_solution_cost(self, solution):
        """Calcule le coût total d'une solution"""
        total_cost = 0.0
        for route in solution:
            for i in range(len(route) - 1):
                total_cost += self.distance[route[i], route[i + 1]]
        return total_cost

    def check_time_window_feasibility(self, route):
        """Vérifie si une route respecte les fenêtres temporelles"""
        current_time = 0.0

        for i in range(len(route) - 1):
            node_i = route[i]
            node_j = route[i + 1]

            travel_time = self.distance[node_i, node_j]
            arrival_time_j = current_time + travel_time

            earliest_j = self.time_windows[node_j, 0]
            latest_j = self.time_windows[node_j, 1]

            if arrival_time_j > latest_j:
                return False

            start_service_j = max(arrival_time_j, earliest_j)
            end_service_j = start_service_j + self.service_time[node_j]
            current_time = end_service_j

        return True

    def is_feasible_route(self, route):
        """Vérifie si une route est faisable (capacité + fenêtres temporelles)"""
        total_demand = sum(self.demand[customer] for customer in route if customer != 0)
        if total_demand > self.capacity:
            return False
        return self.check_time_window_feasibility(route)

    def remove_empty_routes(self, solution):
        """Retire les routes vides"""
        return [route for route in solution if len(route) > 2]

    # ============== OPÉRATEURS DE DESTRUCTION ==============

    def random_removal(self, solution, num_remove):
        """Retire des clients aléatoirement"""
        solution_copy = copy.deepcopy(solution)
        removed = []

        all_customers = []
        for route_idx, route in enumerate(solution_copy):
            for pos, customer in enumerate(route):
                if customer != 0:
                    all_customers.append((route_idx, pos, customer))

        if len(all_customers) == 0:
            return solution_copy, []

        num_to_remove = min(num_remove, len(all_customers))
        customers_to_remove = random.sample(all_customers, num_to_remove)
        customers_to_remove.sort(key=lambda x: (x[0], x[1]), reverse=True)

        for route_idx, pos, customer in customers_to_remove:
            solution_copy[route_idx].pop(pos)
            removed.append(customer)

        return self.remove_empty_routes(solution_copy), removed

    def worst_removal(self, solution, num_remove):
        """Retire les clients les plus coûteux"""
        solution_copy = copy.deepcopy(solution)
        removed = []

        customer_costs = []
        for route_idx, route in enumerate(solution_copy):
            for pos in range(1, len(route) - 1):
                customer = route[pos]
                prev_customer = route[pos - 1]
                next_customer = route[pos + 1]

                cost_with = self.distance[prev_customer, customer] + self.distance[customer, next_customer]
                cost_without = self.distance[prev_customer, next_customer]
                saving = cost_with - cost_without

                customer_costs.append((saving, route_idx, pos, customer))

        if len(customer_costs) == 0:
            return solution_copy, []

        customer_costs.sort(reverse=True)
        num_to_remove = min(num_remove, len(customer_costs))

        customers_to_remove = customer_costs[:num_to_remove]
        customers_to_remove.sort(key=lambda x: (x[1], x[2]), reverse=True)

        for _, route_idx, pos, customer in customers_to_remove:
            solution_copy[route_idx].pop(pos)
            removed.append(customer)

        return self.remove_empty_routes(solution_copy), removed

    def shaw_removal(self, solution, num_remove):
        """Retire des clients similaires (proches géographiquement)"""
        solution_copy = copy.deepcopy(solution)
        removed = []

        all_customers = []
        for route_idx, route in enumerate(solution_copy):
            for pos, customer in enumerate(route):
                if customer != 0:
                    all_customers.append((route_idx, pos, customer))

        if len(all_customers) == 0:
            return solution_copy, []

        seed_customer = random.choice(all_customers)[2]
        removed.append(seed_customer)

        customers_with_distance = []
        for route_idx, pos, customer in all_customers:
            if customer != seed_customer:
                dist = self.distance[seed_customer, customer]
                customers_with_distance.append((dist, route_idx, pos, customer))

        customers_with_distance.sort()

        num_additional = min(num_remove - 1, len(customers_with_distance))
        for i in range(num_additional):
            removed.append(customers_with_distance[i][3])

        for route in solution_copy:
            route[:] = [c for c in route if c not in removed]

        return self.remove_empty_routes(solution_copy), removed

    def time_oriented_removal(self, solution, num_remove):
        """Retire des clients en fonction de leurs contraintes temporelles"""
        solution_copy = copy.deepcopy(solution)
        removed = []

        all_customers = []
        for route_idx, route in enumerate(solution_copy):
            for pos, customer in enumerate(route):
                if customer != 0:
                    tw_width = self.time_windows[customer, 1] - self.time_windows[customer, 0]
                    all_customers.append((tw_width, route_idx, pos, customer))

        if len(all_customers) == 0:
            return solution_copy, []

        all_customers.sort()
        num_to_remove = min(num_remove, len(all_customers))

        customers_to_remove = all_customers[:num_to_remove]
        customers_to_remove.sort(key=lambda x: (x[1], x[2]), reverse=True)

        for _, route_idx, pos, customer in customers_to_remove:
            solution_copy[route_idx].pop(pos)
            removed.append(customer)

        return self.remove_empty_routes(solution_copy), removed

    # ============== OPÉRATEURS DE RÉPARATION ==============

    def greedy_insertion(self, solution, removed_customers):
        """Insère les clients retirés de manière gloutonne"""
        solution_copy = copy.deepcopy(solution)
        remaining = removed_customers[:]

        while remaining:
            best_cost = float('inf')
            best_customer = None
            best_route_idx = None
            best_position = None

            for customer in remaining:
                for route_idx, route in enumerate(solution_copy):
                    current_demand = sum(self.demand[c] for c in route if c != 0)

                    if current_demand + self.demand[customer] <= self.capacity:
                        for pos in range(1, len(route)):
                            prev = route[pos - 1]
                            next_node = route[pos]

                            insertion_cost = (self.distance[prev, customer] +
                                            self.distance[customer, next_node] -
                                            self.distance[prev, next_node])

                            test_route = route[:pos] + [customer] + route[pos:]

                            if self.check_time_window_feasibility(test_route):
                                if insertion_cost < best_cost:
                                    best_cost = insertion_cost
                                    best_customer = customer
                                    best_route_idx = route_idx
                                    best_position = pos

                new_route = [0, customer, 0]
                if self.check_time_window_feasibility(new_route):
                    new_route_cost = 2 * self.distance[0, customer]
                    if new_route_cost < best_cost or best_customer is None:
                        best_customer = customer
                        best_route_idx = -1
                        best_position = 1

            if best_customer is None:
                break

            if best_route_idx == -1:
                solution_copy.append([0, best_customer, 0])
            else:
                solution_copy[best_route_idx].insert(best_position, best_customer)

            remaining.remove(best_customer)

        return solution_copy

    def regret_insertion(self, solution, removed_customers):
        """Insère les clients avec un critère de regret"""
        solution_copy = copy.deepcopy(solution)
        remaining = removed_customers[:]

        while remaining:
            max_regret = -float('inf')
            best_customer = None
            best_insertion = None

            for customer in remaining:
                insertion_costs = []

                for route_idx, route in enumerate(solution_copy):
                    current_demand = sum(self.demand[c] for c in route if c != 0)

                    if current_demand + self.demand[customer] <= self.capacity:
                        for pos in range(1, len(route)):
                            prev = route[pos - 1]
                            next_node = route[pos]

                            cost = (self.distance[prev, customer] +
                                   self.distance[customer, next_node] -
                                   self.distance[prev, next_node])

                            test_route = route[:pos] + [customer] + route[pos:]

                            if self.check_time_window_feasibility(test_route):
                                insertion_costs.append((cost, route_idx, pos))

                new_route = [0, customer, 0]
                if self.check_time_window_feasibility(new_route):
                    new_route_cost = 2 * self.distance[0, customer]
                    insertion_costs.append((new_route_cost, -1, 1))

                if len(insertion_costs) >= 2:
                    insertion_costs.sort()
                    regret = insertion_costs[1][0] - insertion_costs[0][0]

                    if regret > max_regret:
                        max_regret = regret
                        best_customer = customer
                        best_insertion = insertion_costs[0]
                elif len(insertion_costs) == 1:
                    if best_customer is None:
                        best_customer = customer
                        best_insertion = insertion_costs[0]

            if best_customer is None:
                break

            cost, route_idx, pos = best_insertion
            if route_idx == -1:
                solution_copy.append([0, best_customer, 0])
            else:
                solution_copy[route_idx].insert(pos, best_customer)

            remaining.remove(best_customer)

        return solution_copy

    def select_operator(self, weights):
        """Sélectionne un opérateur selon les poids"""
        total = sum(weights.values())
        r = random.uniform(0, total)
        cumsum = 0
        for operator, weight in weights.items():
            cumsum += weight
            if r <= cumsum:
                return operator
        return list(weights.keys())[-1]

    def update_weights(self, operator_type, operator_name, score, segment_size):
        """Met à jour les poids des opérateurs"""
        if operator_type == 'destroy':
            weights = self.destroy_weights
        else:
            weights = self.repair_weights

        weights[operator_name] = 0.8 * weights[operator_name] + 0.2 * score

    def simulated_annealing_acceptance(self, current_cost, new_cost, temperature):
        """Critère d'acceptation du Simulated Annealing"""
        if new_cost < current_cost:
            return True
        delta = new_cost - current_cost
        probability = math.exp(-delta / temperature)
        return random.random() < probability

    def run(self, max_iterations=500, destruction_rate=0.25, segment_size=100,
            initial_temp=None, cooling_rate=0.99975):
        """Exécute l'algorithme ALNS"""

        start_time = time_module.time()

        # Solution initiale
        current_solution = copy.deepcopy(self.initial_solution)
        best_solution = copy.deepcopy(current_solution)

        current_cost = self.calculate_solution_cost(current_solution)
        best_cost = current_cost

        # Température initiale
        if initial_temp is None:
            initial_temp = 0.05 * current_cost
        temperature = initial_temp

        print(f"\nDebut ALNS:")
        print(f"  Cout initial: {current_cost:.2f}")
        print(f"  Temperature initiale: {temperature:.2f}")
        print(f"  Taux de destruction: {destruction_rate:.0%}")

        for iteration in range(max_iterations):
            # Sélection des opérateurs
            destroy_op = self.select_operator(self.destroy_weights)
            repair_op = self.select_operator(self.repair_weights)

            # Calcul du nombre de clients à retirer
            total_customers = sum(1 for route in current_solution for c in route if c != 0)
            num_remove = max(1, int(destruction_rate * total_customers))

            # Destruction
            if destroy_op == 'random':
                destroyed_solution, removed = self.random_removal(current_solution, num_remove)
            elif destroy_op == 'worst':
                destroyed_solution, removed = self.worst_removal(current_solution, num_remove)
            elif destroy_op == 'shaw':
                destroyed_solution, removed = self.shaw_removal(current_solution, num_remove)
            else:  # time_oriented
                destroyed_solution, removed = self.time_oriented_removal(current_solution, num_remove)

            # Réparation
            if repair_op == 'greedy':
                new_solution = self.greedy_insertion(destroyed_solution, removed)
            else:  # regret
                new_solution = self.regret_insertion(destroyed_solution, removed)

            # Évaluation
            new_cost = self.calculate_solution_cost(new_solution)

            # Mise à jour des poids
            score = 0
            if new_cost < best_cost:
                best_solution = copy.deepcopy(new_solution)
                best_cost = new_cost
                current_solution = copy.deepcopy(new_solution)
                current_cost = new_cost
                score = self.sigma1
            elif new_cost < current_cost:
                current_solution = copy.deepcopy(new_solution)
                current_cost = new_cost
                score = self.sigma2
            elif self.simulated_annealing_acceptance(current_cost, new_cost, temperature):
                current_solution = copy.deepcopy(new_solution)
                current_cost = new_cost
                score = self.sigma3

            if score > 0:
                self.update_weights('destroy', destroy_op, score, segment_size)
                self.update_weights('repair', repair_op, score, segment_size)

            # Historique
            self.best_cost_history.append(best_cost)
            self.current_cost_history.append(current_cost)

            # Refroidissement
            temperature *= cooling_rate

            # Affichage de la progression
            if (iteration + 1) % 500 == 0:
                elapsed = time_module.time() - start_time
                improvement = ((self.calculate_solution_cost(self.initial_solution) - best_cost) /
                             self.calculate_solution_cost(self.initial_solution) * 100)
                print(f"  Iteration {iteration + 1}/{max_iterations} | "
                      f"Meilleur cout: {best_cost:.2f} | "
                      f"Amelioration: {improvement:.2f}% | "
                      f"Temps: {elapsed:.1f}s")

        elapsed_time = time_module.time() - start_time

        print(f"\nOptimisation ALNS terminee!")
        print(f"  Cout final: {best_cost:.2f}")
        print(f"  Nombre de routes: {len(best_solution)}")
        print(f"  Temps total: {elapsed_time:.1f}s")

        return {
            'best_solution': best_solution,
            'best_cost': best_cost,
            'initial_cost': self.calculate_solution_cost(self.initial_solution),
            'time': elapsed_time
        }

    def plot_convergence(self, filename='convergence_vrptw_alns.png'):
        """Génère le graphique de convergence"""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.best_cost_history, linewidth=2, color='green', label='Meilleure solution')
        plt.plot(self.current_cost_history, linewidth=1, alpha=0.5, color='orange', label='Solution courante')
        plt.xlabel('Iteration')
        plt.ylabel('Cout')
        plt.title('Convergence ALNS')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        initial_cost = self.calculate_solution_cost(self.initial_solution)
        improvements = [(initial_cost - cost) / initial_cost * 100 for cost in self.best_cost_history]
        plt.plot(improvements, linewidth=2, color='purple')
        plt.xlabel('Iteration')
        plt.ylabel('Amelioration (%)')
        plt.title('Amelioration par rapport a la solution initiale')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nGraphique de convergence sauvegarde: {filename}")

# ==================== EXÉCUTION DE L'ALGORITHME ====================

# Création de l'instance ALNS
alns_vrptw = ALNS_VRPTW(instance_vrplib, initial_solution)

# Exécution de l'algorithme
results_alns_vrptw = alns_vrptw.run(
    max_iterations=500,
    destruction_rate=0.25,
    segment_size=100,
    cooling_rate=0.99975
)

# Mise à jour de la solution pour la suite du script
final_solution_alns_vrptw = results_alns_vrptw['best_solution']
final_cost_alns_vrptw = results_alns_vrptw['best_cost']

# Lecture de la solution optimale
path_file_solution_vrplib = 'code/tests/data/cvrplib/Vrp-Set-Solomon/C208.sol'
solution_vrplib_optimal = vrplib.read_solution(path_file_solution_vrplib)
optimal_cost_vrptw = solution_vrplib_optimal['cost']

# Calcul du Gap
gap_alns_vrptw = ((final_cost_alns_vrptw - optimal_cost_vrptw) / optimal_cost_vrptw) * 100

print("\n" + "="*60)
print("EVALUATION DE LA PERFORMANCE")
print("="*60)
print(f"\nCout optimal connu:              {optimal_cost_vrptw:.2f}")
print(f"Cout solution initiale:          {initial_final_cost:.2f}")
print(f"Cout solution ALNS:              {final_cost_alns_vrptw:.2f}")
print(f"Gap par rapport a l'optimum:     {gap_alns_vrptw:.2f}%")
print(f"Amelioration vs initial:         {((initial_final_cost - final_cost_alns_vrptw) / initial_final_cost * 100):.2f}%")

if gap_alns_vrptw < 5:
    print(f"\nOBJECTIF ATTEINT! Gap < 5%")
else:
    print(f"\nGap superieur a 5%, optimisation supplementaire necessaire")

# Calcul du CO2 pour la solution ALNS
CO2_alns_vrptw_kg = (final_cost_alns_vrptw * 900) / 1000
print(f"\nCO2 emis (solution ALNS):        {CO2_alns_vrptw_kg:.2f} kg")
print(f"Reduction de CO2:                {((initial_final_cost - final_cost_alns_vrptw) * 900 / 1000):.2f} kg")

# Graphiques de convergence
alns_vrptw.plot_convergence('convergence_vrptw_alns.png')

# Graphique de la solution ALNS
plt.figure(figsize=(10, 8))
plt.plot(depot_x, depot_y, 's', color='red', markersize=10, label='Depot', zorder=5)
plt.plot(client_x, client_y, 'o', color='blue', markersize=5, label='Clients')

dimension = len(instance_vrplib['node_coord'])
for i in range(dimension):
    x, y = coord[i]
    plt.text(x + 1, y + 1, str(i), fontsize=8)

cmap_alns_vrptw = plt.colormaps.get_cmap('hsv')
indices_alns_vrptw = np.linspace(0, 1, len(final_solution_alns_vrptw) + 1)
custom_colors_alns_vrptw = cmap_alns_vrptw(indices_alns_vrptw)

for route_index, route in enumerate(final_solution_alns_vrptw):
    route_coord_x = [coord[node_index][0] for node_index in route]
    route_coord_y = [coord[node_index][1] for node_index in route]

    plt.plot(route_coord_x, route_coord_y,
            color=custom_colors_alns_vrptw[route_index],
            linestyle='-',
            linewidth=2,
            alpha=0.8,
            label=f'Route {route_index + 1}')

plt.title(f"Solution optimisee ALNS - {instance_vrplib['name']}\nCout: {final_cost_alns_vrptw:.2f} | Gap: {gap_alns_vrptw:.2f}%")
plt.xlabel("Coordonnee X")
plt.ylabel("Coordonnee Y")
plt.legend()
plt.axis('equal')
plt.savefig('solution_vrptw_alns.png', dpi=300, bbox_inches='tight')
print("Graphique de la solution ALNS sauvegarde: solution_vrptw_alns.png")
plt.show()

# Graphique comparatif de performance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

costs = [initial_final_cost, final_cost_alns_vrptw, optimal_cost_vrptw]
labels = ['Initial', 'ALNS', 'Optimal']
colors = ['orange', 'green', 'blue']

axes[0].bar(labels, costs, color=colors, alpha=0.7, edgecolor='black')
axes[0].set_ylabel('Cout de la solution')
axes[0].set_title('Comparaison des couts (VRPTW)')
axes[0].grid(axis='y', alpha=0.3)

for i, (cost, label) in enumerate(zip(costs, labels)):
    axes[0].text(i, cost + 5, f'{cost:.0f}', ha='center', va='bottom', fontweight='bold')

improvements = [(initial_final_cost - cost) / initial_final_cost * 100 for cost in alns_vrptw.best_cost_history]
axes[1].plot(improvements, linewidth=2, color='green')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Amelioration (%)')
axes[1].set_title('Amelioration par rapport a la solution initiale')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=((initial_final_cost - optimal_cost_vrptw) / initial_final_cost * 100),
                color='blue', linestyle='--', label='Optimum', linewidth=2)
axes[1].legend()

plt.tight_layout()
plt.savefig('performance_vrptw_alns.png', dpi=300, bbox_inches='tight')
print("Graphique de performance sauvegarde: performance_vrptw_alns.png")
plt.show()

print("\n" + "="*60)
print("OPTIMISATION ALNS TERMINEE")
print("="*60)



"""
Lecture de la solution finale VRPTW avec VRPLIB.
Affichage au terminal.
"""
# Lecture de la solution VRPLIB
path_file_solution_vrplib = 'code/tests/data/cvrplib/Vrp-Set-Solomon/C208.sol'
solution_vrplib = vrplib.read_solution(path_file_solution_vrplib)

# Afichage (terminal) de la solution VRPLIB
print("----------------------------------")
print("| Affichage de la solution VRPTW |")
print("----------------------------------")

final_solution = solution_vrplib['routes']

final_solution_with_added_depot_python = [] # pour stocker les routes avec les vrais indices Python + dépot

print(f"Solution finale actuelle (index Python) :")

for i in range (len(final_solution)) :
    print(f"Route n°{i + 1} : {final_solution[i]}")

print(f"Solution finale actuelle avec dépôt ajouté (index Python) :")

# Affichage des routes avec les indices VRPLIB
for route_index, route in enumerate(final_solution) :

    # Avec index Python
    route_corrected_python = [client_index for client_index in route] # pour l'index Python
    route_with_added_depot_python = [0] + route_corrected_python + [0] # ajout du dépôt au début et à la fin
    final_solution_with_added_depot_python.append(route_with_added_depot_python)

for i in range (len(final_solution_with_added_depot_python)) :
    print(f"Routes n°{i + 1}: {final_solution_with_added_depot_python[i]}")
print(f"Dictionnaire de la solution : {solution_vrplib.keys()}") # pour vérifier la" structure du dictionnaire

# Calcul du coût total de la solution finale
final_final_cost = 0.0 # coût total de la solution finale
index6 = 1

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
        route_cost = route_cost + vertice_cost

    final_final_cost = final_final_cost + route_cost
    
    print(f"Coût de la route n°{index6} : {route_cost:.0f}")
    index6 = index6 + 1

print(f"Coût total de la solution finale : {final_final_cost:.0f}")

# Calcul de l'écart entre le coût de la solution finale et la solution la plus optimal
true_final_cost = solution_vrplib['cost']
cost_gap = abs(((true_final_cost - final_final_cost) / final_final_cost) * 100)
print(f"Écart de coût entre la solution finale trouvé et celle qui est la plus optimisé : {cost_gap:.0f} %")

# Calcul du CO2 émis
final_CO2_g_per_km = 900 # gramme de CO2 par km
final_total_CO2_emmited_g = final_final_cost * final_CO2_g_per_km
final_total_CO2_emmited_kg = final_total_CO2_emmited_g / 1000 # conversion kg
print(f"Total de CO2 (en kg) émis pour la solution finale : {final_total_CO2_emmited_kg:.2f} kg")


temps_fin = time.time()
temps_total = temps_fin - temps_debut
print(f"\nTemps d'exécution total: {temps_total:.2f} secondes")

"""
Affichage graphique de la solution VRPTW.
"""
# Affichage graphique des coordonnées du dépôt et des clients
plt.figure(figsize = (10, 8))
plt.plot(depot_x, depot_y, 's', color = 'red', markersize = 10, label = 'Dépôt', zorder = 5)
plt.plot(client_x, client_y, 'o', color = 'blue', markersize = 5, label = 'Clients')

# Numérotation des sommets
for i in range(dimension) :
    x, y = coord[i]
    plt.text(x + 1, y + 1, str(i), fontsize = 8)

# Personalisation d'une palette de couleur
cmap = plt.colormaps.get_cmap('hsv') # choix de la colormap 'hsv'
indices = np.linspace(0, 1, len(solution_vrplib['routes']) + 1) # choix d'un nombre de couleurs aléatoire pour chaque route
custom_colors = cmap(indices)

# Personnalisation des routes
for route_index, route in enumerate(solution_vrplib['routes']):
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
plt.title(f"Solution de l'instance VRP : {instance_vrplib['name']} au coût totale {solution_vrplib['cost']}")
plt.xlabel("Coordonnée X")
plt.ylabel("Coordonnée Y")
plt.legend()
plt.axis('equal') # assure une échelle correcte
plt.show()
