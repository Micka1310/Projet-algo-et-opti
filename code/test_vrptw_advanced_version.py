"""
Lecture du fichier d'instance VRPTW avec VRPLIB.
Affichage au terminal.
"""
import vrplib
import matplotlib.pyplot as plt
import numpy as np

# Lecture du fichier d'instance Solomon
path_file_instance_vrplib = 'tests/data/cvrplib/Vrp-Set-Solomon/C201.txt'
instance_vrplib = vrplib.read_instance(path_file_instance_vrplib, instance_format = "solomon")

# Affichage (terminal) de l'instance Solomon
print("---------------------------------")
print("| Affichage de l'instance VRPTW |")
print("---------------------------------")
print(f"Nom de l'instance : {instance_vrplib['name']}")
print(f"Nombre de véhicule : {instance_vrplib['vehicles']}")
print(f"Capacité d'un camion : {instance_vrplib['capacity']}")
print(f"Coordonnées des sommets (x,y) :")

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
Algorithme utilisé : !!! à remplir (Mickaël) !!!
"""
# !!! Implémentez chacun vos algorithmes ici mes chères collègues :-) (Mickaël) !!!



"""
Lecture de la solution finale VRPTW avec VRPLIB.
Affichage au terminal.
"""
# Lecture de la solution VRPLIB
path_file_solution_vrplib = 'tests/data/cvrplib/Vrp-Set-Solomon/C201.sol'
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
