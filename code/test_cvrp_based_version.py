"""
Lecture du fichier d'instance CVRP avec VRPLIB.
Affichage au terminal.
"""
import vrplib
import matplotlib.pyplot as plt
import numpy as np
import time  # <-- ajouté pour mesurer le temps d'exécution

# Lecture du fichier d'instance VRPLIB
path_file_instance_vrplib = 'tests/data/A-n32-k5.vrp'
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
Contrainte avancé : aucun (CVRP)
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

distance = instance_vrplib['edge_weight']
clients_remaining = list(range(len(demand) - 1)) # liste d'index de clients pas encore visité (- 1 pour l'index python)
print(f"Clients restant (index Python) : {clients_remaining}")

initial_solution = [] # stock une ou plusieurs routes de la solution finale

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
    print(f"Charge finale occupé par le camion : {actual_charge}")
    print(f"Solution initiale actuelle après construction de la route n°{index4} (index Python) :")

    for i in range (len(initial_solution)) :
        print(f"Route n°{i + 1} : {initial_solution[i]}")
    index4 = index4 + 1

print(f"Routes trouvé pour la solution initiale (index VRPLIB) :")

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

dimension = instance_vrplib['dimension']

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
plt.title(f"Solution de l'instance CVRP : {instance_vrplib['name']} au coût totale {initial_final_cost:.0f}")
plt.xlabel("Coordonnée X")
plt.ylabel("Coordonnée Y")
plt.legend()
plt.axis('equal') # assure une échelle correcte
plt.show()



"""
Génération de la solution finale optimisé de la solution initiale CVRP.
Algorithme utilisé : Recuit simulé
"""
# ==============================================================
# OPTIMISATION DU CVRP PAR RECUIT SIMULÉ (avec contrainte de capacité)
# ==============================================================
# Objectif :
#    Minimiser la distance totale parcourue par les camions tout en respectant
#    la capacité maximale de chaque véhicule (ex: 100 unités de charge).
# Méthode :
#    Recuit simulé = algorithme stochastique inspiré du processus physique de refroidissement du métal.
# ==============================================================

print("\n------------------------------------------")
print("| Optimisation CVRP par Recuit Simulé    |")
print("------------------------------------------")

# === Chargement des données de l'instance ===
matrice_distance = np.array(instance_vrplib['edge_weight'], dtype=float)  # Matrice des distances entre les nœuds
capacite_camion = instance_vrplib['capacity']                 # Capacité maximale par camion
demandes_clients = np.array(demand, dtype=float)              # Demande de chaque client

print(f"Capacité d’un camion : {capacite_camion:.0f}")
print(f"Nombre de routes initiales : {len(initial_solution)}")

# === Préparation de la solution initiale ===
# Chaque tournée doit commencer et se terminer au dépôt (nœud 0)
routes_actuelles = [route[:] for route in initial_solution]
for route in routes_actuelles:
    if route[0] != 0:
        route.insert(0, 0)
    if route[-1] != 0:
        route.append(0)

meilleures_routes = [r[:] for r in routes_actuelles]  # Copie initiale de la meilleure solution

# ==============================================================
# FONCTIONS UTILITAIRES
# ==============================================================

def cout_total(routes: list) -> float:
    """Calcule la somme des distances totales de toutes les routes."""
    cout = 0.0
    for route in routes:
        for i in range(len(route) - 1):
            cout += matrice_distance[route[i], route[i + 1]]
    return float(cout)

def charge_route(route: list) -> float:
    """Calcule la charge totale (demande cumulée) d’une route, en ignorant le dépôt."""
    return float(sum(demandes_clients[n] for n in route if n != 0))

def charges_toutes_routes(routes: list) -> list:
    """Retourne la liste des charges pour chaque route."""
    return [charge_route(r) for r in routes]

def respect_capacite(routes: list) -> bool:
    """Vérifie que chaque route respecte la capacité maximale du camion."""
    for route in routes:
        if charge_route(route) > capacite_camion + 1e-9:
            return False
    return True

# ==============================================================
# Initialisation du coût et des charges
# ==============================================================

cout_actuel = cout_total(routes_actuelles)
meilleur_cout = cout_actuel
charges_actuelles = charges_toutes_routes(routes_actuelles)
meilleures_charges = charges_actuelles[:]

print("\n=== Vérification des charges initiales ===")
for i, (route, charge) in enumerate(zip(routes_actuelles, charges_actuelles), start=1):
    depassement = "  Dépasse la capacité !" if charge > capacite_camion else ""
    print(f"Camion {i:02d} | Charge = {charge:7.2f} / {capacite_camion:7.2f} | Clients = {len(route)-2}{depassement}")
print("==========================================\n")

# ==============================================================
# PARAMÈTRES DU RECUIT SIMULÉ
# ==============================================================

generateur_aleatoire = np.random.default_rng(42)
temperature = max(1.0, 0.10 * cout_actuel)  # Température initiale (10% du coût)
facteur_refroidissement = 0.9995    # Facteur de refroidissement
temperature_minimale = 1e-3         # Température minimale (critère d’arrêt)
nombre_max_iterations = 20000       # Nombre maximal d’itérations

print(f"Coût initial : {cout_actuel:.0f} | Température initiale = {temperature:.2f}")

# ==============================================================
# BOUCLE PRINCIPALE DU RECUIT SIMULÉ
# ==============================================================

iteration = 0
start_time = time.perf_counter()  # début de la mesure du temps du recuit simulé

while iteration < nombre_max_iterations and temperature > temperature_minimale:
    iteration += 1

    # Création d’une solution voisine (copie des routes actuelles)
    routes_candidats = [r[:] for r in routes_actuelles]
    charges_candidats = charges_actuelles[:]
    mouvement_effectue = False

    # Choix aléatoire du type de mouvement (0 = 2-opt, 1 = relocalisation, 2 = échange)
    type_mouvement = int(generateur_aleatoire.integers(0, 3))

    # 2-opt (inversion d’un segment de la même route) 
    if type_mouvement == 0:
        routes_possibles = [idx for idx, r in enumerate(routes_candidats) if len(r) > 4]
        if routes_possibles:
            r_idx = int(generateur_aleatoire.choice(routes_possibles))
            route = routes_candidats[r_idx]
            i = int(generateur_aleatoire.integers(1, len(route) - 2))
            j = int(generateur_aleatoire.integers(i + 1, len(route) - 1))
            route[i:j] = route[i:j][::-1]
            mouvement_effectue = True

    #  MOUVEMENT 2 : Relocalisation (déplacer un client vers une autre route) 
    if not mouvement_effectue and type_mouvement == 1:
        sources = [idx for idx, r in enumerate(routes_candidats) if len(r) > 3]
        if sources:
            s = int(generateur_aleatoire.choice(sources))
            route_source = routes_candidats[s]
            i = int(generateur_aleatoire.integers(1, len(route_source) - 1))
            client = route_source[i]
            demande_client = demandes_clients[client]

            # Retrait du client
            route_source.pop(i)
            charges_candidats[s] -= demande_client

            # Suppression de la route vide
            source_supprimee = False
            if len(route_source) <= 2:
                del routes_candidats[s]
                del charges_candidats[s]
                source_supprimee = True

            # Choix d’une route de destination
            d = int(generateur_aleatoire.integers(0, len(routes_candidats)))
            if d < len(routes_candidats):
                route_dest = routes_candidats[d]
                position_insertion = int(generateur_aleatoire.integers(1, len(route_dest)))
                if charges_candidats[d] + demande_client <= capacite_camion:
                    route_dest.insert(position_insertion, client)
                    charges_candidats[d] += demande_client
                    mouvement_effectue = True
                else:
                    # Restauration si dépassement
                    if source_supprimee:
                        routes_candidats.insert(s, [0, client, 0])
                        charges_candidats.insert(s, demande_client)
                    else:
                        route_source.insert(i, client)
                        charges_candidats[s] += demande_client
            else:
                # Création d’une nouvelle route si possible
                if demande_client <= capacite_camion:
                    routes_candidats.append([0, client, 0])
                    charges_candidats.append(demande_client)
                    mouvement_effectue = True
                else:
                    if source_supprimee:
                        routes_candidats.insert(s, [0, client, 0])
                        charges_candidats.insert(s, demande_client)
                    else:
                        route_source.insert(i, client)
                        charges_candidats[s] += demande_client

    # Échange de clients entre deux routes
    if not mouvement_effectue and type_mouvement == 2:
        routes_possibles = [idx for idx, r in enumerate(routes_candidats) if len(r) > 3]
        if len(routes_possibles) >= 2:
            r1, r2 = generateur_aleatoire.choice(routes_possibles, 2, replace=False)
            route1, route2 = routes_candidats[r1], routes_candidats[r2]
            i = int(generateur_aleatoire.integers(1, len(route1) - 1))
            j = int(generateur_aleatoire.integers(1, len(route2) - 1))
            a, b = route1[i], route2[j]
            da, db = demandes_clients[a], demandes_clients[b]
            nouvelle_charge1 = charges_candidats[r1] - da + db
            nouvelle_charge2 = charges_candidats[r2] - db + da
            if nouvelle_charge1 <= capacite_camion and nouvelle_charge2 <= capacite_camion:
                route1[i], route2[j] = b, a
                charges_candidats[r1] = nouvelle_charge1
                charges_candidats[r2] = nouvelle_charge2
                mouvement_effectue = True

    #  Si aucun mouvement valide, on continue avec refroidissement 
    if not mouvement_effectue:
        temperature = max(temperature * facteur_refroidissement, temperature_minimale)
        continue

    # Vérification de la contrainte de capacité
    if not respect_capacite(routes_candidats):
        temperature = max(temperature * facteur_refroidissement, temperature_minimale)
        continue

    # Calcul du coût de la nouvelle solution
    cout_candidat = cout_total(routes_candidats)
    variation_cout = cout_candidat - cout_actuel

    #  Critère d'acceptation du recuit simulé
    accepte = (variation_cout < 0) or (generateur_aleatoire.random() < np.exp(-variation_cout / temperature))
    if accepte:
        routes_actuelles = [r[:] for r in routes_candidats]
        cout_actuel = cout_candidat
        charges_actuelles = charges_candidats[:]

        print(f"\n[itération {iteration}] Solution acceptée | coût = {cout_actuel:.0f} | T = {temperature:.6f}")
        for k, (r, c) in enumerate(zip(routes_actuelles, charges_actuelles), start=1):
            depassement = "  Dépasse capacité !" if c > capacite_camion else ""
            print(f"   Camion {k:02d} | Charge = {c:7.2f} / {capacite_camion:7.2f} | Clients = {len(r)-2}{depassement}")
        print("-----------------------------------------------------")

        # Mise à jour de la meilleure solution
        if cout_candidat < meilleur_cout:
            meilleures_routes = [r[:] for r in routes_candidats]
            meilleur_cout = cout_candidat
            meilleures_charges = charges_candidats[:]

    # Refroidissement progressif
    temperature = max(temperature * facteur_refroidissement, temperature_minimale)

end_time = time.perf_counter()           # fin de la mesure
elapsed = end_time - start_time          # temps d'exécution du recuit simulé en secondes

# ==============================================================
# AFFICHAGE FINAL
# ==============================================================

print("\n=== Meilleure solution trouvée ===")
for k, (r, c) in enumerate(zip(meilleures_routes, meilleures_charges), start=1):
    depassement = "  Dépasse capacité !" if c > capacite_camion else ""
    print(f"Camion {k:02d} | Charge finale = {c:.2f} / {capacite_camion} | Clients = {len(r)-2} | Route = {r}{depassement}")
print("====================================================\n")

print(f"Meilleur coût rencontré : {meilleur_cout:.0f}")
print(f"Temps d'exécution du recuit simulé : {elapsed:.2f} secondes")

# Calcul des émissions de CO2
co2_g_par_km = 900.0
co2_total_kg = (meilleur_cout * co2_g_par_km) / 1000.0
print(f"Total CO2 émis : {co2_total_kg:.2f} kg")

# ==============================================================
# VISUALISATION GRAPHIQUE DES ROUTES
# ==============================================================

plt.figure(figsize=(10, 8))
plt.plot(depot_x, depot_y, 's', color='red', markersize=10, label='Dépôt', zorder=5)
plt.plot(client_x, client_y, 'o', color='blue', markersize=5, label='Clients')

# Affichage du numéro des clients
for i in range(dimension):
    x, y = coord[i]
    plt.text(x + 1, y + 1, str(i + 1), fontsize=8)

# Couleurs différentes pour chaque route
couleurs = plt.colormaps.get_cmap('hsv')
indices = np.linspace(0, 1, len(meilleures_routes) + 1)
couleurs_routes = couleurs(indices)

# Tracé de chaque route finale
for r_index, route in enumerate(meilleures_routes):
    x_points = [coord[node][0] for node in route]
    y_points = [coord[node][1] for node in route]
    plt.plot(x_points, y_points,
             color=couleurs_routes[r_index],
             linestyle='-',
             linewidth=2,
             alpha=0.8,
             label=f'Route {r_index + 1}')

plt.title(f"Recuit simulé : coût = {meilleur_cout:.0f}")
plt.xlabel("Coordonnée X")
plt.ylabel("Coordonnée Y")
plt.legend()
plt.axis('equal')
plt.show()

"""
Lecture de la solution finale CVRP avec VRPLIB.
Affichage au terminal.
"""

# Lecture de la solution VRPLIB
path_file_solution_vrplib = 'tests/data/A-n32-k5.sol'
solution_vrplib = vrplib.read_solution(path_file_solution_vrplib)

# Afichage (terminal) de la solution VRPLIB
print("---------------------------------")
print("| Affichage de la solution CVRP |")
print("---------------------------------")

final_solution = routes_actuelles

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
print(f"Écart de coût entre la solution finale trouvé et celle qui est la plus optimisé : {cost_gap:.2f} %")

# Calcul du CO2 émis
final_CO2_g_per_km = 900 # gramme de CO2 par km
final_total_CO2_emmited_g = final_final_cost * final_CO2_g_per_km
final_total_CO2_emmited_kg = final_total_CO2_emmited_g / 1000 # conversion kg
print(f"Total de CO2 (en kg) émis pour la solution finale : {final_total_CO2_emmited_kg:.2f} kg")



""""
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
plt.title(f"Solution de l'instance CVRP : {instance_vrplib['name']} au coût totale {solution_vrplib['cost']}")
plt.xlabel("Coordonnée X")
plt.ylabel("Coordonnée Y")
plt.legend()
plt.axis('equal') # assure une échelle correcte
plt.show()
