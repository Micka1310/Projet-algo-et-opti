"""
Lecture du fichier d'instance VRPTW avec VRPLIB.
Affichage au terminal.
"""
import vrplib
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import time  # <-- pour mesurer le temps d'exécution

# Lecture du fichier d'instance Solomon
path_file_instance_vrplib = 'tests/data/cvrplib/Vrp-Set-Solomon/C201.txt'
instance_vrplib = vrplib.read_instance(path_file_instance_vrplib, instance_format="solomon")

# Affichage (terminal) de l'instance Solomon
print("---------------------------------")
print("| Affichage de l'instance VRPTW |")
print("---------------------------------")
print(f"Nom de l'instance : {instance_vrplib['name']}")
print(f"Nombre de véhicule : {instance_vrplib['vehicles']}")
print(f"Capacité d'un camion : {instance_vrplib['capacity']}")
print(f"Coordonnées des sommets (x,y) :")

index3 = 0
for i in instance_vrplib['node_coord']:
    print(f"Sommet {index3} : {i}")
    index3 = index3 + 1

print("Commande des clients :")

index1 = 0
for i in instance_vrplib['demand']:
    print(f"Client au sommet {index1} : {i} objet(s)")
    index1 = index1 + 1

index7 = 0
print(f"Fenêtre temporelle : ")
for i in instance_vrplib['time_window']:
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
depot_index = 0  # le dépôt est à l'index 0
depot_x, depot_y = coord[depot_index][0], coord[depot_index][1]

# Coordonnées des clients
client_coords = np.delete(coord, depot_index, axis=0)  # suppression de la ligne de coordonnée du dépôt
client_x = client_coords[:, 0]  # toute la première colonne pris (coordonnée x)
client_y = client_coords[:, 1]  # toute la deuxième colonne pris (coordonnée y)

# On recherche le client le plus proche du dépôt
distance_coords = []  # stocke les distances qui sépare le dépôt à chaque clients

for i in range(len(client_coords)):
    abs_client_x = abs(client_x[i] - depot_x)
    abs_client_y = abs(client_y[i] - depot_y)
    sum_abs_x_y = abs_client_x + abs_client_y
    distance_coords.append(sum_abs_x_y)

print(f"Liste des distances : {distance_coords}")
print(f"Distance minimale : {min(distance_coords)}")

# Initialisation de la solution initiale
max_capacity = instance_vrplib['capacity']
demand = instance_vrplib['demand']
print(f"Charge de chaque commande : {demand}")

clients_remaining = list(range(len(demand) - 1))  # liste d'index de clients pas encore visité (- 1 pour l'index python)
print(f"Clients restant (index Python) : {clients_remaining}")

initial_solution = []  # stock une ou plusieurs routes de la solution finale

print("Début de la construction de la solution initiale...")

# Fonction qui permet de vérifier la contrainte temporelle
def check_temporal_time(temporal_route, time_windows, service_time, distance):
    """
    temporal_route : tableau d'indice (index VRPLIB) de sommet d'une route sans dépôt
    """
    full_route = [0] + temporal_route + [0]  # ajout du dépôt
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

        start_service_j = max(arrival_time_j, earliest_j)
        end_service_j = start_service_j + service_time[node_j]
        current_time = end_service_j

    return is_feasible


distance = instance_vrplib['edge_weight']
time_windows = instance_vrplib['time_window']
service_time = instance_vrplib['service_time']
index4 = 1

while clients_remaining:
    print(f"Construction de la route n°{index4}...")

    distances_remaining = [distance_coords[i] for i in clients_remaining]
    index_min_distance_remaining = np.argmin(distances_remaining)
    client_choosed = clients_remaining[index_min_distance_remaining]
    print(f"Client restant choisi en premier pour la route : {client_choosed}")

    client_choosed_vrplib = client_choosed + 1
    print(f"Client restant choisi (index VRPLIB) : {client_choosed_vrplib}")

    current_route = [client_choosed_vrplib]
    actual_charge = demand[client_choosed_vrplib]

    clients_remaining.remove(client_choosed)

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

                    time_feasible = check_temporal_time(temporal_route, time_windows, service_time, distance)

                    if time_feasible:
                        insertion_distance = distance[client_i, true_client_index] + distance[true_client_index, client_j] - distance[client_i, client_j]

                        if insertion_distance < best_insertion_cost:
                            best_insertion_cost = insertion_distance
                            min_index_cost = client_index
                            best_index = i

        if min_index_cost is not None:
            client_to_add = min_index_cost + 1
            actual_charge = actual_charge + demand[client_to_add]
            current_route.insert(best_index, client_to_add)
            clients_remaining.remove(min_index_cost)
        else:
            continue_extension = False
            print("fin de la construction de la route actuelle")

    initial_solution.append([0] + current_route + [0])
    print(f"Charge finale occupé par le camion : {actual_charge}")
    print(f"Solution initiale actuelle après construction de la route n°{index4} (index Python) :")

    for i in range(len(initial_solution)):
        print(f"Route n°{i + 1} : {initial_solution[i]}")
    index4 = index4 + 1

print(f"Routes trouvé pour la solution initiale (index VRPLIB) :")
for route_index, route in enumerate(initial_solution):
    route_corrected = [client_index for client_index in route]
    print(f"Route n°{route_index + 1}: {route_corrected}")

# Calcul du coût total de la solution initiale
initial_final_cost = 0.0
index5 = 1

print(f"Calcul du coût total de la solution initiale...")

for route in initial_solution:
    route_cost = 0.0
    for i in range(len(route) - 1):
        start_node_index = route[i]
        end_node_index = route[i + 1]
        vertice_cost = distance[start_node_index, end_node_index]
        route_cost = route_cost + vertice_cost

    initial_final_cost = initial_final_cost + route_cost

    print(f"Coût de la route n°{index5} : {route_cost:.0f}")
    index5 = index5 + 1

print(f"Coût total de la solution initiale : {initial_final_cost:.0f}")

# Calcul du CO2 émis
initial_CO2_g_per_km = 900
initial_total_CO2_emmited_g = initial_final_cost * initial_CO2_g_per_km
initial_total_CO2_emmited_kg = initial_total_CO2_emmited_g / 1000
print(f"Total de CO2 (en kg) émis pour la solution initiale : {initial_total_CO2_emmited_kg:.2f} kg")


"""
Affichage graphique de la solution initiale VRPTW.
"""
plt.figure(figsize=(10, 8))
plt.plot(depot_x, depot_y, 's', color='red', markersize=10, label='Dépôt', zorder=5)
plt.plot(client_x, client_y, 'o', color='blue', markersize=5, label='Clients')

dimension = len(instance_vrplib['node_coord'])

for i in range(dimension):
    x, y = coord[i]
    plt.text(x + 1, y + 1, str(i), fontsize=8)

cmap = plt.colormaps.get_cmap('hsv')
indices = np.linspace(0, 1, len(initial_solution) + 1)
custom_colors = cmap(indices)

for route_index, route in enumerate(initial_solution):
    route_coord_x = [coord[node_index][0] for node_index in route]
    route_coord_y = [coord[node_index][1] for node_index in route]

    plt.plot(route_coord_x, route_coord_y,
             color=custom_colors[route_index],
             linestyle='-',
             linewidth=2,
             alpha=0.8,
             label=f'Route {route_index + 1}')

plt.title(f"Solution de l'instance VRPTW : {instance_vrplib['name']} au coût totale {initial_final_cost:.0f}")
plt.xlabel("Coordonnée X")
plt.ylabel("Coordonnée Y")
plt.legend()
plt.axis('equal')
plt.show()


"""
Génération de la solution finale optimisée de la solution initiale VRPTW.
Algorithme utilisé : Recuit simulé
"""
# =======================
# RECUIT SIMULE VRPTW 
# =======================

# ------------------------
# Outils de vérification
# ------------------------

def charge_route(route, demandes) -> float:
    return float(sum(demandes[i] for i in route if i != 0))

def cout_route(route, matrice_dist) -> float:
    c = 0.0
    for k in range(len(route) - 1):
        c += matrice_dist[route[k], route[k + 1]]
    return float(c)

def route_faisable_temporal(route, fenetres, temps_service, matrice_dist) -> bool:
    t = 0.0
    for k in range(len(route) - 1):
        i, j = route[k], route[k + 1]
        t = t + matrice_dist[i, j]
        earliest, latest = fenetres[j, 0], fenetres[j, 1]
        if t > latest:
            return False
        if t < earliest:
            t = earliest
        t = t + temps_service[j]
    return True

def solution_faisable(routes, demandes, capacite, fenetres, temps_service, matrice_dist, nb_vehicules_max) -> bool:
    if len(routes) > nb_vehicules_max:
        return False
    for r in routes:
        if charge_route(r, demandes) > capacite:
            return False
        if not route_faisable_temporal(r, fenetres, temps_service, matrice_dist):
            return False
    return True

def cout_solution(routes, matrice_dist) -> float:
    return float(sum(cout_route(r, matrice_dist) for r in routes))


# ------------------------
# Construction état initial
# ------------------------
rs_courantes = [r[:] for r in initial_solution]
for r in rs_courantes:
    if r[0] != 0:
        r.insert(0, 0)
    if r[-1] != 0:
        r.append(0)

cout_courant = cout_solution(rs_courantes, distance)
rs_meilleures = [r[:] for r in rs_courantes]
cout_meilleur = cout_courant

print("\n------------------------------------------")
print("| Recuit simulé VRPTW (capacité + TW)    |")
print("------------------------------------------")
print(f"Coût initial (glouton) : {cout_courant:.0f}")
print(f"Nb véhicules initial   : {len(rs_courantes)} / {instance_vrplib['vehicles']}")

# ------------------------
# Paramètres SA
# ------------------------
rng = random.Random(42)
T = max(1.0, 0.10 * cout_courant)
alpha = 0.999
T_min = 1e-3
max_iter = 100000

# ------------------------
# Fonctions de mouvements
# ------------------------

def mv_2opt_intra(routes):
    cand = [r[:] for r in routes]
    indices = [ri for ri, r in enumerate(cand) if len(r) > 4]
    if not indices:
        return routes, False, ()
    ri = rng.choice(indices)
    r = cand[ri]
    i = rng.randrange(1, len(r) - 2)
    j = rng.randrange(i + 1, len(r) - 1)
    r[i:j] = reversed(r[i:j])
    return cand, True, (ri,)

def mv_relocation(routes):
    cand = [r[:] for r in routes]
    elig = [ri for ri, r in enumerate(cand) if len(r) > 3]
    if not elig:
        return routes, False, ()
    s = rng.choice(elig)
    rs = cand[s]
    i = rng.randrange(1, len(rs) - 1)
    v = rs.pop(i)

    route_supprimee = False
    if len(rs) <= 2:
        del cand[s]
        route_supprimee = True

    if not cand:
        cand.append([0, v, 0])
        return cand, True, (0,)

    d = rng.randrange(0, len(cand))
    rd = cand[d]

    pos = rng.randrange(1, len(rd))
    rd.insert(pos, v)

    return cand, True, (d, s) if not route_supprimee else (d,)

def mv_swap(routes):
    cand = [r[:] for r in routes]
    elig = [ri for ri, r in enumerate(cand) if len(r) > 3]
    if len(elig) < 2:
        return routes, False, ()
    r1i, r2i = rng.sample(elig, 2)
    R1, R2 = cand[r1i], cand[r2i]
    i = rng.randrange(1, len(R1) - 1)
    j = rng.randrange(1, len(R2) - 1)
    R1[i], R2[j] = R2[j], R1[i]
    return cand, True, (r1i, r2i)


# ------------------------
# Boucle principale SA
# ------------------------
it = 0
start_time = time.perf_counter()  # début mesure temps

while it < max_iter and T > T_min:
    it += 1

    move_type = rng.randrange(3)
    if move_type == 0:
        cand_routes, ok, touched = mv_2opt_intra(rs_courantes)
    elif move_type == 1:
        cand_routes, ok, touched = mv_relocation(rs_courantes)
    else:
        cand_routes, ok, touched = mv_swap(rs_courantes)

    if not ok:
        T *= alpha
        continue

    if len(cand_routes) > instance_vrplib['vehicles']:
        T *= alpha
        continue

    faisable_local = True
    for ri in range(len(cand_routes)):
        r = cand_routes[ri]
        if charge_route(r, demand) > max_capacity:
            faisable_local = False
            break
        if not route_faisable_temporal(r, time_windows, service_time, distance):
            faisable_local = False
            break
    if not faisable_local:
        T *= alpha
        continue

    cout_cand = cout_solution(cand_routes, distance)
    delta = cout_cand - cout_courant

    if delta < 0 or rng.random() < math.exp(-delta / T):
        rs_courantes = [r[:] for r in cand_routes]
        cout_courant = cout_cand

        if cout_courant < cout_meilleur:
            rs_meilleures = [r[:] for r in rs_courantes]
            cout_meilleur = cout_courant

    T *= alpha

end_time = time.perf_counter()  # fin mesure temps
elapsed = end_time - start_time

# ------------------------
# Résultats recuit
# ------------------------
print("\nRésultats du recuit simulé VRPTW")
print(f"Temps d'exécution du recuit simulé : {elapsed:.2f} secondes")
print(f"Coût meilleur = {cout_meilleur:.0f} | nb véhicules = {len(rs_meilleures)} / {instance_vrplib['vehicles']}")
for k, r in enumerate(rs_meilleures, start=1):
    ch = charge_route(r, demand)
    print(f"Route {k:02d} | charge = {ch:.0f} | longueur = {cout_route(r, distance):.0f} | {r}")

co2_g_km = 900.0
co2_total_kg_recuit = (cout_meilleur * co2_g_km) / 1000.0
print(f"CO2 total (kg) pour la solution recuit : {co2_total_kg_recuit:.2f} kg")

# Affichage graphique de la solution recuit
plt.figure(figsize=(10, 8))
plt.plot(coord[0][0], coord[0][1], 's', color='red', markersize=10, label='Dépôt', zorder=5)
plt.scatter([x for (x, y) in coord[1:]], [y for (x, y) in coord[1:]], s=25, c='blue', label='Clients')
for i, (x, y) in enumerate(coord):
    plt.text(x + 1, y + 1, str(i), fontsize=8)

cmap = plt.colormaps.get_cmap('hsv')
indices = np.linspace(0, 1, len(rs_meilleures) + 1)
couleurs = cmap(indices)

for r_idx, route in enumerate(rs_meilleures):
    xs = [coord[n][0] for n in route]
    ys = [coord[n][1] for n in route]
    plt.plot(xs, ys, color=couleurs[r_idx], linewidth=2, alpha=0.85, label=f"RS Route {r_idx + 1}")

plt.title(f"Recuit simulé VRPTW : coût {cout_meilleur:.0f}")
plt.xlabel("Coordonnée X")
plt.ylabel("Coordonnée Y")
plt.legend()
plt.axis('equal')
plt.show()


"""
Lecture de la solution finale VRPTW avec VRPLIB.
Affichage au terminal.
"""
path_file_solution_vrplib = 'tests/data/cvrplib/Vrp-Set-Solomon/C201.sol'
solution_vrplib = vrplib.read_solution(path_file_solution_vrplib)

print("----------------------------------")
print("| Affichage de la solution VRPTW |")
print("----------------------------------")

final_solution = solution_vrplib['routes']
final_solution_with_added_depot_python = []

print(f"Solution finale actuelle (index Python) :")
for i in range(len(final_solution)):
    print(f"Route n°{i + 1} : {final_solution[i]}")

print(f"Solution finale actuelle avec dépôt ajouté (index Python) :")
for route_index, route in enumerate(final_solution):
    route_corrected_python = [client_index for client_index in route]
    route_with_added_depot_python = [0] + route_corrected_python + [0]
    final_solution_with_added_depot_python.append(route_with_added_depot_python)

for i in range(len(final_solution_with_added_depot_python)):
    print(f"Routes n°{i + 1}: {final_solution_with_added_depot_python[i]}")
print(f"Dictionnaire de la solution : {solution_vrplib.keys()}")

final_final_cost = 0.0
index6 = 1

print(f"Calcul du coût total de la solution finale...")

for route in final_solution_with_added_depot_python:
    route_cost = 0.0
    for i in range(len(route) - 1):
        start_node_index = route[i]
        end_node_index = route[i + 1]
        vertice_cost = distance[start_node_index, end_node_index]
        route_cost = route_cost + vertice_cost

    final_final_cost = final_final_cost + route_cost

    print(f"Coût de la route n°{index6} : {route_cost:.0f}")
    index6 = index6 + 1

print(f"Coût total de la solution finale : {final_final_cost:.0f}")

true_final_cost = solution_vrplib['cost']
cost_gap = ((cout_meilleur - final_final_cost) / final_final_cost) * 100
print(f"Écart de coût entre la solution finale trouvé et celle qui est la plus optimisé : {cost_gap:.2f} %")

final_CO2_g_per_km = 900
final_total_CO2_emmited_g = final_final_cost * final_CO2_g_per_km
final_total_CO2_emmited_kg = final_total_CO2_emmited_g / 1000
print(f"Total de CO2 (en kg) émis pour la solution finale : {final_total_CO2_emmited_kg:.2f} kg")


"""
Affichage graphique de la solution VRPTW (solution VRPLIB).
"""
plt.figure(figsize=(10, 8))
plt.plot(depot_x, depot_y, 's', color='red', markersize=10, label='Dépôt', zorder=5)
plt.plot(client_x, client_y, 'o', color='blue', markersize=5, label='Clients')

for i in range(dimension):
    x, y = coord[i]
    plt.text(x + 1, y + 1, str(i), fontsize=8)

cmap = plt.colormaps.get_cmap('hsv')
indices = np.linspace(0, 1, len(solution_vrplib['routes']) + 1)
custom_colors = cmap(indices)

for route_index, route in enumerate(solution_vrplib['routes']):
    full_route = [0] + route + [0]
    route_coord_x = [coord[node_index][0] for node_index in full_route]
    route_coord_y = [coord[node_index][1] for node_index in full_route]

    plt.plot(route_coord_x, route_coord_y,
             color=custom_colors[route_index],
             linestyle='-',
             linewidth=2,
             alpha=0.8,
             label=f'Route {route_index + 1}')

plt.title(f"Solution de l'instance VRP : {instance_vrplib['name']} au coût totale {solution_vrplib['cost']}")
plt.xlabel("Coordonnée X")
plt.ylabel("Coordonnée Y")
plt.legend()
plt.axis('equal')
plt.show()
