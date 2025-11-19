"""
Parser pour les instances VRPTW (Vehicle Routing Problem with Time Windows)

Ce module permet de lire les instances au format Solomon (.txt et .vrptw)
incluant les fenêtres temporelles et temps de service.

Formats supportés:
- Format Solomon .txt (colonnes READY TIME, DUE DATE, SERVICE TIME)
- Format VRPTW standard (TIME_WINDOW_SECTION)
"""

import numpy as np
import re


def parse_solomon_txt(file_path):
    """
    Parse un fichier Solomon au format .txt.

    Format attendu:
    C101

    VEHICLE
    NUMBER     CAPACITY
      25         200

    CUSTOMER
    CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME
        0      40         50          0          0       1236          0
        1      45         68         10        912        967         90
        ...

    Args:
        file_path (str): Chemin vers le fichier .txt

    Returns:
        dict: Instance au format standard
            {
                'name': str,
                'type': 'VRPTW',
                'dimension': int,
                'capacity': int,
                'vehicles': int,
                'node_coord': np.array,
                'demand': np.array,
                'time_window': np.array (n x 2) - [ready_time, due_date],
                'service_time': np.array,
                'edge_weight': np.array,
                'depot': int (=1)
            }
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Extraction du nom
    instance_name = lines[0].strip()

    # Trouver la section VEHICLE
    vehicle_idx = None
    for i, line in enumerate(lines):
        if 'VEHICLE' in line:
            vehicle_idx = i
            break

    # Extraction capacité et nombre de véhicules
    capacity_line = lines[vehicle_idx + 2].strip().split()
    vehicles = int(capacity_line[0])
    capacity = int(capacity_line[1])

    # Trouver la section CUSTOMER
    customer_idx = None
    for i, line in enumerate(lines):
        if 'CUSTOMER' in line:
            customer_idx = i
            break

    # Lecture des clients (commencer après la ligne d'en-tête)
    data_start = customer_idx + 3

    customers = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) >= 7:
            cust_no = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            demand = int(parts[3])
            ready_time = int(parts[4])
            due_date = int(parts[5])
            service_time = int(parts[6])

            customers.append({
                'id': cust_no,
                'x': x,
                'y': y,
                'demand': demand,
                'ready_time': ready_time,
                'due_date': due_date,
                'service_time': service_time
            })

    # Construction de l'instance
    n = len(customers)

    node_coord = np.array([[c['x'], c['y']] for c in customers])
    demand = np.array([c['demand'] for c in customers])
    time_window = np.array([[c['ready_time'], c['due_date']] for c in customers])
    service_time = np.array([c['service_time'] for c in customers])

    # Calcul de la matrice de distances (distance euclidienne)
    edge_weight = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = node_coord[i][0] - node_coord[j][0]
                dy = node_coord[i][1] - node_coord[j][1]
                dist = np.sqrt(dx**2 + dy**2)
                edge_weight[i][j] = int(dist + 0.5)  # Arrondi

    instance = {
        'name': instance_name,
        'type': 'VRPTW',
        'dimension': n,
        'capacity': capacity,
        'vehicles': vehicles,
        'node_coord': node_coord,
        'demand': demand,
        'time_window': time_window,
        'service_time': service_time,
        'edge_weight': edge_weight,
        'depot': 1,  # Format VRPLIB (1-indexed)
        'comment': f'Solomon instance {instance_name} - VRPTW with {n} customers'
    }

    return instance


def parse_vrptw_format(file_path):
    """
    Parse un fichier au format VRPTW standard.

    Format attendu:
    NAME : C101
    TYPE : CVRPTW
    DIMENSION : 26
    VEHICLES : 3
    CAPACITY : 200
    SERVICE_TIME : 90
    EDGE_WEIGHT_TYPE : FLOOR_2D
    NODE_COORD_SECTION
    1 40 50
    2 45 68
    ...
    DEMAND_SECTION
    1 0
    2 10
    ...
    TIME_WINDOW_SECTION
    1 0 1236
    2 912 967
    ...
    DEPOT_SECTION
    1
    -1
    EOF

    Args:
        file_path (str): Chemin vers le fichier .vrptw

    Returns:
        dict: Instance au format standard (même structure que parse_solomon_txt)
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # Extraction des métadonnées
    name_match = re.search(r'NAME\s*:\s*(\S+)', content)
    instance_name = name_match.group(1) if name_match else 'Unknown'

    dimension_match = re.search(r'DIMENSION\s*:\s*(\d+)', content)
    dimension = int(dimension_match.group(1)) if dimension_match else 0

    vehicles_match = re.search(r'VEHICLES\s*:\s*(\d+)', content)
    vehicles = int(vehicles_match.group(1)) if vehicles_match else 10

    capacity_match = re.search(r'CAPACITY\s*:\s*(\d+)', content)
    capacity = int(capacity_match.group(1)) if capacity_match else 100

    service_time_match = re.search(r'SERVICE_TIME\s*:\s*(\d+)', content)
    default_service_time = int(service_time_match.group(1)) if service_time_match else 0

    # Extraction des coordonnées
    coord_section = re.search(r'NODE_COORD_SECTION\s+(.*?)(?=\n\w+_SECTION|\nEOF)',
                             content, re.DOTALL)
    coordinates = []
    if coord_section:
        for line in coord_section.group(1).strip().split('\n'):
            parts = line.split()
            if len(parts) >= 3:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coordinates.append((node_id, x, y))

    # Extraction des demandes
    demand_section = re.search(r'DEMAND_SECTION\s+(.*?)(?=\n\w+_SECTION|\nEOF)',
                              content, re.DOTALL)
    demands = {}
    if demand_section:
        for line in demand_section.group(1).strip().split('\n'):
            parts = line.split()
            if len(parts) >= 2:
                node_id = int(parts[0])
                demand = int(parts[1])
                demands[node_id] = demand

    # Extraction des fenêtres temporelles
    tw_section = re.search(r'TIME_WINDOW_SECTION\s+(.*?)(?=\n\w+_SECTION|\nEOF)',
                          content, re.DOTALL)
    time_windows = {}
    if tw_section:
        for line in tw_section.group(1).strip().split('\n'):
            parts = line.split()
            if len(parts) >= 3:
                node_id = int(parts[0])
                ready_time = int(parts[1])
                due_date = int(parts[2])
                time_windows[node_id] = (ready_time, due_date)

    # Construction des arrays NumPy (en utilisant l'ordre des coordonnées)
    n = len(coordinates)
    node_coord = np.zeros((n, 2))
    demand_array = np.zeros(n, dtype=int)
    time_window_array = np.zeros((n, 2), dtype=int)
    service_time_array = np.full(n, default_service_time, dtype=int)

    for idx, (node_id, x, y) in enumerate(coordinates):
        node_coord[idx] = [x, y]
        demand_array[idx] = demands.get(node_id, 0)
        time_window_array[idx] = time_windows.get(node_id, (0, 9999))

    # Calcul de la matrice de distances
    edge_weight = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = node_coord[i][0] - node_coord[j][0]
                dy = node_coord[i][1] - node_coord[j][1]
                dist = np.floor(np.sqrt(dx**2 + dy**2))  # FLOOR_2D
                edge_weight[i][j] = dist

    instance = {
        'name': instance_name,
        'type': 'VRPTW',
        'dimension': n,
        'capacity': capacity,
        'vehicles': vehicles,
        'node_coord': node_coord,
        'demand': demand_array,
        'time_window': time_window_array,
        'service_time': service_time_array,
        'edge_weight': edge_weight,
        'depot': 1,
        'comment': f'VRPTW instance {instance_name} with {n} customers'
    }

    return instance


def load_vrptw_instance(file_path):
    """
    Charge une instance VRPTW depuis un fichier .txt ou .vrptw.

    Détecte automatiquement le format et appelle le parser approprié.

    Args:
        file_path (str): Chemin vers le fichier

    Returns:
        dict: Instance au format standard

    Raises:
        ValueError: Si le format n'est pas reconnu
    """
    if file_path.endswith('.txt'):
        return parse_solomon_txt(file_path)
    elif file_path.endswith('.vrptw'):
        return parse_vrptw_format(file_path)
    else:
        raise ValueError(f"Format de fichier non supporté: {file_path}. "
                        "Utilisez .txt (Solomon) ou .vrptw")


def verify_time_windows(instance):
    """
    Vérifie la cohérence des fenêtres temporelles d'une instance.

    Args:
        instance (dict): Instance VRPTW

    Returns:
        tuple: (is_valid: bool, messages: list)
    """
    messages = []
    is_valid = True

    time_windows = instance['time_window']
    n = len(time_windows)

    for i in range(n):
        ready, due = time_windows[i]

        # Vérifier ready <= due
        if ready > due:
            messages.append(f"Client {i}: ready_time ({ready}) > due_date ({due})")
            is_valid = False

        # Vérifier fenêtre non-vide
        if ready == due and instance['service_time'][i] > 0:
            messages.append(f"Client {i}: fenêtre vide mais service_time > 0")
            is_valid = False

    # Vérifier le dépôt (index 0)
    depot_ready, depot_due = time_windows[0]
    if depot_ready != 0:
        messages.append(f"Dépôt: ready_time devrait être 0 (actuellement {depot_ready})")
        is_valid = False

    if is_valid:
        messages.append("[OK] Toutes les fenêtres temporelles sont cohérentes")

    return is_valid, messages


# Exemple d'utilisation
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]

        try:
            instance = load_vrptw_instance(file_path)

            print(f"Instance: {instance['name']}")
            print(f"Type: {instance['type']}")
            print(f"Dimension: {instance['dimension']}")
            print(f"Capacité: {instance['capacity']}")
            print(f"Véhicules: {instance['vehicles']}")

            print(f"\nPremiers clients:")
            for i in range(min(5, instance['dimension'])):
                coord = instance['node_coord'][i]
                demand = instance['demand'][i]
                tw = instance['time_window'][i]
                service = instance['service_time'][i]
                print(f"  Client {i}: coord=({coord[0]:.0f}, {coord[1]:.0f}), "
                      f"demand={demand}, tw=[{tw[0]}, {tw[1]}], service={service}")

            # Vérification
            is_valid, messages = verify_time_windows(instance)
            print(f"\nVérification des fenêtres temporelles:")
            for msg in messages:
                print(f"  {msg}")

        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Usage: python vrptw_parser.py <fichier.txt|fichier.vrptw>")
        print("\nExemple:")
        print("  python vrptw_parser.py 'instances avancées/C101.txt'")
        print("  python vrptw_parser.py 'instances avancées/C101.25.3.vrptw'")
