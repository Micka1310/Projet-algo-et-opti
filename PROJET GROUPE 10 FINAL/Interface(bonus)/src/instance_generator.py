"""
Générateur d'instances VRP personnalisées.
Permet de créer des instances aléatoires avec un nombre de clients défini.
"""

import numpy as np
import random


class VRPInstanceGenerator:
    """Générateur d'instances VRP avec paramètres personnalisables."""

    def __init__(self, n_clients, depot_location=(50, 50), grid_size=100, seed=None):
        """
        Initialise le générateur d'instances.

        Args:
            n_clients: Nombre de clients à générer
            depot_location: Coordonnées (x, y) du dépôt
            grid_size: Taille de la grille (coordonnées de 0 à grid_size)
            seed: Graine aléatoire pour la reproductibilité
        """
        self.n_clients = n_clients
        self.depot_location = depot_location
        self.grid_size = grid_size

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self, capacity_range=(50, 150), demand_range=(5, 30),
                         n_vehicles=None, with_time_windows=False):
        """
        Génère une instance VRP complète.

        Args:
            capacity_range: Tuple (min, max) pour la capacité des véhicules
            demand_range: Tuple (min, max) pour les demandes des clients
            n_vehicles: Nombre de véhicules (calculé auto si None)
            with_time_windows: Ajoute des fenêtres temporelles si True

        Returns:
            dict: Instance VRP au format compatible avec vrplib
        """
        # Génération des coordonnées
        node_coords = [self.depot_location]
        for _ in range(self.n_clients):
            x = random.randint(0, self.grid_size)
            y = random.randint(0, self.grid_size)
            node_coords.append((x, y))

        # Génération des demandes
        demands = [0]  # Dépôt a demande 0
        for _ in range(self.n_clients):
            demand = random.randint(demand_range[0], demand_range[1])
            demands.append(demand)

        # Capacité des véhicules
        capacity = random.randint(capacity_range[0], capacity_range[1])

        # Estimation du nombre de véhicules nécessaires
        if n_vehicles is None:
            total_demand = sum(demands)
            n_vehicles = max(3, int(np.ceil(total_demand / capacity * 1.2)))

        # Calcul de la matrice de distances (Euclidienne)
        n_nodes = len(node_coords)
        edge_weight = np.zeros((n_nodes, n_nodes))

        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    x1, y1 = node_coords[i]
                    x2, y2 = node_coords[j]
                    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    edge_weight[i][j] = int(dist + 0.5)  # Arrondi

        # Construction de l'instance
        instance = {
            'name': f'Generated-n{self.n_clients}-k{n_vehicles}',
            'comment': f'Instance générée automatiquement avec {self.n_clients} clients',
            'type': 'CVRP',
            'dimension': n_nodes,
            'edge_weight_type': 'EXPLICIT',
            'edge_weight_format': 'FULL_MATRIX',
            'capacity': capacity,
            'node_coord': np.array(node_coords),
            'demand': np.array(demands),
            'depot': 1,  # Format VRPLIB
            'edge_weight': edge_weight,
            'n_vehicles': n_vehicles
        }

        # Ajout optionnel des fenêtres temporelles
        if with_time_windows:
            instance = self._add_time_windows(instance)

        return instance

    def _add_time_windows(self, instance):
        """
        Ajoute des fenêtres temporelles à l'instance.

        Args:
            instance: Instance VRP de base

        Returns:
            dict: Instance enrichie avec fenêtres temporelles
        """
        n_nodes = instance['dimension']

        # Fenêtre du dépôt (journée complète)
        time_windows = [(0, 480)]  # 8h de travail (en minutes)

        # Fenêtres pour les clients
        for i in range(1, n_nodes):
            # Génération de fenêtres temporelles réalistes
            earliest = random.randint(0, 300)
            latest = earliest + random.randint(60, 180)
            time_windows.append((earliest, min(latest, 480)))

        # Service time (temps de service chez chaque client)
        service_times = [0] + [random.randint(10, 30) for _ in range(n_nodes - 1)]

        instance['time_window'] = np.array(time_windows)
        instance['service_time'] = np.array(service_times)
        instance['type'] = 'VRPTW'

        return instance

    def generate_clustered_instance(self, n_clusters=3, **kwargs):
        """
        Génère une instance avec des clients regroupés en clusters.

        Args:
            n_clusters: Nombre de clusters à créer
            **kwargs: Arguments passés à generate_instance

        Returns:
            dict: Instance VRP avec structure en clusters
        """
        # Centres des clusters
        cluster_centers = []
        margin = self.grid_size // 4

        for _ in range(n_clusters):
            cx = random.randint(margin, self.grid_size - margin)
            cy = random.randint(margin, self.grid_size - margin)
            cluster_centers.append((cx, cy))

        # Génération des clients autour des centres
        node_coords = [self.depot_location]
        clients_per_cluster = self.n_clients // n_clusters

        for cluster_idx, (cx, cy) in enumerate(cluster_centers):
            n_clients_cluster = clients_per_cluster
            if cluster_idx == n_clusters - 1:
                # Dernier cluster prend les clients restants
                n_clients_cluster = self.n_clients - (clients_per_cluster * (n_clusters - 1))

            for _ in range(n_clients_cluster):
                # Dispersion autour du centre
                radius = self.grid_size // 6
                angle = random.uniform(0, 2 * np.pi)
                r = random.uniform(0, radius)

                x = int(cx + r * np.cos(angle))
                y = int(cy + r * np.sin(angle))

                # Contrainte dans la grille
                x = max(0, min(self.grid_size, x))
                y = max(0, min(self.grid_size, y))

                node_coords.append((x, y))

        # Génération du reste de l'instance
        instance = self.generate_instance(**kwargs)
        instance['node_coord'] = np.array(node_coords)
        instance['comment'] += f' (clustered with {n_clusters} clusters)'

        return instance


def save_instance_vrplib_format(instance, filename):
    """
    Sauvegarde une instance au format VRPLIB.

    Args:
        instance: Dictionnaire contenant l'instance
        filename: Chemin du fichier de sortie
    """
    with open(filename, 'w') as f:
        f.write(f"NAME : {instance['name']}\n")
        f.write(f"COMMENT : {instance['comment']}\n")
        f.write(f"TYPE : {instance['type']}\n")
        f.write(f"DIMENSION : {instance['dimension']}\n")
        f.write(f"EDGE_WEIGHT_TYPE : {instance['edge_weight_type']}\n")
        f.write(f"CAPACITY : {instance['capacity']}\n")

        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(instance['node_coord'], 1):
            f.write(f"{i} {x} {y}\n")

        f.write("DEMAND_SECTION\n")
        for i, demand in enumerate(instance['demand'], 1):
            f.write(f"{i} {demand}\n")

        f.write("DEPOT_SECTION\n")
        f.write(f"{instance['depot']}\n")
        f.write("-1\n")
        f.write("EOF\n")
