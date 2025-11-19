"""
Génération de solutions initiales pour le VRP.
Implémente l'heuristique de Clarke & Wright et l'insertion séquentielle.
"""

import numpy as np
from copy import deepcopy


class InitialSolutionGenerator:
    """
    Générateur de solutions initiales pour le VRP.
    """

    def __init__(self, instance):
        """
        Initialise le générateur.

        Args:
            instance: Instance VRP (dict)
        """
        self.instance = instance
        self.dimension = instance['dimension']
        self.capacity = instance['capacity']
        self.demands = instance['demand']
        self.distances = instance['edge_weight']
        self.depot = 0  # Index Python du dépôt

    def clarke_wright(self):
        """
        Heuristique de Clarke & Wright (économies).

        Returns:
            list: Solution initiale (liste de routes)
        """
        print("🏗️ Génération solution initiale : Clarke & Wright")

        # Étape 1 : Initialisation - chaque client a sa propre route
        routes = []
        for i in range(1, self.dimension):
            routes.append([self.depot, i, self.depot])

        # Étape 2 : Calcul des économies s_ij = d(0,i) + d(0,j) - d(i,j)
        savings = []
        for i in range(1, self.dimension):
            for j in range(i + 1, self.dimension):
                saving = (self.distances[self.depot][i] +
                         self.distances[self.depot][j] -
                         self.distances[i][j])
                savings.append((saving, i, j))

        # Tri décroissant des économies
        savings.sort(reverse=True, key=lambda x: x[0])

        # Étape 3 : Fusion des routes selon les économies
        for saving_value, i, j in savings:
            # Recherche des routes contenant i et j
            route_i = None
            route_j = None
            route_i_idx = None
            route_j_idx = None

            for idx, route in enumerate(routes):
                if i in route:
                    route_i = route
                    route_i_idx = idx
                if j in route:
                    route_j = route
                    route_j_idx = idx

            if route_i is None or route_j is None or route_i_idx == route_j_idx:
                continue

            # Vérification : i et j doivent être en début ou fin de route
            # (pour pouvoir fusionner sans créer de sous-tour)
            i_is_extreme = (route_i[1] == i or route_i[-2] == i)
            j_is_extreme = (route_j[1] == j or route_j[-2] == j)

            if not (i_is_extreme and j_is_extreme):
                continue

            # Vérification de la capacité
            demand_i = sum(self.demands[c] for c in route_i if c != self.depot)
            demand_j = sum(self.demands[c] for c in route_j if c != self.depot)

            if demand_i + demand_j > self.capacity:
                continue

            # Fusion des routes
            # Retirer les dépôts internes et fusionner
            new_route = self._merge_routes(route_i, route_j, i, j)

            if new_route:
                # Remplacement dans la liste
                routes[route_i_idx] = new_route
                routes.pop(route_j_idx)

        print(f"   ✅ Solution générée : {len(routes)} routes")
        return routes

    def _merge_routes(self, route1, route2, i, j):
        """
        Fusionne deux routes en connectant les clients i et j.

        Args:
            route1, route2: Routes à fusionner
            i, j: Clients à connecter

        Returns:
            list: Route fusionnée ou None si impossible
        """
        # Copie pour manipulation
        r1 = route1[1:-1]  # Sans dépôt
        r2 = route2[1:-1]

        # Déterminer l'orientation
        if r1[-1] == i and r2[0] == j:
            merged = r1 + r2
        elif r1[-1] == i and r2[-1] == j:
            merged = r1 + r2[::-1]
        elif r1[0] == i and r2[0] == j:
            merged = r1[::-1] + r2
        elif r1[0] == i and r2[-1] == j:
            merged = r2 + r1
        else:
            return None

        return [self.depot] + merged + [self.depot]

    def nearest_neighbor(self):
        """
        Heuristique du plus proche voisin (greedy).

        Returns:
            list: Solution initiale
        """
        print("🏗️ Génération solution initiale : Plus Proche Voisin")

        routes = []
        unvisited = list(range(1, self.dimension))

        while unvisited:
            route = [self.depot]
            current_capacity = 0
            current_node = self.depot

            while unvisited:
                # Trouver le client le plus proche non visité et faisable
                best_distance = float('inf')
                best_customer = None

                for customer in unvisited:
                    if current_capacity + self.demands[customer] <= self.capacity:
                        dist = self.distances[current_node][customer]
                        if dist < best_distance:
                            best_distance = dist
                            best_customer = customer

                if best_customer is None:
                    break

                # Ajout du client à la route
                route.append(best_customer)
                current_capacity += self.demands[best_customer]
                current_node = best_customer
                unvisited.remove(best_customer)

            # Retour au dépôt
            route.append(self.depot)
            routes.append(route)

        print(f"   ✅ Solution générée : {len(routes)} routes")
        return routes

    def sequential_insertion(self):
        """
        Insertion séquentielle gloutonne (version du fichier de test).

        Returns:
            list: Solution initiale
        """
        print("🏗️ Génération solution initiale : Insertion Séquentielle")

        # Calcul des distances de chaque client au dépôt
        distances_to_depot = []
        for i in range(1, self.dimension):
            distances_to_depot.append((self.distances[self.depot][i], i))

        distances_to_depot.sort()

        clients_remaining = [c for _, c in distances_to_depot]
        solution = []

        while clients_remaining:
            # Démarrer une nouvelle route avec le client le plus proche
            first_customer = clients_remaining[0]
            current_route = [self.depot, first_customer, self.depot]
            current_capacity = self.demands[first_customer]
            clients_remaining.remove(first_customer)

            # Insertion gloutonne dans la route
            continue_insertion = True
            while clients_remaining and continue_insertion:
                best_cost = float('inf')
                best_customer = None
                best_position = None

                # Tester chaque client restant
                for customer in clients_remaining:
                    # Vérification capacité
                    if current_capacity + self.demands[customer] > self.capacity:
                        continue

                    # Tester toutes les positions d'insertion
                    for pos in range(1, len(current_route)):
                        # Coût d'insertion
                        cost_increase = (self.distances[current_route[pos-1]][customer] +
                                       self.distances[customer][current_route[pos]] -
                                       self.distances[current_route[pos-1]][current_route[pos]])

                        if cost_increase < best_cost:
                            best_cost = cost_increase
                            best_customer = customer
                            best_position = pos

                # Insertion du meilleur client
                if best_customer is not None:
                    current_route.insert(best_position, best_customer)
                    current_capacity += self.demands[best_customer]
                    clients_remaining.remove(best_customer)
                else:
                    continue_insertion = False

            solution.append(current_route)

        print(f"   ✅ Solution générée : {len(solution)} routes")
        return solution

    def calculate_cost(self, solution):
        """
        Calcule le coût total d'une solution.

        Args:
            solution: Liste de routes

        Returns:
            float: Coût total
        """
        total_cost = 0.0
        for route in solution:
            for i in range(len(route) - 1):
                total_cost += self.distances[route[i]][route[i + 1]]
        return total_cost

    def verify_solution(self, solution):
        """
        Vérifie la validité d'une solution.

        Args:
            solution: Liste de routes

        Returns:
            tuple: (is_valid, error_message)
        """
        visited = set()

        for route_idx, route in enumerate(solution):
            # Vérifier que la route commence et finit au dépôt
            if route[0] != self.depot or route[-1] != self.depot:
                return False, f"Route {route_idx} ne commence/finit pas au dépôt"

            # Vérifier la capacité
            route_demand = sum(self.demands[c] for c in route if c != self.depot)
            if route_demand > self.capacity:
                return False, f"Route {route_idx} dépasse la capacité ({route_demand} > {self.capacity})"

            # Vérifier les doublons
            for customer in route[1:-1]:
                if customer in visited:
                    return False, f"Client {customer} visité plusieurs fois"
                visited.add(customer)

        # Vérifier que tous les clients sont visités
        all_customers = set(range(1, self.dimension))
        if visited != all_customers:
            missing = all_customers - visited
            return False, f"Clients manquants: {missing}"

        return True, "Solution valide"
