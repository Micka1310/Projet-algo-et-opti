"""
Implémentation de l'algorithme de Recherche Tabou (Tabu Search).
"""

import numpy as np
import random
import time
from copy import deepcopy
from collections import deque


class TabuSearch:
    """
    Recherche Tabou pour le CVRP.
    """

    def __init__(self, instance, time_limit=180, tabu_tenure=20, max_iterations_without_improvement=500):
        """
        Initialise l'algorithme de Recherche Tabou.

        Args:
            instance: Instance VRP (dict)
            time_limit: Temps limite en secondes
            tabu_tenure: Durée de la liste tabou
            max_iterations_without_improvement: Nombre max d'itérations sans amélioration
        """
        self.instance = instance
        self.time_limit = time_limit
        self.tabu_tenure = tabu_tenure
        self.max_iterations_without_improvement = max_iterations_without_improvement

        self.dimension = instance['dimension']
        self.capacity = instance['capacity']
        self.demands = instance['demand']
        self.distances = instance['edge_weight']

        # Liste tabou (stocke les mouvements interdits)
        self.tabu_list = deque(maxlen=tabu_tenure)

        self.best_solution = None
        self.best_cost = float('inf')
        self.current_solution = None
        self.current_cost = float('inf')

        self.cost_history = []
        self.iteration_count = 0
        self.iterations_without_improvement = 0

    def solve(self, initial_solution):
        """
        Résout le VRP avec la Recherche Tabou.

        Args:
            initial_solution: Solution initiale (liste de routes)

        Returns:
            tuple: (meilleure solution, coût, historique)
        """
        start_time = time.time()

        # Initialisation
        self.current_solution = deepcopy(initial_solution)
        self.current_cost = self._calculate_cost(self.current_solution)
        self.best_solution = deepcopy(self.current_solution)
        self.best_cost = self.current_cost

        self.cost_history = [self.best_cost]

        print(f"🔍 Recherche Tabou démarrée - Coût initial: {self.current_cost:.2f}")
        print(f"   Tenure tabou: {self.tabu_tenure}")

        # Boucle principale
        while (time.time() - start_time < self.time_limit and
               self.iterations_without_improvement < self.max_iterations_without_improvement):

            self.iteration_count += 1

            # Génération du voisinage
            neighborhood = self._generate_neighborhood(self.current_solution)

            if not neighborhood:
                print("⚠️ Aucun voisin trouvé, arrêt prématuré")
                break

            # Sélection du meilleur voisin non-tabou (ou avec critère d'aspiration)
            best_neighbor = None
            best_neighbor_cost = float('inf')
            best_move = None

            for neighbor, move in neighborhood:
                neighbor_cost = self._calculate_cost(neighbor)

                # Critère d'aspiration : accepter si meilleur que le best global
                if neighbor_cost < self.best_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor_cost
                    best_move = move
                    break

                # Sinon, vérifier si le mouvement n'est pas tabou
                if move not in self.tabu_list:
                    if neighbor_cost < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = neighbor_cost
                        best_move = move

            # Si aucun voisin acceptable trouvé
            if best_neighbor is None:
                # Forcer l'acceptation du meilleur voisin (intensification)
                best_neighbor, best_move = neighborhood[0]
                best_neighbor_cost = self._calculate_cost(best_neighbor)

            # Mise à jour de la solution courante
            self.current_solution = deepcopy(best_neighbor)
            self.current_cost = best_neighbor_cost

            # Ajout du mouvement à la liste tabou
            self.tabu_list.append(best_move)

            # Mise à jour de la meilleure solution
            if best_neighbor_cost < self.best_cost:
                self.best_solution = deepcopy(best_neighbor)
                self.best_cost = best_neighbor_cost
                self.iterations_without_improvement = 0
                print(f"✨ Iter {self.iteration_count}: Nouvelle meilleure solution! Coût: {self.best_cost:.2f}")
            else:
                self.iterations_without_improvement += 1

            # Logging périodique
            if self.iteration_count % 100 == 0:
                print(f"🔄 Iter {self.iteration_count}: Coût actuel={self.current_cost:.2f}, "
                      f"Meilleur={self.best_cost:.2f}, Sans amélioration={self.iterations_without_improvement}")

            # Enregistrement historique
            self.cost_history.append(self.best_cost)

            # Diversification si blocage
            if self.iterations_without_improvement > self.max_iterations_without_improvement // 2:
                self.current_solution = self._diversify(self.current_solution)
                self.current_cost = self._calculate_cost(self.current_solution)
                print(f"🔀 Diversification appliquée à l'itération {self.iteration_count}")
                self.iterations_without_improvement = 0

        elapsed_time = time.time() - start_time
        print(f"✅ Recherche Tabou terminée en {elapsed_time:.2f}s - {self.iteration_count} itérations")
        print(f"   Coût final: {self.best_cost:.2f}")

        return self.best_solution, self.best_cost, self.cost_history

    def _calculate_cost(self, solution):
        """Calcule le coût total d'une solution."""
        total_cost = 0.0
        for route in solution:
            for i in range(len(route) - 1):
                total_cost += self.distances[route[i]][route[i + 1]]
        return total_cost

    def _generate_neighborhood(self, solution):
        """
        Génère un voisinage complet avec différents opérateurs.

        Returns:
            list: Liste de tuples (neighbor_solution, move_description)
        """
        neighborhood = []

        # Opérateur 1: Swap intra-route
        for route_idx, route in enumerate(solution):
            if len(route) <= 3:
                continue
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    neighbor = deepcopy(solution)
                    neighbor[route_idx][i], neighbor[route_idx][j] = \
                        neighbor[route_idx][j], neighbor[route_idx][i]
                    move = ('swap_intra', route_idx, i, j)
                    neighborhood.append((neighbor, move))

        # Opérateur 2: Relocation inter-route
        if len(solution) >= 2:
            for r1_idx in range(len(solution)):
                route1 = solution[r1_idx]
                if len(route1) <= 2:
                    continue

                for r2_idx in range(len(solution)):
                    if r1_idx == r2_idx:
                        continue

                    route2 = solution[r2_idx]

                    for i in range(1, len(route1) - 1):
                        customer = route1[i]

                        # Vérification capacité
                        route2_demand = sum(self.demands[c] for c in route2 if c != 0)
                        if route2_demand + self.demands[customer] > self.capacity:
                            continue

                        for j in range(1, len(route2)):
                            neighbor = deepcopy(solution)
                            customer = neighbor[r1_idx].pop(i)
                            neighbor[r2_idx].insert(j, customer)

                            # Nettoyage routes vides
                            neighbor = [r for r in neighbor if len(r) > 2]

                            move = ('relocate', r1_idx, r2_idx, i, j)
                            neighborhood.append((neighbor, move))

        # Limitation de la taille du voisinage (pour performance)
        if len(neighborhood) > 100:
            neighborhood = random.sample(neighborhood, 100)

        return neighborhood

    def _diversify(self, solution):
        """
        Applique une stratégie de diversification pour sortir d'un optimum local.

        Args:
            solution: Solution actuelle

        Returns:
            Solution diversifiée
        """
        new_solution = deepcopy(solution)

        # Application de plusieurs perturbations
        n_perturbations = random.randint(3, 7)

        for _ in range(n_perturbations):
            operator = random.choice(['swap', 'relocate', '2opt'])

            if operator == 'swap' and len(new_solution) >= 2:
                # Swap inter-route
                r1_idx = random.randint(0, len(new_solution) - 1)
                r2_idx = random.randint(0, len(new_solution) - 1)

                while r2_idx == r1_idx and len(new_solution) > 1:
                    r2_idx = random.randint(0, len(new_solution) - 1)

                route1 = new_solution[r1_idx]
                route2 = new_solution[r2_idx]

                if len(route1) > 2 and len(route2) > 2:
                    i = random.randint(1, len(route1) - 2)
                    j = random.randint(1, len(route2) - 2)

                    # Vérification capacités
                    route1_demand = sum(self.demands[c] for c in route1 if c != 0)
                    route2_demand = sum(self.demands[c] for c in route2 if c != 0)

                    new_r1_demand = route1_demand - self.demands[route1[i]] + self.demands[route2[j]]
                    new_r2_demand = route2_demand - self.demands[route2[j]] + self.demands[route1[i]]

                    if new_r1_demand <= self.capacity and new_r2_demand <= self.capacity:
                        route1[i], route2[j] = route2[j], route1[i]

            elif operator == 'relocate' and len(new_solution) >= 2:
                # Relocation
                r1_idx = random.randint(0, len(new_solution) - 1)
                r2_idx = random.randint(0, len(new_solution) - 1)

                while r2_idx == r1_idx and len(new_solution) > 1:
                    r2_idx = random.randint(0, len(new_solution) - 1)

                route1 = new_solution[r1_idx]
                route2 = new_solution[r2_idx]

                if len(route1) > 2:
                    i = random.randint(1, len(route1) - 2)
                    customer = route1[i]

                    route2_demand = sum(self.demands[c] for c in route2 if c != 0)
                    if route2_demand + self.demands[customer] <= self.capacity:
                        route1.pop(i)
                        insert_pos = random.randint(1, len(route2) - 1)
                        route2.insert(insert_pos, customer)

                        # Nettoyage
                        new_solution = [r for r in new_solution if len(r) > 2]

            elif operator == '2opt':
                # 2-opt intra-route
                if len(new_solution) > 0:
                    r_idx = random.randint(0, len(new_solution) - 1)
                    route = new_solution[r_idx]

                    if len(route) > 3:
                        i = random.randint(1, len(route) - 2)
                        j = random.randint(1, len(route) - 2)

                        if i > j:
                            i, j = j, i

                        if i != j:
                            new_solution[r_idx] = route[:i] + route[i:j+1][::-1] + route[j+1:]

        return new_solution
