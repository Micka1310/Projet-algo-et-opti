"""
Implémentation de l'algorithme du Recuit Simulé (Simulated Annealing).
"""

import numpy as np
import random
import time
from copy import deepcopy


class SimulatedAnnealing:
    """
    Recuit Simulé pour le CVRP.
    """

    def __init__(self, instance, time_limit=180, temperature=1000, cooling_rate=0.995, min_temperature=0.1):
        """
        Initialise l'algorithme de Recuit Simulé.

        Args:
            instance: Instance VRP (dict)
            time_limit: Temps limite en secondes
            temperature: Température initiale
            cooling_rate: Taux de refroidissement
            min_temperature: Température minimale
        """
        self.instance = instance
        self.time_limit = time_limit
        self.temperature = temperature
        self.initial_temperature = temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature

        self.dimension = instance['dimension']
        self.capacity = instance['capacity']
        self.demands = instance['demand']
        self.distances = instance['edge_weight']

        self.best_solution = None
        self.best_cost = float('inf')
        self.current_solution = None
        self.current_cost = float('inf')

        self.cost_history = []
        self.iteration_count = 0

    def solve(self, initial_solution):
        """
        Résout le VRP avec le Recuit Simulé.

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

        print(f"🌡️ Recuit Simulé démarré - Coût initial: {self.current_cost:.2f}")
        print(f"   Température initiale: {self.temperature:.2f}")

        # Boucle principale
        while time.time() - start_time < self.time_limit and self.temperature > self.min_temperature:
            self.iteration_count += 1

            # Génération d'une solution voisine
            neighbor_solution = self._generate_neighbor(self.current_solution)
            neighbor_cost = self._calculate_cost(neighbor_solution)

            # Différence de coût
            delta = neighbor_cost - self.current_cost

            # Critère d'acceptation
            if delta < 0:
                # Meilleure solution - acceptation automatique
                self.current_solution = deepcopy(neighbor_solution)
                self.current_cost = neighbor_cost

                if neighbor_cost < self.best_cost:
                    self.best_solution = deepcopy(neighbor_solution)
                    self.best_cost = neighbor_cost
                    print(f"✨ Iter {self.iteration_count}: Nouvelle meilleure solution! Coût: {self.best_cost:.2f}")

            else:
                # Solution moins bonne - acceptation probabiliste
                probability = np.exp(-delta / self.temperature)
                if random.random() < probability:
                    self.current_solution = deepcopy(neighbor_solution)
                    self.current_cost = neighbor_cost

            # Refroidissement
            self.temperature *= self.cooling_rate

            # Logging périodique
            if self.iteration_count % 500 == 0:
                print(f"🔄 Iter {self.iteration_count}: T={self.temperature:.2f}, "
                      f"Coût actuel={self.current_cost:.2f}, Meilleur={self.best_cost:.2f}")

            # Enregistrement historique
            self.cost_history.append(self.best_cost)

        elapsed_time = time.time() - start_time
        print(f"✅ Recuit Simulé terminé en {elapsed_time:.2f}s - {self.iteration_count} itérations")
        print(f"   Température finale: {self.temperature:.2f}")
        print(f"   Coût final: {self.best_cost:.2f}")

        return self.best_solution, self.best_cost, self.cost_history

    def _calculate_cost(self, solution):
        """Calcule le coût total d'une solution."""
        total_cost = 0.0
        for route in solution:
            for i in range(len(route) - 1):
                total_cost += self.distances[route[i]][route[i + 1]]
        return total_cost

    def _generate_neighbor(self, solution):
        """
        Génère une solution voisine par opérateurs de recherche locale.
        """
        new_solution = deepcopy(solution)

        # Choix aléatoire d'un opérateur
        operator = random.choice(['2opt', 'swap', 'relocate', 'exchange'])

        if operator == '2opt':
            return self._two_opt(new_solution)
        elif operator == 'swap':
            return self._swap_customers(new_solution)
        elif operator == 'relocate':
            return self._relocate_customer(new_solution)
        elif operator == 'exchange':
            return self._exchange_customers(new_solution)

        return new_solution

    def _two_opt(self, solution):
        """Opérateur 2-opt intra-route."""
        if not solution or len(solution) == 0:
            return solution

        # Sélection aléatoire d'une route
        route_idx = random.randint(0, len(solution) - 1)
        route = solution[route_idx]

        if len(route) <= 3:  # Pas assez de clients
            return solution

        # Sélection de deux positions (sans le dépôt)
        i = random.randint(1, len(route) - 2)
        j = random.randint(1, len(route) - 2)

        if i > j:
            i, j = j, i

        if i == j:
            return solution

        # Inversion du segment
        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
        solution[route_idx] = new_route

        return solution

    def _swap_customers(self, solution):
        """Échange de deux clients dans la même route."""
        if not solution or len(solution) == 0:
            return solution

        route_idx = random.randint(0, len(solution) - 1)
        route = solution[route_idx]

        if len(route) <= 3:
            return solution

        # Sélection de deux clients (sans dépôt)
        i = random.randint(1, len(route) - 2)
        j = random.randint(1, len(route) - 2)

        # Échange
        route[i], route[j] = route[j], route[i]

        return solution

    def _relocate_customer(self, solution):
        """Relocalisation d'un client vers une autre route."""
        if len(solution) < 2:
            return solution

        # Sélection de deux routes différentes
        route1_idx = random.randint(0, len(solution) - 1)
        route2_idx = random.randint(0, len(solution) - 1)

        while route2_idx == route1_idx and len(solution) > 1:
            route2_idx = random.randint(0, len(solution) - 1)

        route1 = solution[route1_idx]
        route2 = solution[route2_idx]

        if len(route1) <= 2:  # Uniquement dépôt
            return solution

        # Sélection d'un client dans route1
        customer_idx = random.randint(1, len(route1) - 2)
        customer = route1[customer_idx]

        # Vérification capacité route2
        route2_demand = sum(self.demands[c] for c in route2 if c != 0)
        if route2_demand + self.demands[customer] > self.capacity:
            return solution

        # Retrait du client de route1
        route1.pop(customer_idx)

        # Insertion dans route2
        if len(route2) == 2:  # Que le dépôt
            insert_pos = 1
        else:
            insert_pos = random.randint(1, len(route2) - 1)

        route2.insert(insert_pos, customer)

        # Nettoyage des routes vides
        solution = [r for r in solution if len(r) > 2]

        return solution

    def _exchange_customers(self, solution):
        """Échange de clients entre deux routes différentes."""
        if len(solution) < 2:
            return solution

        # Sélection de deux routes
        route1_idx = random.randint(0, len(solution) - 1)
        route2_idx = random.randint(0, len(solution) - 1)

        while route2_idx == route1_idx and len(solution) > 1:
            route2_idx = random.randint(0, len(solution) - 1)

        route1 = solution[route1_idx]
        route2 = solution[route2_idx]

        if len(route1) <= 2 or len(route2) <= 2:
            return solution

        # Sélection des clients
        customer1_idx = random.randint(1, len(route1) - 2)
        customer2_idx = random.randint(1, len(route2) - 2)

        customer1 = route1[customer1_idx]
        customer2 = route2[customer2_idx]

        # Vérification des capacités
        route1_demand = sum(self.demands[c] for c in route1 if c != 0)
        route2_demand = sum(self.demands[c] for c in route2 if c != 0)

        new_route1_demand = route1_demand - self.demands[customer1] + self.demands[customer2]
        new_route2_demand = route2_demand - self.demands[customer2] + self.demands[customer1]

        if new_route1_demand > self.capacity or new_route2_demand > self.capacity:
            return solution

        # Échange
        route1[customer1_idx] = customer2
        route2[customer2_idx] = customer1

        return solution
