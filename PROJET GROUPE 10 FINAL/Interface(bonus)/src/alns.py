"""
Implémentation de l'algorithme ALNS (Adaptive Large Neighborhood Search).
Version avancée avec opérateurs de destruction/réparation adaptatifs.
"""

import numpy as np
import random
import time
from copy import deepcopy


class ALNS:
    """
    Adaptive Large Neighborhood Search pour le CVRP/VRPTW.
    """

    def __init__(self, instance, time_limit=180, temperature=100, cooling_rate=0.995):
        """
        Initialise l'algorithme ALNS.

        Args:
            instance: Instance VRP (dict)
            time_limit: Temps limite en secondes
            temperature: Température initiale pour le critère d'acceptation
            cooling_rate: Taux de refroidissement (alpha)
        """
        self.instance = instance
        self.time_limit = time_limit
        self.temperature = temperature
        self.cooling_rate = cooling_rate

        self.dimension = instance['dimension']
        self.capacity = instance['capacity']
        self.demands = instance['demand']
        self.distances = instance['edge_weight']

        # Poids des opérateurs (adaptatifs)
        self.destroy_weights = {'random': 1.0, 'worst': 1.0, 'shaw': 1.0}
        self.repair_weights = {'greedy': 1.0, 'regret2': 1.0}

        # Scores des opérateurs
        self.destroy_scores = {'random': 0, 'worst': 0, 'shaw': 0}
        self.repair_scores = {'greedy': 0, 'regret2': 0}

        # Paramètres de scoring
        self.sigma1 = 33  # Nouvelle meilleure solution
        self.sigma2 = 9   # Solution acceptée (amélioration)
        self.sigma3 = 13  # Solution acceptée (pas d'amélioration)

        self.best_solution = None
        self.best_cost = float('inf')
        self.current_solution = None
        self.current_cost = float('inf')

        # Historique
        self.cost_history = []
        self.iteration_count = 0

    def solve(self, initial_solution):
        """
        Résout le VRP avec ALNS.

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

        print(f"[ALNS] Demarrage - Cout initial: {self.current_cost:.2f}")

        # Boucle principale
        while time.time() - start_time < self.time_limit:
            self.iteration_count += 1

            # Sélection des opérateurs
            destroy_op = self._select_operator(self.destroy_weights)
            repair_op = self._select_operator(self.repair_weights)

            # Application des opérateurs
            new_solution = self._destroy(self.current_solution, destroy_op)
            new_solution = self._repair(new_solution, repair_op)

            # Évaluation
            new_cost = self._calculate_cost(new_solution)

            # Critère d'acceptation (Simulated Annealing)
            accept = False
            score_index = 0

            if new_cost < self.best_cost:
                # Nouvelle meilleure solution
                self.best_solution = deepcopy(new_solution)
                self.best_cost = new_cost
                self.current_solution = deepcopy(new_solution)
                self.current_cost = new_cost
                accept = True
                score_index = 1
                print(f"[ALNS] Iter {self.iteration_count}: Nouvelle meilleure solution! Cout: {self.best_cost:.2f}")

            elif new_cost < self.current_cost:
                # Amélioration de la solution courante
                self.current_solution = deepcopy(new_solution)
                self.current_cost = new_cost
                accept = True
                score_index = 2

            elif self._accept_worse_solution(new_cost):
                # Acceptation probabiliste
                self.current_solution = deepcopy(new_solution)
                self.current_cost = new_cost
                accept = True
                score_index = 3

            # Mise à jour des scores
            if accept:
                if score_index == 1:
                    self.destroy_scores[destroy_op] += self.sigma1
                    self.repair_scores[repair_op] += self.sigma1
                elif score_index == 2:
                    self.destroy_scores[destroy_op] += self.sigma2
                    self.repair_scores[repair_op] += self.sigma2
                elif score_index == 3:
                    self.destroy_scores[destroy_op] += self.sigma3
                    self.repair_scores[repair_op] += self.sigma3

            # Mise à jour des poids toutes les 100 itérations
            if self.iteration_count % 100 == 0:
                self._update_weights()
                print(f"[ALNS] Iter {self.iteration_count}: Cout actuel: {self.current_cost:.2f}, Meilleur: {self.best_cost:.2f}")

            # Refroidissement
            self.temperature *= self.cooling_rate

            # Enregistrement historique
            self.cost_history.append(self.best_cost)

        elapsed_time = time.time() - start_time
        print(f"[ALNS] Termine en {elapsed_time:.2f}s - {self.iteration_count} iterations")
        print(f"   Coût final: {self.best_cost:.2f}")

        return self.best_solution, self.best_cost, self.cost_history

    def _calculate_cost(self, solution):
        """Calcule le coût total d'une solution."""
        total_cost = 0.0
        for route in solution:
            for i in range(len(route) - 1):
                total_cost += self.distances[route[i]][route[i + 1]]
        return total_cost

    def _accept_worse_solution(self, new_cost):
        """Critère d'acceptation type recuit simulé."""
        if self.temperature <= 0:
            return False
        delta = new_cost - self.current_cost
        probability = np.exp(-delta / self.temperature)
        return random.random() < probability

    def _select_operator(self, weights):
        """Sélection d'un opérateur selon les poids adaptatifs."""
        operators = list(weights.keys())
        probs = np.array(list(weights.values()))
        probs = probs / probs.sum()  # Normalisation
        return np.random.choice(operators, p=probs)

    def _update_weights(self):
        """Mise à jour adaptive des poids des opérateurs."""
        reaction_factor = 0.1

        for op in self.destroy_weights:
            self.destroy_weights[op] = (1 - reaction_factor) * self.destroy_weights[op] + \
                                       reaction_factor * self.destroy_scores[op]
            self.destroy_scores[op] = 0  # Reset

        for op in self.repair_weights:
            self.repair_weights[op] = (1 - reaction_factor) * self.repair_weights[op] + \
                                      reaction_factor * self.repair_scores[op]
            self.repair_scores[op] = 0  # Reset

        # Assurer des poids minimaux
        for weights in [self.destroy_weights, self.repair_weights]:
            for op in weights:
                weights[op] = max(0.1, weights[op])

    # ===== OPÉRATEURS DE DESTRUCTION =====

    def _destroy(self, solution, operator):
        """Applique un opérateur de destruction."""
        if operator == 'random':
            return self._random_removal(solution)
        elif operator == 'worst':
            return self._worst_removal(solution)
        elif operator == 'shaw':
            return self._shaw_removal(solution)
        return solution

    def _random_removal(self, solution, removal_ratio=0.3):
        """Suppression aléatoire de clients."""
        new_solution = deepcopy(solution)
        all_customers = []

        # Extraction de tous les clients (sans dépôt)
        for route in new_solution:
            all_customers.extend([c for c in route if c != 0])

        # Nombre de clients à retirer
        n_remove = max(1, int(len(all_customers) * removal_ratio))
        to_remove = random.sample(all_customers, n_remove)

        # Suppression des clients
        for route in new_solution:
            route[:] = [c for c in route if c not in to_remove]

        # Nettoyage des routes vides
        new_solution = [r for r in new_solution if len(r) > 2]  # Garde [0, ..., 0]

        return new_solution, to_remove

    def _worst_removal(self, solution, removal_ratio=0.3):
        """Suppression des clients avec le plus grand coût d'insertion."""
        new_solution = deepcopy(solution)
        customers_costs = []

        # Calcul du coût d'insertion de chaque client
        for route_idx, route in enumerate(new_solution):
            for i in range(1, len(route) - 1):
                customer = route[i]
                if customer == 0:
                    continue

                # Coût actuel
                cost_with = self.distances[route[i-1]][customer] + \
                           self.distances[customer][route[i+1]]

                # Coût sans
                cost_without = self.distances[route[i-1]][route[i+1]]

                insertion_cost = cost_with - cost_without
                customers_costs.append((customer, insertion_cost, route_idx, i))

        # Tri par coût décroissant
        customers_costs.sort(key=lambda x: x[1], reverse=True)

        # Retrait des pires clients
        n_remove = max(1, int(len(customers_costs) * removal_ratio))
        to_remove = [c[0] for c in customers_costs[:n_remove]]

        # Suppression
        for route in new_solution:
            route[:] = [c for c in route if c not in to_remove]

        new_solution = [r for r in new_solution if len(r) > 2]

        return new_solution, to_remove

    def _shaw_removal(self, solution, removal_ratio=0.3):
        """Suppression de clients similaires (proximité géographique + demande)."""
        new_solution = deepcopy(solution)
        all_customers = []

        for route in new_solution:
            all_customers.extend([c for c in route if c != 0])

        if not all_customers:
            return new_solution, []

        # Sélection d'un client de départ
        seed_customer = random.choice(all_customers)

        # Calcul de la similarité (distance + différence de demande)
        similarities = []
        for c in all_customers:
            if c != seed_customer:
                dist = self.distances[seed_customer][c]
                demand_diff = abs(self.demands[seed_customer] - self.demands[c])
                similarity = dist + demand_diff * 0.1
                similarities.append((c, similarity))

        # Tri par similarité
        similarities.sort(key=lambda x: x[1])

        # Sélection des clients à retirer
        n_remove = max(1, int(len(all_customers) * removal_ratio))
        to_remove = [seed_customer] + [c[0] for c in similarities[:n_remove-1]]

        # Suppression
        for route in new_solution:
            route[:] = [c for c in route if c not in to_remove]

        new_solution = [r for r in new_solution if len(r) > 2]

        return new_solution, to_remove

    # ===== OPÉRATEURS DE RÉPARATION =====

    def _repair(self, solution_and_removed, operator):
        """Applique un opérateur de réparation."""
        if isinstance(solution_and_removed, tuple):
            solution, removed = solution_and_removed
        else:
            return solution_and_removed

        if operator == 'greedy':
            return self._greedy_insertion(solution, removed)
        elif operator == 'regret2':
            return self._regret2_insertion(solution, removed)
        return solution

    def _greedy_insertion(self, solution, removed_customers):
        """Insertion gloutonne - meilleur coût immédiat."""
        new_solution = deepcopy(solution)

        for customer in removed_customers:
            best_cost = float('inf')
            best_position = None

            # Test de toutes les positions possibles
            for route_idx, route in enumerate(new_solution):
                for pos in range(1, len(route)):
                    # Vérification capacité
                    route_demand = sum(self.demands[c] for c in route if c != 0)
                    if route_demand + self.demands[customer] > self.capacity:
                        continue

                    # Coût d'insertion
                    cost_increase = self.distances[route[pos-1]][customer] + \
                                   self.distances[customer][route[pos]] - \
                                   self.distances[route[pos-1]][route[pos]]

                    if cost_increase < best_cost:
                        best_cost = cost_increase
                        best_position = (route_idx, pos)

            # Insertion ou création de nouvelle route
            if best_position:
                route_idx, pos = best_position
                new_solution[route_idx].insert(pos, customer)
            else:
                # Nouvelle route
                new_solution.append([0, customer, 0])

        return new_solution

    def _regret2_insertion(self, solution, removed_customers):
        """Insertion avec regret-2 (différence entre meilleur et 2ème meilleur)."""
        new_solution = deepcopy(solution)

        while removed_customers:
            max_regret = -float('inf')
            best_customer = None
            best_insertion = None

            for customer in removed_customers:
                costs = []

                # Calcul des deux meilleures insertions
                for route_idx, route in enumerate(new_solution):
                    for pos in range(1, len(route)):
                        route_demand = sum(self.demands[c] for c in route if c != 0)
                        if route_demand + self.demands[customer] > self.capacity:
                            continue

                        cost_increase = self.distances[route[pos-1]][customer] + \
                                       self.distances[customer][route[pos]] - \
                                       self.distances[route[pos-1]][route[pos]]

                        costs.append((cost_increase, route_idx, pos))

                if len(costs) == 0:
                    # Nouvelle route forcée
                    regret = float('inf')
                    best_cost = 0
                    best_pos = (len(new_solution), 1)
                elif len(costs) == 1:
                    regret = costs[0][0]
                    best_cost, route_idx, pos = costs[0]
                    best_pos = (route_idx, pos)
                else:
                    costs.sort(key=lambda x: x[0])
                    regret = costs[1][0] - costs[0][0]
                    best_cost, route_idx, pos = costs[0]
                    best_pos = (route_idx, pos)

                # Mise à jour du meilleur regret
                if regret > max_regret:
                    max_regret = regret
                    best_customer = customer
                    best_insertion = best_pos

            # Insertion du client avec le plus grand regret
            if best_insertion:
                route_idx, pos = best_insertion
                if route_idx >= len(new_solution):
                    new_solution.append([0, best_customer, 0])
                else:
                    new_solution[route_idx].insert(pos, best_customer)

            removed_customers.remove(best_customer)

        return new_solution
