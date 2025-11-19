
import vrplib
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
from typing import List, Tuple, Dict
import time


class ALNS_CVRP:
    """
    Classe implémentant l'algorithme ALNS pour le problème CVRP
    """

    def __init__(self, instance_vrplib: Dict, initial_solution: List[List[int]]):
        """
        Initialisation de l'algorithme ALNS

        Args:
            instance_vrplib: Dictionnaire contenant les données de l'instance
            initial_solution: Solution initiale (liste de routes avec dépôt)
        """
        self.instance = instance_vrplib
        self.initial_solution = copy.deepcopy(initial_solution)
        self.current_solution = copy.deepcopy(initial_solution)
        self.best_solution = copy.deepcopy(initial_solution)

        # Paramètres de l'instance
        self.distance = instance_vrplib['edge_weight']
        self.demand = instance_vrplib['demand']
        self.capacity = instance_vrplib['capacity']

        # Calcul des coûts
        self.current_cost = self.calculate_solution_cost(self.current_solution)
        self.best_cost = self.current_cost
        self.initial_cost = self.current_cost

        # Paramètres ALNS
        self.destroy_operators = [
            self.random_removal,
            self.worst_removal,
            self.shaw_removal
        ]
        self.repair_operators = [
            self.greedy_insertion,
            self.regret_insertion
        ]

        # Poids adaptatifs (initialisés à 1.0)
        self.destroy_weights = [1.0] * len(self.destroy_operators)
        self.repair_weights = [1.0] * len(self.repair_operators)

        # Scores pour mise à jour des poids
        self.destroy_scores = [0] * len(self.destroy_operators)
        self.repair_scores = [0] * len(self.repair_operators)
        self.destroy_usage = [0] * len(self.destroy_operators)
        self.repair_usage = [0] * len(self.repair_operators)

        # Paramètres de scoring
        self.sigma1 = 33  # Score pour nouvelle meilleure solution
        self.sigma2 = 9   # Score pour solution acceptée
        self.sigma3 = 13  # Score pour solution améliorée (non meilleure globale)

        # Paramètres Simulated Annealing
        self.temperature = 0.0
        self.cooling_rate = 0.99975

        # Historique pour analyse
        self.cost_history = [self.current_cost]
        self.best_cost_history = [self.best_cost]
        self.iteration_times = []

    def calculate_solution_cost(self, solution: List[List[int]]) -> float:
        """Calcule le coût total d'une solution"""
        total_cost = 0.0
        for route in solution:
            for i in range(len(route) - 1):
                total_cost += self.distance[route[i], route[i + 1]]
        return total_cost

    def is_feasible_route(self, route: List[int]) -> bool:
        """Vérifie si une route respecte la contrainte de capacité"""
        load = sum(self.demand[node] for node in route if node != 0)
        return load <= self.capacity

    def is_feasible_solution(self, solution: List[List[int]]) -> bool:
        """Vérifie si une solution est faisable"""
        for route in solution:
            if not self.is_feasible_route(route):
                return False
        return True

    # ==================== OPÉRATEURS DE DESTRUCTION ====================

    def random_removal(self, solution: List[List[int]], num_remove: int) -> Tuple[List[List[int]], List[int]]:
        """Supprime aléatoirement num_remove clients"""
        sol_copy = copy.deepcopy(solution)
        removed_customers = []

        # Collecter tous les clients (sans le dépôt)
        all_customers = []
        for route in sol_copy:
            for node in route:
                if node != 0:
                    all_customers.append(node)

        # Supprimer aléatoirement num_remove clients
        if len(all_customers) >= num_remove:
            removed_customers = random.sample(all_customers, num_remove)
        else:
            removed_customers = all_customers.copy()

        # Supprimer les clients des routes
        for customer in removed_customers:
            for route in sol_copy:
                if customer in route:
                    route.remove(customer)

        # Supprimer les routes vides
        sol_copy = [route for route in sol_copy if len(route) > 2]

        return sol_copy, removed_customers

    def worst_removal(self, solution: List[List[int]], num_remove: int) -> Tuple[List[List[int]], List[int]]:
        """Supprime les clients qui contribuent le plus au coût"""
        sol_copy = copy.deepcopy(solution)
        removed_customers = []

        # Calculer le coût de retrait de chaque client
        customer_costs = []
        for route in sol_copy:
            for i in range(1, len(route) - 1):
                customer = route[i]
                # Coût de retrait = coût actuel - coût sans le client
                current_cost = (self.distance[route[i-1], customer] +
                               self.distance[customer, route[i+1]])
                direct_cost = self.distance[route[i-1], route[i+1]]
                saving = current_cost - direct_cost
                customer_costs.append((customer, saving))

        # Trier par coût décroissant et supprimer les pires
        customer_costs.sort(key=lambda x: x[1], reverse=True)
        num_to_remove = min(num_remove, len(customer_costs))

        for i in range(num_to_remove):
            customer = customer_costs[i][0]
            removed_customers.append(customer)
            for route in sol_copy:
                if customer in route:
                    route.remove(customer)

        # Supprimer les routes vides
        sol_copy = [route for route in sol_copy if len(route) > 2]

        return sol_copy, removed_customers

    def shaw_removal(self, solution: List[List[int]], num_remove: int) -> Tuple[List[List[int]], List[int]]:
        """Supprime des clients similaires (basé sur la distance)"""
        sol_copy = copy.deepcopy(solution)
        removed_customers = []

        # Collecter tous les clients
        all_customers = []
        for route in sol_copy:
            for node in route:
                if node != 0:
                    all_customers.append(node)

        if len(all_customers) == 0:
            return sol_copy, removed_customers

        # Sélectionner un client initial aléatoirement
        seed_customer = random.choice(all_customers)
        removed_customers.append(seed_customer)

        # Supprimer le seed customer
        for route in sol_copy:
            if seed_customer in route:
                route.remove(seed_customer)

        # Supprimer des clients similaires (proches du seed)
        remaining_customers = [c for c in all_customers if c != seed_customer]

        while len(removed_customers) < num_remove and remaining_customers:
            # Calculer la distance moyenne des clients restants au seed
            distances = []
            for customer in remaining_customers:
                dist = self.distance[seed_customer, customer]
                distances.append((customer, dist))

            # Trier par distance et sélectionner aléatoirement parmi les plus proches
            distances.sort(key=lambda x: x[1])
            num_candidates = min(5, len(distances))
            candidate = random.choice(distances[:num_candidates])[0]

            removed_customers.append(candidate)
            remaining_customers.remove(candidate)

            for route in sol_copy:
                if candidate in route:
                    route.remove(candidate)

        # Supprimer les routes vides
        sol_copy = [route for route in sol_copy if len(route) > 2]

        return sol_copy, removed_customers

    # ==================== OPÉRATEURS DE RÉPARATION ====================

    def greedy_insertion(self, partial_solution: List[List[int]],
                         removed_customers: List[int]) -> List[List[int]]:
        """Insertion gloutonne : insère chaque client à la meilleure position"""
        solution = copy.deepcopy(partial_solution)
        unserved = removed_customers.copy()

        while unserved:
            best_customer = None
            best_route_idx = None
            best_position = None
            best_cost_increase = float('inf')

            # Essayer d'insérer chaque client non servi
            for customer in unserved:
                # Essayer dans chaque route existante
                for route_idx, route in enumerate(solution):
                    # Essayer chaque position dans la route
                    for pos in range(1, len(route)):
                        # Vérifier la faisabilité
                        test_route = route.copy()
                        test_route.insert(pos, customer)

                        if self.is_feasible_route(test_route):
                            # Calculer l'augmentation de coût
                            cost_increase = (self.distance[route[pos-1], customer] +
                                           self.distance[customer, route[pos]] -
                                           self.distance[route[pos-1], route[pos]])

                            if cost_increase < best_cost_increase:
                                best_cost_increase = cost_increase
                                best_customer = customer
                                best_route_idx = route_idx
                                best_position = pos

                # Essayer de créer une nouvelle route
                new_route = [0, customer, 0]
                if self.is_feasible_route(new_route):
                    cost_increase = 2 * self.distance[0, customer]
                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_customer = customer
                        best_route_idx = -1
                        best_position = -1

            # Insérer le meilleur client trouvé
            if best_customer is not None:
                if best_route_idx == -1:
                    # Créer une nouvelle route
                    solution.append([0, best_customer, 0])
                else:
                    # Insérer dans la route existante
                    solution[best_route_idx].insert(best_position, best_customer)

                unserved.remove(best_customer)
            else:
                # Impossible d'insérer ce client, créer une nouvelle route de force
                customer = unserved.pop(0)
                solution.append([0, customer, 0])

        return solution

    def regret_insertion(self, partial_solution: List[List[int]],
                         removed_customers: List[int]) -> List[List[int]]:
        """Insertion avec regret : priorise les clients difficiles à insérer"""
        solution = copy.deepcopy(partial_solution)
        unserved = removed_customers.copy()

        while unserved:
            best_customer = None
            best_route_idx = None
            best_position = None
            max_regret = -float('inf')
            best_cost = float('inf')

            # Calculer le regret pour chaque client non servi
            for customer in unserved:
                insertion_costs = []
                insertion_details = []

                # Trouver les meilleures insertions pour ce client
                for route_idx, route in enumerate(solution):
                    for pos in range(1, len(route)):
                        test_route = route.copy()
                        test_route.insert(pos, customer)

                        if self.is_feasible_route(test_route):
                            cost_increase = (self.distance[route[pos-1], customer] +
                                           self.distance[customer, route[pos]] -
                                           self.distance[route[pos-1], route[pos]])
                            insertion_costs.append(cost_increase)
                            insertion_details.append((route_idx, pos, cost_increase))

                # Ajouter l'option de nouvelle route
                new_route_cost = 2 * self.distance[0, customer]
                insertion_costs.append(new_route_cost)
                insertion_details.append((-1, -1, new_route_cost))

                # Calculer le regret (différence entre meilleure et deuxième meilleure)
                if len(insertion_costs) >= 2:
                    insertion_costs.sort()
                    regret = insertion_costs[1] - insertion_costs[0]
                else:
                    regret = 0

                # Sélectionner le client avec le plus grand regret
                if regret > max_regret or (regret == max_regret and insertion_costs[0] < best_cost):
                    max_regret = regret
                    best_cost = insertion_costs[0]
                    best_customer = customer

                    # Trouver les détails de la meilleure insertion
                    for route_idx, pos, cost in insertion_details:
                        if cost == insertion_costs[0]:
                            best_route_idx = route_idx
                            best_position = pos
                            break

            # Insérer le client avec le plus grand regret
            if best_customer is not None:
                if best_route_idx == -1:
                    solution.append([0, best_customer, 0])
                else:
                    solution[best_route_idx].insert(best_position, best_customer)

                unserved.remove(best_customer)
            else:
                # Fallback
                customer = unserved.pop(0)
                solution.append([0, customer, 0])

        return solution

    # ==================== MÉCANISME ADAPTATIF ====================

    def select_operator(self, weights: List[float]) -> int:
        """Sélectionne un opérateur selon les poids (roulette wheel)"""
        total_weight = sum(weights)
        if total_weight == 0:
            return random.randint(0, len(weights) - 1)

        probabilities = [w / total_weight for w in weights]
        return np.random.choice(len(weights), p=probabilities)

    def update_weights(self, segment: int):
        """Mise à jour des poids selon les performances"""
        reaction_factor = 0.1

        # Mise à jour des poids de destruction
        for i in range(len(self.destroy_operators)):
            if self.destroy_usage[i] > 0:
                avg_score = self.destroy_scores[i] / self.destroy_usage[i]
                self.destroy_weights[i] = (self.destroy_weights[i] * (1 - reaction_factor) +
                                           reaction_factor * avg_score)
                # Réinitialiser les compteurs
                self.destroy_scores[i] = 0
                self.destroy_usage[i] = 0

        # Mise à jour des poids de réparation
        for i in range(len(self.repair_operators)):
            if self.repair_usage[i] > 0:
                avg_score = self.repair_scores[i] / self.repair_usage[i]
                self.repair_weights[i] = (self.repair_weights[i] * (1 - reaction_factor) +
                                          reaction_factor * avg_score)
                # Réinitialiser les compteurs
                self.repair_scores[i] = 0
                self.repair_usage[i] = 0

    def accept_solution(self, new_cost: float, current_cost: float) -> bool:
        """Critère d'acceptation avec Simulated Annealing"""
        if new_cost < current_cost:
            return True

        if self.temperature > 0:
            probability = np.exp(-(new_cost - current_cost) / self.temperature)
            return random.random() < probability

        return False

    # ==================== ALGORITHME PRINCIPAL ====================

    def run(self, max_iterations: int = 500, destruction_rate: float = 0.3,
            segment_size: int = 100, initial_temp: float = None) -> Dict:
        """
        Exécute l'algorithme ALNS

        Args:
            max_iterations: Nombre d'itérations maximum
            destruction_rate: Pourcentage de clients à détruire (0.0 - 1.0)
            segment_size: Taille des segments pour mise à jour des poids
            initial_temp: Température initiale (None pour auto-calcul)

        Returns:
            Dictionnaire avec les résultats
        """
        print(f"\n{'='*60}")
        print(f"Démarrage de l'algorithme ALNS pour CVRP")
        print(f"{'='*60}")
        print(f"Coût initial: {self.initial_cost:.2f}")
        print(f"Nombre d'itérations: {max_iterations}")
        print(f"Taux de destruction: {destruction_rate*100:.1f}%")

        # Calcul température initiale si non fournie
        if initial_temp is None:
            self.temperature = 0.05 * self.initial_cost
        else:
            self.temperature = initial_temp

        print(f"Température initiale: {self.temperature:.2f}")

        # Compter le nombre total de clients
        num_customers = sum(1 for route in self.current_solution
                           for node in route if node != 0)
        num_to_remove = max(1, int(num_customers * destruction_rate))

        start_time = time.time()
        iterations_without_improvement = 0

        for iteration in range(max_iterations):
            iter_start = time.time()

            # Sélectionner les opérateurs
            destroy_idx = self.select_operator(self.destroy_weights)
            repair_idx = self.select_operator(self.repair_weights)

            # Appliquer destruction
            partial_solution, removed = self.destroy_operators[destroy_idx](
                self.current_solution, num_to_remove
            )

            # Appliquer réparation
            new_solution = self.repair_operators[repair_idx](
                partial_solution, removed
            )

            # Calculer le coût de la nouvelle solution
            if self.is_feasible_solution(new_solution):
                new_cost = self.calculate_solution_cost(new_solution)

                # Mise à jour des scores
                score = 0
                if new_cost < self.best_cost:
                    # Nouvelle meilleure solution
                    self.best_solution = copy.deepcopy(new_solution)
                    self.best_cost = new_cost
                    self.current_solution = copy.deepcopy(new_solution)
                    self.current_cost = new_cost
                    score = self.sigma1
                    iterations_without_improvement = 0
                    print(f"Itération {iteration+1}: Nouvelle meilleure solution! Coût = {self.best_cost:.2f}")
                elif new_cost < self.current_cost:
                    # Solution améliorée
                    self.current_solution = copy.deepcopy(new_solution)
                    self.current_cost = new_cost
                    score = self.sigma3
                    iterations_without_improvement += 1
                elif self.accept_solution(new_cost, self.current_cost):
                    # Solution acceptée
                    self.current_solution = copy.deepcopy(new_solution)
                    self.current_cost = new_cost
                    score = self.sigma2
                    iterations_without_improvement += 1
                else:
                    iterations_without_improvement += 1

                # Mise à jour des scores des opérateurs
                self.destroy_scores[destroy_idx] += score
                self.destroy_usage[destroy_idx] += 1
                self.repair_scores[repair_idx] += score
                self.repair_usage[repair_idx] += 1

            # Mise à jour des poids tous les segment_size itérations
            if (iteration + 1) % segment_size == 0:
                self.update_weights(iteration // segment_size)

            # Refroidissement de la température
            self.temperature *= self.cooling_rate

            # Enregistrement de l'historique
            self.cost_history.append(self.current_cost)
            self.best_cost_history.append(self.best_cost)
            iter_time = time.time() - iter_start
            self.iteration_times.append(iter_time)

            # Affichage périodique
            if (iteration + 1) % 500 == 0:
                improvement = ((self.initial_cost - self.best_cost) / self.initial_cost) * 100
                print(f"Itération {iteration+1}/{max_iterations} | "
                      f"Meilleur: {self.best_cost:.2f} | "
                      f"Actuel: {self.current_cost:.2f} | "
                      f"Amélioration: {improvement:.2f}% | "
                      f"Temp: {self.temperature:.4f}")

        total_time = time.time() - start_time

        # Résultats finaux
        final_improvement = ((self.initial_cost - self.best_cost) / self.initial_cost) * 100

        print(f"\n{'='*60}")
        print(f"ALNS terminé!")
        print(f"{'='*60}")
        print(f"Coût initial:      {self.initial_cost:.2f}")
        print(f"Coût final:        {self.best_cost:.2f}")
        print(f"Amélioration:      {final_improvement:.2f}%")
        print(f"Temps total:       {total_time:.2f}s")
        print(f"Nombre de routes:  {len(self.best_solution)}")

        return {
            'best_solution': self.best_solution,
            'best_cost': self.best_cost,
            'initial_cost': self.initial_cost,
            'improvement_pct': final_improvement,
            'cost_history': self.cost_history,
            'best_cost_history': self.best_cost_history,
            'iteration_times': self.iteration_times,
            'total_time': total_time
        }

    def plot_convergence(self, save_path: str = None):
        """Affiche le graphique de convergence"""
        plt.figure(figsize=(12, 6))

        iterations = range(len(self.cost_history))
        plt.plot(iterations, self.cost_history, label='Coût actuel', alpha=0.6, linewidth=0.8)
        plt.plot(iterations, self.best_cost_history, label='Meilleur coût',
                 linewidth=2, color='red')

        plt.xlabel('Itération')
        plt.ylabel('Coût de la solution')
        plt.title('Convergence de l\'algorithme ALNS pour CVRP')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
