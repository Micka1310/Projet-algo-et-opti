"""
Tabu Search pour VRPTW
Implémentation pour le problème avancé avec contraintes de capacité et fenêtres temporelles
Auteur: Mickaël
"""
import vrplib
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
from typing import List, Tuple, Dict, Set
import time


class TabuSearch_VRPTW:
    """
    Classe implémentant l'algorithme Tabu Search pour le problème VRPTW
    """

    def __init__(self, instance_vrplib: Dict, initial_solution: List[List[int]]):
        """
        Initialisation de l'algorithme Tabu Search pour VRPTW

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
        self.time_windows = instance_vrplib['time_window']
        self.service_time = instance_vrplib['service_time']

        # Calcul des coûts
        self.current_cost = self.calculate_solution_cost(self.current_solution)
        self.best_cost = self.current_cost
        self.initial_cost = self.current_cost

        # Liste taboue
        self.tabu_list = []
        self.tabu_tenure = 12  # Légèrement plus élevé pour VRPTW

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

    def calculate_route_load(self, route: List[int]) -> int:
        """Calcule la charge d'une route"""
        return sum(self.demand[node] for node in route if node != 0)

    def check_time_window_feasibility(self, route: List[int]) -> bool:
        """
        Vérifie si une route respecte les fenêtres temporelles

        Args:
            route: Route à vérifier (avec dépôt)

        Returns:
            True si la route est faisable temporellement
        """
        current_time = 0.0

        for i in range(len(route) - 1):
            node_i = route[i]
            node_j = route[i + 1]

            # Temps de voyage
            travel_time = self.distance[node_i, node_j]
            arrival_time = current_time + travel_time

            # Fenêtre temporelle du nœud j
            earliest_j = self.time_windows[node_j, 0]
            latest_j = self.time_windows[node_j, 1]

            # Vérifier si on arrive trop tard
            if arrival_time > latest_j:
                return False

            # Début du service (attendre si on arrive trop tôt)
            start_service = max(arrival_time, earliest_j)

            # Fin du service
            end_service = start_service + self.service_time[node_j]

            # Mettre à jour le temps actuel
            current_time = end_service

        return True

    def is_feasible_route(self, route: List[int]) -> bool:
        """Vérifie si une route respecte toutes les contraintes"""
        # Vérifier la capacité
        load = self.calculate_route_load(route)
        if load > self.capacity:
            return False

        # Vérifier les fenêtres temporelles
        if not self.check_time_window_feasibility(route):
            return False

        return True

    def is_feasible_solution(self, solution: List[List[int]]) -> bool:
        """Vérifie si une solution est faisable"""
        for route in solution:
            if not self.is_feasible_route(route):
                return False
        return True

    # ==================== MOUVEMENTS DE VOISINAGE ====================

    def swap_within_route(self, solution: List[List[int]], route_idx: int,
                         pos1: int, pos2: int) -> List[List[int]]:
        """Échange deux clients dans la même route"""
        new_solution = copy.deepcopy(solution)
        route = new_solution[route_idx]
        route[pos1], route[pos2] = route[pos2], route[pos1]
        return new_solution

    def swap_between_routes(self, solution: List[List[int]], route1_idx: int,
                           route2_idx: int, pos1: int, pos2: int) -> List[List[int]]:
        """Échange deux clients entre deux routes différentes"""
        new_solution = copy.deepcopy(solution)
        route1 = new_solution[route1_idx]
        route2 = new_solution[route2_idx]
        route1[pos1], route2[pos2] = route2[pos2], route1[pos1]
        return new_solution

    def relocate_customer(self, solution: List[List[int]], from_route_idx: int,
                         to_route_idx: int, customer_pos: int,
                         insert_pos: int) -> List[List[int]]:
        """Déplace un client d'une route à une autre"""
        new_solution = copy.deepcopy(solution)

        # Retirer le client de la route source
        customer = new_solution[from_route_idx].pop(customer_pos)

        # Insérer dans la route destination
        new_solution[to_route_idx].insert(insert_pos, customer)

        # Supprimer les routes vides
        new_solution = [route for route in new_solution if len(route) > 2]

        return new_solution

    def two_opt_within_route(self, solution: List[List[int]], route_idx: int,
                            i: int, j: int) -> List[List[int]]:
        """Applique 2-opt sur une route (inverse un segment)"""
        new_solution = copy.deepcopy(solution)
        route = new_solution[route_idx]

        # Inverser le segment entre i et j
        route[i:j+1] = reversed(route[i:j+1])

        return new_solution

    def two_opt_between_routes(self, solution: List[List[int]], route1_idx: int,
                               route2_idx: int, pos1: int, pos2: int) -> List[List[int]]:
        """Échange les extrémités de deux routes (2-opt*)"""
        new_solution = copy.deepcopy(solution)
        route1 = new_solution[route1_idx]
        route2 = new_solution[route2_idx]

        # Échanger les segments après pos1 et pos2
        new_route1 = route1[:pos1+1] + route2[pos2+1:]
        new_route2 = route2[:pos2+1] + route1[pos1+1:]

        new_solution[route1_idx] = new_route1
        new_solution[route2_idx] = new_route2

        return new_solution

    # ==================== GESTION DE LA LISTE TABOUE ====================

    def create_move_hash(self, move_type: str, params: Tuple) -> str:
        """Crée un hash unique pour un mouvement"""
        return f"{move_type}:{params}"

    def is_tabu(self, move_hash: str) -> bool:
        """Vérifie si un mouvement est tabou"""
        return move_hash in [move[0] for move in self.tabu_list]

    def add_to_tabu_list(self, move_hash: str):
        """Ajoute un mouvement à la liste taboue"""
        self.tabu_list.append((move_hash, self.tabu_tenure))

    def update_tabu_list(self):
        """Décrémente les durées et supprime les mouvements expirés"""
        self.tabu_list = [(move, tenure - 1) for move, tenure in self.tabu_list if tenure > 1]

    def aspiration_criterion(self, new_cost: float) -> bool:
        """Critère d'aspiration : accepte un mouvement tabou s'il améliore le meilleur coût"""
        return new_cost < self.best_cost

    # ==================== GÉNÉRATION DU VOISINAGE ====================

    def generate_neighborhood(self, solution: List[List[int]],
                            neighborhood_size: int = 40) -> List[Tuple]:
        """
        Génère un échantillon du voisinage de la solution (adapté VRPTW)

        Returns:
            Liste de (nouvelle_solution, move_hash, nouveau_coût)
        """
        neighbors = []

        # 1. Swap within route (échange dans la même route)
        for route_idx, route in enumerate(solution):
            if len(route) <= 3:  # Seulement dépôt
                continue

            # Limiter le nombre d'essais (moins pour VRPTW car plus de contraintes)
            num_swaps = min(3, (len(route) - 2) * (len(route) - 3) // 2)
            for _ in range(num_swaps):
                pos1 = random.randint(1, len(route) - 2)
                pos2 = random.randint(1, len(route) - 2)
                if pos1 != pos2:
                    new_sol = self.swap_within_route(solution, route_idx, pos1, pos2)
                    if self.is_feasible_solution(new_sol):
                        move_hash = self.create_move_hash("swap_within",
                                                         (route_idx, pos1, pos2))
                        cost = self.calculate_solution_cost(new_sol)
                        neighbors.append((new_sol, move_hash, cost))

        # 2. Swap between routes (échange entre routes)
        for r1_idx in range(len(solution)):
            for r2_idx in range(r1_idx + 1, len(solution)):
                route1, route2 = solution[r1_idx], solution[r2_idx]
                if len(route1) <= 2 or len(route2) <= 2:
                    continue

                # Limiter les essais
                num_swaps = min(2, (len(route1) - 2) * (len(route2) - 2))
                for _ in range(num_swaps):
                    pos1 = random.randint(1, len(route1) - 2)
                    pos2 = random.randint(1, len(route2) - 2)

                    new_sol = self.swap_between_routes(solution, r1_idx, r2_idx,
                                                       pos1, pos2)
                    if self.is_feasible_solution(new_sol):
                        move_hash = self.create_move_hash("swap_between",
                                                         (r1_idx, r2_idx, pos1, pos2))
                        cost = self.calculate_solution_cost(new_sol)
                        neighbors.append((new_sol, move_hash, cost))

        # 3. Relocate (déplacement d'un client)
        for from_idx, from_route in enumerate(solution):
            if len(from_route) <= 3:
                continue

            for to_idx, to_route in enumerate(solution):
                if from_idx == to_idx:
                    continue

                # Limiter les essais
                num_relocations = min(2, len(from_route) - 2)
                for _ in range(num_relocations):
                    cust_pos = random.randint(1, len(from_route) - 2)
                    insert_pos = random.randint(1, len(to_route) - 1)

                    new_sol = self.relocate_customer(solution, from_idx, to_idx,
                                                     cust_pos, insert_pos)
                    if self.is_feasible_solution(new_sol):
                        move_hash = self.create_move_hash("relocate",
                                                         (from_idx, to_idx, cust_pos))
                        cost = self.calculate_solution_cost(new_sol)
                        neighbors.append((new_sol, move_hash, cost))

        # 4. 2-opt within route
        for route_idx, route in enumerate(solution):
            if len(route) <= 4:
                continue

            num_2opts = min(2, (len(route) - 2) * (len(route) - 3) // 2)
            for _ in range(num_2opts):
                i = random.randint(1, len(route) - 3)
                j = random.randint(i + 1, len(route) - 2)

                new_sol = self.two_opt_within_route(solution, route_idx, i, j)
                if self.is_feasible_solution(new_sol):
                    move_hash = self.create_move_hash("2opt_within",
                                                     (route_idx, i, j))
                    cost = self.calculate_solution_cost(new_sol)
                    neighbors.append((new_sol, move_hash, cost))

        # Limiter la taille du voisinage
        if len(neighbors) > neighborhood_size:
            neighbors = random.sample(neighbors, neighborhood_size)

        return neighbors

    # ==================== ALGORITHME PRINCIPAL ====================

    def run(self, max_iterations: int = 3000, tabu_tenure: int = 12,
            neighborhood_size: int = 40, diversification_freq: int = 120) -> Dict:
        """
        Exécute l'algorithme Tabu Search pour VRPTW

        Args:
            max_iterations: Nombre d'itérations maximum
            tabu_tenure: Durée de vie d'un mouvement dans la liste taboue
            neighborhood_size: Taille de l'échantillon du voisinage
            diversification_freq: Fréquence de diversification

        Returns:
            Dictionnaire avec les résultats
        """
        print(f"\n{'='*60}")
        print(f"Démarrage de l'algorithme Tabu Search pour VRPTW")
        print(f"{'='*60}")
        print(f"Coût initial: {self.initial_cost:.2f}")
        print(f"Nombre d'itérations: {max_iterations}")
        print(f"Tenure taboue: {tabu_tenure}")
        print(f"Taille du voisinage: {neighborhood_size}")

        self.tabu_tenure = tabu_tenure
        start_time = time.time()
        iterations_without_improvement = 0

        for iteration in range(max_iterations):
            iter_start = time.time()

            # Générer le voisinage
            neighbors = self.generate_neighborhood(self.current_solution,
                                                   neighborhood_size)

            if not neighbors:
                print(f"Itération {iteration+1}: Aucun voisin faisable trouvé")
                break

            # Trouver le meilleur mouvement non-tabou
            best_neighbor = None
            best_neighbor_cost = float('inf')
            best_move_hash = None

            for neighbor, move_hash, cost in neighbors:
                # Accepter si non-tabou OU si critère d'aspiration est satisfait
                if not self.is_tabu(move_hash) or self.aspiration_criterion(cost):
                    if cost < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = cost
                        best_move_hash = move_hash

            # Si aucun mouvement acceptable trouvé, prendre le meilleur même si tabou
            if best_neighbor is None:
                best_neighbor, best_move_hash, best_neighbor_cost = min(
                    neighbors, key=lambda x: x[2]
                )

            # Mettre à jour la solution courante
            self.current_solution = best_neighbor
            self.current_cost = best_neighbor_cost

            # Ajouter le mouvement à la liste taboue
            self.add_to_tabu_list(best_move_hash)

            # Mettre à jour la meilleure solution
            if self.current_cost < self.best_cost:
                self.best_solution = copy.deepcopy(self.current_solution)
                self.best_cost = self.current_cost
                iterations_without_improvement = 0
                print(f"Itération {iteration+1}: Nouvelle meilleure solution! Coût = {self.best_cost:.2f}")
            else:
                iterations_without_improvement += 1

            # Mise à jour de la liste taboue
            self.update_tabu_list()

            # Diversification (perturbation) si stagnation
            if iterations_without_improvement > 0 and iteration % diversification_freq == 0:
                print(f"Itération {iteration+1}: Diversification appliquée")
                self.diversify_solution()

            # Enregistrement de l'historique
            self.cost_history.append(self.current_cost)
            self.best_cost_history.append(self.best_cost)
            iter_time = time.time() - iter_start
            self.iteration_times.append(iter_time)

            # Affichage périodique
            if (iteration + 1) % 300 == 0:
                improvement = ((self.initial_cost - self.best_cost) / self.initial_cost) * 100
                print(f"Itération {iteration+1}/{max_iterations} | "
                      f"Meilleur: {self.best_cost:.2f} | "
                      f"Actuel: {self.current_cost:.2f} | "
                      f"Amélioration: {improvement:.2f}% | "
                      f"Tabu list size: {len(self.tabu_list)}")

        total_time = time.time() - start_time

        # Résultats finaux
        final_improvement = ((self.initial_cost - self.best_cost) / self.initial_cost) * 100

        print(f"\n{'='*60}")
        print(f"Tabu Search terminé!")
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

    def diversify_solution(self):
        """Applique une perturbation pour diversifier la recherche (adapté VRPTW)"""
        # Effectuer plusieurs mouvements aléatoires
        num_perturbations = random.randint(2, 5)  # Moins agressif pour VRPTW

        for _ in range(num_perturbations):
            # Choisir un type de mouvement aléatoire
            move_type = random.choice(['swap', 'relocate', '2opt'])

            if move_type == 'swap' and len(self.current_solution) >= 2:
                # Swap entre deux routes aléatoires
                r1_idx = random.randint(0, len(self.current_solution) - 1)
                r2_idx = random.randint(0, len(self.current_solution) - 1)

                route1 = self.current_solution[r1_idx]
                route2 = self.current_solution[r2_idx]

                if len(route1) > 2 and len(route2) > 2:
                    pos1 = random.randint(1, len(route1) - 2)
                    pos2 = random.randint(1, len(route2) - 2)

                    new_sol = self.swap_between_routes(self.current_solution,
                                                       r1_idx, r2_idx, pos1, pos2)
                    if self.is_feasible_solution(new_sol):
                        self.current_solution = new_sol
                        self.current_cost = self.calculate_solution_cost(new_sol)

            elif move_type == 'relocate' and len(self.current_solution) >= 2:
                # Relocation aléatoire
                from_idx = random.randint(0, len(self.current_solution) - 1)
                to_idx = random.randint(0, len(self.current_solution) - 1)

                from_route = self.current_solution[from_idx]
                to_route = self.current_solution[to_idx]

                if len(from_route) > 3:
                    cust_pos = random.randint(1, len(from_route) - 2)
                    insert_pos = random.randint(1, len(to_route) - 1)

                    new_sol = self.relocate_customer(self.current_solution,
                                                     from_idx, to_idx,
                                                     cust_pos, insert_pos)
                    if self.is_feasible_solution(new_sol):
                        self.current_solution = new_sol
                        self.current_cost = self.calculate_solution_cost(new_sol)

            elif move_type == '2opt':
                # 2-opt sur une route aléatoire
                route_idx = random.randint(0, len(self.current_solution) - 1)
                route = self.current_solution[route_idx]

                if len(route) > 4:
                    i = random.randint(1, len(route) - 3)
                    j = random.randint(i + 1, len(route) - 2)

                    new_sol = self.two_opt_within_route(self.current_solution,
                                                        route_idx, i, j)
                    if self.is_feasible_solution(new_sol):
                        self.current_solution = new_sol
                        self.current_cost = self.calculate_solution_cost(new_sol)

    def plot_convergence(self, save_path: str = None):
        """Affiche le graphique de convergence"""
        plt.figure(figsize=(12, 6))

        iterations = range(len(self.cost_history))
        plt.plot(iterations, self.cost_history, label='Coût actuel',
                alpha=0.6, linewidth=0.8)
        plt.plot(iterations, self.best_cost_history, label='Meilleur coût',
                 linewidth=2, color='red')

        plt.xlabel('Itération')
        plt.ylabel('Coût de la solution')
        plt.title('Convergence de l\'algorithme Tabu Search pour VRPTW')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
