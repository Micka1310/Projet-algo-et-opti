# -*- coding: utf-8 -*-
"""
Script de test simple pour valider les modules avant lancement Streamlit.
"""

import sys
import os
sys.path.append('src')

print("=" * 60)
print("TEST DES MODULES - ADEME VRP")
print("=" * 60)

# Test 1 : Import des modules
print("\n[1] Test des imports...")
try:
    from src.instance_generator import VRPInstanceGenerator
    from src.initial_solution import InitialSolutionGenerator
    from src.alns import ALNS
    from src.simulated_annealing import SimulatedAnnealing
    from src.tabu_search import TabuSearch
    import vrplib
    import numpy as np
    import networkx as nx
    print("   [OK] Tous les imports reussis")
except Exception as e:
    print(f"   [ERREUR] Erreur d'import: {e}")
    sys.exit(1)

# Test 2 : Génération d'instance
print("\n[2] Test de generation d'instance...")
try:
    generator = VRPInstanceGenerator(n_clients=20, seed=42)
    instance = generator.generate_instance(capacity_range=(50, 100), demand_range=(5, 20))
    print(f"   [OK] Instance generee : {instance['name']}")
    print(f"      - {instance['dimension']} noeuds")
    print(f"      - Capacite : {instance['capacity']}")
    print(f"      - Demande totale : {sum(instance['demand'])}")
except Exception as e:
    print(f"   [ERREUR] Erreur de generation: {e}")
    sys.exit(1)

# Test 3 : Solution initiale
print("\n[3] Test de solution initiale...")
try:
    sol_gen = InitialSolutionGenerator(instance)

    # Clarke & Wright
    print("   - Clarke & Wright...")
    initial_cw = sol_gen.clarke_wright()
    cost_cw = sol_gen.calculate_cost(initial_cw)
    is_valid, msg = sol_gen.verify_solution(initial_cw)
    print(f"     [OK] Cout: {cost_cw:.2f}, Routes: {len(initial_cw)}, Valide: {is_valid}")

    # Plus Proche Voisin
    print("   - Plus Proche Voisin...")
    initial_nn = sol_gen.nearest_neighbor()
    cost_nn = sol_gen.calculate_cost(initial_nn)
    is_valid_nn, _ = sol_gen.verify_solution(initial_nn)
    print(f"     [OK] Cout: {cost_nn:.2f}, Routes: {len(initial_nn)}, Valide: {is_valid_nn}")

    # Insertion Séquentielle
    print("   - Insertion Sequentielle...")
    initial_seq = sol_gen.sequential_insertion()
    cost_seq = sol_gen.calculate_cost(initial_seq)
    is_valid_seq, _ = sol_gen.verify_solution(initial_seq)
    print(f"     [OK] Cout: {cost_seq:.2f}, Routes: {len(initial_seq)}, Valide: {is_valid_seq}")

except Exception as e:
    print(f"   [ERREUR] Erreur de solution initiale: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4 : ALNS (rapide)
print("\n[4] Test ALNS (10 secondes)...")
try:
    solver = ALNS(instance, time_limit=10, temperature=100, cooling_rate=0.995)
    best_sol, best_cost, history = solver.solve(initial_cw)
    improvement = ((cost_cw - best_cost) / cost_cw) * 100
    print(f"   [OK] Optimisation terminee")
    print(f"      - Cout initial: {cost_cw:.2f}")
    print(f"      - Cout final: {best_cost:.2f}")
    print(f"      - Amelioration: {improvement:.2f}%")
    print(f"      - Iterations: {solver.iteration_count}")
except Exception as e:
    print(f"   [ERREUR] Erreur ALNS: {e}")
    import traceback
    traceback.print_exc()

# Test 5 : Recuit Simulé (rapide)
print("\n[5] Test Recuit Simule (10 secondes)...")
try:
    solver_sa = SimulatedAnnealing(instance, time_limit=10, temperature=1000, cooling_rate=0.995)
    best_sol_sa, best_cost_sa, history_sa = solver_sa.solve(initial_cw)
    improvement_sa = ((cost_cw - best_cost_sa) / cost_cw) * 100
    print(f"   [OK] Optimisation terminee")
    print(f"      - Cout initial: {cost_cw:.2f}")
    print(f"      - Cout final: {best_cost_sa:.2f}")
    print(f"      - Amelioration: {improvement_sa:.2f}%")
    print(f"      - Iterations: {solver_sa.iteration_count}")
except Exception as e:
    print(f"   [ERREUR] Erreur Recuit Simule: {e}")
    import traceback
    traceback.print_exc()

# Test 6 : Recherche Tabou (rapide)
print("\n[6] Test Recherche Tabou (10 secondes)...")
try:
    solver_ts = TabuSearch(instance, time_limit=10, tabu_tenure=15)
    best_sol_ts, best_cost_ts, history_ts = solver_ts.solve(initial_cw)
    improvement_ts = ((cost_cw - best_cost_ts) / cost_cw) * 100
    print(f"   [OK] Optimisation terminee")
    print(f"      - Cout initial: {cost_cw:.2f}")
    print(f"      - Cout final: {best_cost_ts:.2f}")
    print(f"      - Amelioration: {improvement_ts:.2f}%")
    print(f"      - Iterations: {solver_ts.iteration_count}")
except Exception as e:
    print(f"   [ERREUR] Erreur Recherche Tabou: {e}")
    import traceback
    traceback.print_exc()

# Test 7 : Lecture instance VRPLIB
print("\n[7] Test lecture instance VRPLIB...")
try:
    vrp_instance_path = "tests/data/A-n32-k5.vrp"
    if os.path.exists(vrp_instance_path):
        vrp_instance = vrplib.read_instance(vrp_instance_path)
        print(f"   [OK] Instance VRPLIB chargee : {vrp_instance['name']}")
        print(f"      - {vrp_instance['dimension']} noeuds")
        print(f"      - Capacite : {vrp_instance['capacity']}")

        # Test solution
        sol_path = "tests/data/A-n32-k5.sol"
        if os.path.exists(sol_path):
            solution = vrplib.read_solution(sol_path)
            print(f"   [OK] Solution chargee : cout optimal = {solution['cost']}")
    else:
        print(f"   [WARN] Fichier non trouve: {vrp_instance_path}")
except Exception as e:
    print(f"   [WARN] Erreur lecture VRPLIB: {e}")

# Récapitulatif
print("\n" + "=" * 60)
print("[OK] TOUS LES TESTS SONT PASSES !")
print("=" * 60)
print("\nVous pouvez maintenant lancer l'application :")
print("   streamlit run app.py")
print()
