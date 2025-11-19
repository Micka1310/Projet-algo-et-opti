"""
Interface Streamlit pour l'optimisation de tournées de livraison (ADEME).
Application interactive pour résoudre le VRP avec différentes métaheuristiques.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import vrplib
import time
from pathlib import Path
import io

# Import des modules locaux
import sys
sys.path.append('src')
from instance_generator import VRPInstanceGenerator, save_instance_vrplib_format
from initial_solution import InitialSolutionGenerator
from alns import ALNS
from simulated_annealing import SimulatedAnnealing
from tabu_search import TabuSearch
from vrptw_parser import load_vrptw_instance, verify_time_windows
from vrptw_constraints import is_solution_feasible, calculate_solution_total_time


# Configuration de la page
st.set_page_config(
    page_title="ADEME - Optimisation VRP",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Fonction principale de l'application."""

    # En-tête
    st.markdown('<h1 class="main-header">🚚 ADEME - Optimisation de Tournées de Livraison</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    **Projet CesiCDP** - Algorithmique & Optimisation Combinatoire
    *Résolution du Vehicle Routing Problem (VRP) avec métaheuristiques avancées*
    """)

    st.divider()

    # Sidebar - Configuration
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=ADEME+Logo",
                 use_container_width=True)

        st.header("⚙️ Configuration")

        # Choix du mode
        mode = st.radio(
            "Mode d'instance",
            ["📤 Charger une instance VRP", "🎲 Générer une instance"],
            index=0
        )

        st.divider()

        # Choix de l'algorithme
        st.subheader("🧠 Algorithme")
        algorithm = st.selectbox(
            "Métaheuristique",
            ["ALNS", "Recuit Simulé", "Recherche Tabou"],
            help="Sélectionnez l'algorithme d'optimisation"
        )

        # Paramètres de l'algorithme
        st.subheader("⏱️ Paramètres")
        time_limit = st.slider("Temps limite (secondes)", 10, 300, 60, 10)

        if algorithm == "ALNS":
            temperature = st.slider("Température initiale", 10, 200, 100, 10)
            cooling_rate = st.slider("Taux de refroidissement", 0.90, 0.999, 0.995, 0.001)
        elif algorithm == "Recuit Simulé":
            temperature = st.slider("Température initiale", 100, 2000, 1000, 100)
            cooling_rate = st.slider("Taux de refroidissement", 0.90, 0.999, 0.995, 0.001)
        else:  # Recherche Tabou
            tabu_tenure = st.slider("Tenure tabou", 5, 50, 20, 5)
            max_iter_no_improve = st.slider("Max itérations sans amélioration", 100, 1000, 500, 100)

        st.divider()

        # Méthode de solution initiale
        st.subheader("🏗️ Solution Initiale")
        initial_method = st.selectbox(
            "Méthode",
            ["Clarke & Wright", "Plus Proche Voisin", "Insertion Séquentielle"],
            help="Heuristique pour la solution de départ"
        )

        # Stockage des paramètres dans session_state
        st.session_state['algorithm'] = algorithm
        st.session_state['time_limit'] = time_limit
        st.session_state['initial_method'] = initial_method

        if algorithm == "ALNS":
            st.session_state['temperature'] = temperature
            st.session_state['cooling_rate'] = cooling_rate
        elif algorithm == "Recuit Simulé":
            st.session_state['temperature'] = temperature
            st.session_state['cooling_rate'] = cooling_rate
        else:  # Recherche Tabou
            st.session_state['tabu_tenure'] = tabu_tenure
            st.session_state['max_iter_no_improve'] = max_iter_no_improve

    # Zone principale
    if mode == "📤 Charger une instance VRP":
        load_instance_mode()
    else:
        generate_instance_mode()

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>Développé par <b>CesiCDP Team</b> | Tuteur: Mohamed Belgacem | FISE A3 Informatique 2025</p>
        <p>📚 <a href='https://github.com/CesiCDP/ademe-vrp'>Code Source</a> |
        📖 <a href='#'>Documentation</a> |
        🌱 <a href='https://www.ademe.fr/'>ADEME</a></p>
    </div>
    """, unsafe_allow_html=True)


def load_instance_mode():
    """Mode : Chargement d'une instance VRP existante."""

    st.header("📤 Charger une Instance VRP")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Upload de fichier
        uploaded_file = st.file_uploader(
            "Fichier d'instance (.vrp, .txt, .vrptw)",
            type=['vrp', 'txt', 'vrptw'],
            help="Format VRPLIB, Solomon (.txt) ou VRPTW (.vrptw)"
        )

        # Fichier solution optionnel
        solution_file = st.file_uploader(
            "Fichier solution optionnel (.sol)",
            type=['sol', 'txt'],
            help="Solution de référence pour comparaison"
        )

    with col2:
        st.info("""
        **Formats supportés:**
        - VRPLIB (.vrp)
        - Solomon (.txt) avec fenêtres temporelles
        - VRPTW (.vrptw) avec fenêtres temporelles

        **Instances disponibles:**
        - tests/data/A-n32-k5.vrp
        - instances avancées/ (VRPTW Solomon)
        """)

    if uploaded_file is not None:
        # Chargement de l'instance
        try:
            # Sauvegarder temporairement le fichier
            file_extension = uploaded_file.name.split('.')[-1]
            temp_path = Path(f"temp_instance.{file_extension}")
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            # Lecture selon le format
            if file_extension in ['txt', 'vrptw']:
                # Utiliser le nouveau parser VRPTW
                instance = load_vrptw_instance(str(temp_path))
                st.success(f"✅ Instance VRPTW chargée : **{instance['name']}**")

                # Vérification des fenêtres temporelles
                is_valid, messages = verify_time_windows(instance)
                if not is_valid:
                    st.warning("⚠️ Problèmes détectés dans les fenêtres temporelles:")
                    for msg in messages:
                        st.write(f"  - {msg}")
            else:
                # Utiliser vrplib pour .vrp
                instance = vrplib.read_instance(str(temp_path))
                st.success(f"✅ Instance chargée : **{instance['name']}**")

            # Affichage des informations
            display_instance_info(instance)

            # Solution de référence
            reference_solution = None
            reference_cost = None

            if solution_file is not None:
                temp_sol_path = Path("temp_solution.sol")
                with open(temp_sol_path, 'wb') as f:
                    f.write(solution_file.getbuffer())

                try:
                    solution_data = vrplib.read_solution(str(temp_sol_path))
                    reference_solution = solution_data['routes']
                    reference_cost = solution_data.get('cost')

                    st.success(f"✅ Solution de référence chargée - Coût: **{reference_cost}**")
                except Exception as e:
                    st.warning(f"⚠️ Impossible de charger la solution: {e}")

            # Bouton de résolution
            if st.button("🚀 Lancer l'optimisation", type="primary", use_container_width=True):
                solve_vrp(instance, reference_solution, reference_cost)

        except Exception as e:
            st.error(f"❌ Erreur de chargement: {e}")


def generate_instance_mode():
    """Mode : Génération d'une instance personnalisée."""

    st.header("🎲 Générer une Instance VRP")

    col1, col2, col3 = st.columns(3)

    with col1:
        n_clients = st.number_input(
            "Nombre de clients",
            min_value=10,
            max_value=500,
            value=50,
            step=10,
            help="Nombre de clients à livrer"
        )

    with col2:
        capacity_min = st.number_input("Capacité min", min_value=30, max_value=100, value=50, step=10)
        capacity_max = st.number_input("Capacité max", min_value=50, max_value=200, value=150, step=10)

    with col3:
        demand_min = st.number_input("Demande min", min_value=1, max_value=20, value=5, step=1)
        demand_max = st.number_input("Demande max", min_value=10, max_value=50, value=30, step=5)

    col4, col5, col6 = st.columns(3)

    with col4:
        grid_size = st.slider("Taille de la grille", 50, 200, 100, 10)

    with col5:
        instance_type = st.selectbox("Type d'instance", ["Aléatoire", "Clusterisée"])
        n_clusters = st.number_input("Nombre de clusters", 2, 10, 3) if instance_type == "Clusterisée" else 0

    with col6:
        seed = st.number_input("Graine aléatoire (reproductibilité)", 0, 9999, 42, 1)

    if st.button("🎲 Générer l'instance", type="primary"):
        with st.spinner("Génération de l'instance..."):
            # Génération
            generator = VRPInstanceGenerator(
                n_clients=n_clients,
                depot_location=(grid_size // 2, grid_size // 2),
                grid_size=grid_size,
                seed=seed
            )

            if instance_type == "Aléatoire":
                instance = generator.generate_instance(
                    capacity_range=(capacity_min, capacity_max),
                    demand_range=(demand_min, demand_max)
                )
            else:
                instance = generator.generate_clustered_instance(
                    n_clusters=n_clusters,
                    capacity_range=(capacity_min, capacity_max),
                    demand_range=(demand_min, demand_max)
                )

            # Stockage dans session state
            st.session_state['generated_instance'] = instance

            st.success(f"✅ Instance générée : **{instance['name']}**")

            # Affichage
            display_instance_info(instance)

            # Téléchargement
            buffer = io.StringIO()
            temp_file = Path("temp_generated.vrp")
            save_instance_vrplib_format(instance, temp_file)

            with open(temp_file, 'r') as f:
                vrp_content = f.read()

            st.download_button(
                "💾 Télécharger l'instance (.vrp)",
                vrp_content,
                file_name=f"{instance['name']}.vrp",
                mime="text/plain"
            )

    # Résolution de l'instance générée
    if 'generated_instance' in st.session_state:
        if st.button("🚀 Lancer l'optimisation", type="primary", use_container_width=True):
            solve_vrp(st.session_state['generated_instance'], None, None)


def display_instance_info(instance):
    """Affiche les informations de l'instance."""

    st.subheader("📊 Informations de l'instance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Clients", instance['dimension'] - 1)

    with col2:
        st.metric("Capacité", instance['capacity'])

    with col3:
        total_demand = sum(instance['demand'][1:])
        st.metric("Demande totale", int(total_demand))

    with col4:
        min_vehicles = int(np.ceil(total_demand / instance['capacity']))
        st.metric("Véhicules min (théorique)", min_vehicles)

    # Visualisation des clients
    with st.expander("🗺️ Visualiser les clients"):
        fig, ax = plt.subplots(figsize=(10, 8))

        coords = instance['node_coord']
        depot_x, depot_y = coords[0]
        client_coords = coords[1:]

        # Dépôt
        ax.plot(depot_x, depot_y, 's', color='red', markersize=15, label='Dépôt', zorder=5)

        # Clients
        if len(client_coords) > 0:
            client_x = [c[0] for c in client_coords]
            client_y = [c[1] for c in client_coords]
            ax.scatter(client_x, client_y, c='blue', s=50, alpha=0.6, label='Clients')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Localisation des clients')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        st.pyplot(fig)
        plt.close()


def solve_vrp(instance, reference_solution, reference_cost):
    """Résout le VRP avec l'algorithme sélectionné."""

    st.divider()
    st.header("🔧 Résolution en cours...")

    # Récupération des paramètres depuis sidebar
    algorithm = st.session_state.get('algorithm', 'ALNS')
    time_limit = st.session_state.get('time_limit', 60)
    initial_method = st.session_state.get('initial_method', 'Clarke & Wright')

    # Conteneurs pour les résultats
    progress_placeholder = st.empty()
    metrics_placeholder = st.empty()

    try:
        # Étape 1 : Solution initiale
        progress_placeholder.info("🏗️ Génération de la solution initiale...")

        generator = InitialSolutionGenerator(instance)

        if initial_method == "Clarke & Wright":
            initial_solution = generator.clarke_wright()
        elif initial_method == "Plus Proche Voisin":
            initial_solution = generator.nearest_neighbor()
        else:
            initial_solution = generator.sequential_insertion()

        initial_cost = generator.calculate_cost(initial_solution)
        is_valid, msg = generator.verify_solution(initial_solution)

        if not is_valid:
            st.error(f"❌ Solution initiale invalide: {msg}")
            return

        st.success(f"✅ Solution initiale générée - Coût: **{initial_cost:.2f}** - {len(initial_solution)} routes")

        # Étape 2 : Optimisation
        progress_placeholder.info(f"🧠 Optimisation avec **{algorithm}**...")

        start_time = time.time()

        # Récupération des paramètres depuis session_state
        if algorithm == "ALNS":
            temperature = st.session_state.get('temperature', 100)
            cooling_rate = st.session_state.get('cooling_rate', 0.995)
            solver = ALNS(instance, time_limit=time_limit, temperature=temperature, cooling_rate=cooling_rate)
        elif algorithm == "Recuit Simulé":
            temperature = st.session_state.get('temperature', 1000)
            cooling_rate = st.session_state.get('cooling_rate', 0.995)
            solver = SimulatedAnnealing(instance, time_limit=time_limit, temperature=temperature, cooling_rate=cooling_rate)
        else:  # Recherche Tabou
            tabu_tenure = st.session_state.get('tabu_tenure', 20)
            max_iter_no_improve = st.session_state.get('max_iter_no_improve', 500)
            solver = TabuSearch(instance, time_limit=time_limit, tabu_tenure=tabu_tenure, max_iterations_without_improvement=max_iter_no_improve)

        # Résolution avec barre de progression
        with st.spinner(f"⏳ Résolution en cours (max {time_limit}s)..."):
            best_solution, best_cost, cost_history = solver.solve(initial_solution)

        elapsed_time = time.time() - start_time

        progress_placeholder.empty()

        # Affichage des résultats
        st.success(f"✅ Optimisation terminée en **{elapsed_time:.2f}s**")

        display_results(
            instance=instance,
            initial_solution=initial_solution,
            initial_cost=initial_cost,
            best_solution=best_solution,
            best_cost=best_cost,
            cost_history=cost_history,
            reference_solution=reference_solution,
            reference_cost=reference_cost,
            algorithm=algorithm,
            elapsed_time=elapsed_time
        )

    except Exception as e:
        st.error(f"❌ Erreur lors de la résolution: {e}")
        import traceback
        st.code(traceback.format_exc())


def display_results(instance, initial_solution, initial_cost, best_solution, best_cost,
                    cost_history, reference_solution, reference_cost, algorithm, elapsed_time):
    """Affiche les résultats de l'optimisation."""

    st.divider()
    st.header("📈 Résultats")

    # Métriques principales
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Coût initial", f"{initial_cost:.2f} km")

    with col2:
        improvement = ((initial_cost - best_cost) / initial_cost) * 100
        st.metric("Coût final", f"{best_cost:.2f} km", f"-{improvement:.2f}%")

    with col3:
        st.metric("Nombre de routes", len(best_solution))

    with col4:
        st.metric("Temps d'exécution", f"{elapsed_time:.2f}s")

    with col5:
        # Distance économisée
        distance_saved = initial_cost - best_cost
        st.metric("Distance économisée", f"{distance_saved:.2f} km", f"{improvement:.1f}%")

    # Calcul et affichage de l'impact environnemental (CO₂)
    st.divider()
    st.subheader("🌱 Impact Environnemental")

    # Facteur d'émission CO₂ : 0.18 kg CO₂/km (véhicule utilitaire léger diesel)
    co2_factor = 0.18

    co2_initial = initial_cost * co2_factor
    co2_optimized = best_cost * co2_factor
    co2_saved = co2_initial - co2_optimized
    co2_reduction_percent = (co2_saved / co2_initial) * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "CO₂ initial",
            f"{co2_initial:.2f} kg",
            help="Émissions de CO₂ pour la solution initiale (facteur: 0.18 kg CO₂/km)"
        )

    with col2:
        st.metric(
            "CO₂ optimisé",
            f"{co2_optimized:.2f} kg",
            delta=f"-{co2_reduction_percent:.1f}%",
            delta_color="inverse",
            help="Émissions de CO₂ pour la solution optimisée"
        )

    with col3:
        st.metric(
            "CO₂ économisé",
            f"{co2_saved:.2f} kg",
            help="Réduction d'émissions de CO₂ par tournée"
        )

    with col4:
        # Impact annuel (250 jours ouvrés)
        co2_annual = co2_saved * 250
        st.metric(
            "Impact annuel",
            f"{co2_annual/1000:.2f} tonnes",
            help="Économie de CO₂ sur 250 jours ouvrés (1 tournée/jour)"
        )

    # Mise en perspective
    st.info(f"""
    💡 **Mise en perspective :**
    - 🌳 Cette réduction équivaut à **{co2_saved/25:.1f} arbres** absorbant du CO₂ pendant un an
    - 🚗 Ou à **{co2_saved*1000/140:.0f} km** parcourus par une voiture particulière
    - 🏢 Pour une flotte de **10 véhicules** : **{co2_annual*10/1000:.1f} tonnes CO₂/an** économisées

    *(Facteur d'émission : 0.18 kg CO₂/km pour véhicule utilitaire léger diesel)*
    """)

    # Comparaison avec solution de référence
    if reference_cost is not None:
        st.subheader("🔍 Comparaison avec la solution optimale")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Solution de référence", f"{reference_cost:.2f}")

        with col2:
            gap = ((best_cost - reference_cost) / reference_cost) * 100
            st.metric("Gap", f"{gap:.2f}%", delta_color="inverse")

        with col3:
            quality = "Excellent" if gap < 3 else "Bon" if gap < 7 else "Acceptable" if gap < 10 else "À améliorer"
            st.metric("Qualité", quality)

    # Graphiques
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📉 Convergence",
        "🗺️ Solution initiale",
        "🗺️ Solution finale",
        "📊 NetworkX",
        "🌱 Impact CO₂"
    ])

    with tab1:
        plot_convergence(cost_history, initial_cost, reference_cost)

    with tab2:
        plot_solution_matplotlib(instance, initial_solution, "Solution Initiale", initial_cost)

    with tab3:
        plot_solution_matplotlib(instance, best_solution, "Solution Optimisée", best_cost)

    with tab4:
        plot_solution_networkx(instance, best_solution, "Solution Optimisée (NetworkX)", best_cost)

    with tab5:
        plot_co2_impact(initial_cost, best_cost, co2_factor)

    # Détails des routes
    with st.expander("📋 Détails des routes"):
        display_route_details(instance, best_solution)

    # Export
    st.subheader("💾 Export")

    col1, col2 = st.columns(2)

    with col1:
        # Export solution
        solution_text = format_solution_for_export(best_solution, best_cost)
        st.download_button(
            "📥 Télécharger la solution (.sol)",
            solution_text,
            file_name=f"solution_{algorithm}_{int(best_cost)}.sol",
            mime="text/plain"
        )

    with col2:
        # Export CSV
        csv_text = format_routes_to_csv(instance, best_solution)
        st.download_button(
            "📥 Télécharger les routes (.csv)",
            csv_text,
            file_name=f"routes_{algorithm}.csv",
            mime="text/csv"
        )


def plot_co2_impact(initial_cost, best_cost, co2_factor=0.18):
    """Affiche les graphiques d'impact environnemental CO₂."""

    # Calculs
    co2_initial = initial_cost * co2_factor
    co2_optimized = best_cost * co2_factor
    co2_saved = co2_initial - co2_optimized
    distance_saved = initial_cost - best_cost

    # Créer une grille de 2x2
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Impact Environnemental - Réduction des émissions CO₂',
                 fontsize=16, fontweight='bold', y=0.995)

    # Graphique 1 : Comparaison Distance
    ax1.bar(['Initial', 'Optimisé'], [initial_cost, best_cost],
            color=['#ff6b6b', '#51cf66'], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Distance (km)', fontsize=11, fontweight='bold')
    ax1.set_title('Distance Totale', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate([initial_cost, best_cost]):
        ax1.text(i, v + initial_cost*0.02, f'{v:.1f} km', ha='center', fontweight='bold')
    # Ajouter la réduction
    ax1.text(0.5, max(initial_cost, best_cost)*0.5,
             f'↓ {distance_saved:.1f} km\n({(distance_saved/initial_cost)*100:.1f}%)',
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
             fontweight='bold')

    # Graphique 2 : Comparaison CO₂
    ax2.bar(['Initial', 'Optimisé'], [co2_initial, co2_optimized],
            color=['#ff6b6b', '#51cf66'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Émissions CO₂ (kg)', fontsize=11, fontweight='bold')
    ax2.set_title('Émissions de CO₂ par Tournée', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for i, v in enumerate([co2_initial, co2_optimized]):
        ax2.text(i, v + co2_initial*0.02, f'{v:.1f} kg', ha='center', fontweight='bold')
    # Ajouter la réduction
    ax2.text(0.5, max(co2_initial, co2_optimized)*0.5,
             f'↓ {co2_saved:.1f} kg\n({(co2_saved/co2_initial)*100:.1f}%)',
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
             fontweight='bold')

    # Graphique 3 : Impact annuel
    days = [1, 50, 100, 150, 200, 250]
    co2_saved_cumulative = [co2_saved * d for d in days]
    ax3.plot(days, co2_saved_cumulative, marker='o', linewidth=2.5,
             color='#2ecc71', markersize=8)
    ax3.fill_between(days, co2_saved_cumulative, alpha=0.3, color='#2ecc71')
    ax3.set_xlabel('Jours ouvrés', fontsize=11, fontweight='bold')
    ax3.set_ylabel('CO₂ économisé cumulé (kg)', fontsize=11, fontweight='bold')
    ax3.set_title('Économies Cumulées sur l\'Année', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    # Ajouter annotation pour 250 jours
    ax3.annotate(f'{co2_saved*250:.0f} kg\n({co2_saved*250/1000:.2f} tonnes)',
                 xy=(250, co2_saved*250), xytext=(200, co2_saved*250*0.7),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2),
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Graphique 4 : Impact flotte (1 à 10 véhicules)
    n_vehicles = list(range(1, 11))
    co2_fleet_annual = [co2_saved * 250 * n / 1000 for n in n_vehicles]  # En tonnes
    bars = ax4.bar(n_vehicles, co2_fleet_annual, color='#3498db', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Nombre de véhicules', fontsize=11, fontweight='bold')
    ax4.set_ylabel('CO₂ économisé (tonnes/an)', fontsize=11, fontweight='bold')
    ax4.set_title('Impact Annuel selon Taille de Flotte', fontsize=12, fontweight='bold')
    ax4.set_xticks(n_vehicles)
    ax4.grid(axis='y', alpha=0.3)
    # Mettre en évidence 10 véhicules
    bars[-1].set_color('#e74c3c')
    bars[-1].set_alpha(0.9)
    ax4.text(10, co2_fleet_annual[-1] + max(co2_fleet_annual)*0.05,
             f'{co2_fleet_annual[-1]:.1f} t',
             ha='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Tableau récapitulatif
    st.markdown("### 📊 Tableau Récapitulatif")

    import pandas as pd

    recap_data = {
        'Métrique': [
            'Distance initiale',
            'Distance optimisée',
            'Distance économisée',
            'Réduction (%)',
            '',
            'CO₂ initial (par tournée)',
            'CO₂ optimisé (par tournée)',
            'CO₂ économisé (par tournée)',
            'Réduction CO₂ (%)',
            '',
            'Économie annuelle (250 jours)',
            'Économie flotte 10 véhicules/an',
            'Équivalent arbres (par tournée)',
            'Équivalent km voiture (par tournée)'
        ],
        'Valeur': [
            f'{initial_cost:.2f} km',
            f'{best_cost:.2f} km',
            f'{distance_saved:.2f} km',
            f'{(distance_saved/initial_cost)*100:.2f}%',
            '',
            f'{co2_initial:.2f} kg',
            f'{co2_optimized:.2f} kg',
            f'{co2_saved:.2f} kg',
            f'{(co2_saved/co2_initial)*100:.2f}%',
            '',
            f'{co2_saved*250/1000:.2f} tonnes',
            f'{co2_saved*250*10/1000:.2f} tonnes',
            f'{co2_saved/25:.1f} arbres',
            f'{co2_saved*1000/140:.0f} km'
        ]
    }

    df_recap = pd.DataFrame(recap_data)
    st.dataframe(df_recap, use_container_width=True, hide_index=True)


def plot_convergence(cost_history, initial_cost, reference_cost=None):
    """Affiche le graphique de convergence."""

    fig, ax = plt.subplots(figsize=(12, 6))

    iterations = list(range(len(cost_history)))
    ax.plot(iterations, cost_history, linewidth=2, label='Coût actuel', color='#1f77b4')
    ax.axhline(y=initial_cost, color='red', linestyle='--', alpha=0.7, label='Coût initial')

    if reference_cost is not None:
        ax.axhline(y=reference_cost, color='green', linestyle='--', alpha=0.7, label='Solution optimale')

    ax.set_xlabel('Itération', fontsize=12)
    ax.set_ylabel('Coût total', fontsize=12)
    ax.set_title('Convergence de l\'algorithme', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close()


def plot_solution_matplotlib(instance, solution, title, cost):
    """Visualise une solution avec Matplotlib."""

    fig, ax = plt.subplots(figsize=(12, 10))

    coords = instance['node_coord']
    depot_x, depot_y = coords[0]

    # Dépôt
    ax.plot(depot_x, depot_y, 's', color='red', markersize=15, label='Dépôt', zorder=5)

    # Clients
    for i in range(1, len(coords)):
        x, y = coords[i]
        ax.plot(x, y, 'o', color='lightblue', markersize=8, zorder=3)
        ax.text(x + 1, y + 1, str(i), fontsize=8, zorder=4)

    # Routes
    cmap = plt.cm.get_cmap('tab10')

    for route_idx, route in enumerate(solution):
        color = cmap(route_idx % 10)

        route_x = [coords[node][0] for node in route]
        route_y = [coords[node][1] for node in route]

        ax.plot(route_x, route_y, '-', color=color, linewidth=2, alpha=0.7,
                label=f'Route {route_idx + 1}')

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(f'{title} - Coût: {cost:.2f}', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    st.pyplot(fig)
    plt.close()


def plot_solution_networkx(instance, solution, title, cost):
    """Visualise une solution avec NetworkX."""

    st.subheader(title)

    # Création du graphe
    G = nx.DiGraph()

    coords = instance['node_coord']
    pos = {i: (coords[i][0], coords[i][1]) for i in range(len(coords))}

    # Ajout des nœuds
    G.add_node(0, label='Dépôt', node_type='depot')
    for i in range(1, len(coords)):
        G.add_node(i, label=f'C{i}', node_type='client', demand=instance['demand'][i])

    # Ajout des arêtes (routes)
    edge_colors = []
    cmap = plt.cm.get_cmap('tab10')

    for route_idx, route in enumerate(solution):
        color = cmap(route_idx % 10)
        for i in range(len(route) - 1):
            G.add_edge(route[i], route[i+1], route=route_idx, color=color)
            edge_colors.append(color)

    # Dessin
    fig, ax = plt.subplots(figsize=(14, 12))

    # Nœuds
    depot_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'depot']
    client_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'client']

    nx.draw_networkx_nodes(G, pos, nodelist=depot_nodes, node_color='red',
                          node_size=500, node_shape='s', ax=ax, label='Dépôt')
    nx.draw_networkx_nodes(G, pos, nodelist=client_nodes, node_color='lightblue',
                          node_size=200, ax=ax, label='Clients')

    # Arêtes
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2,
                          alpha=0.6, arrows=True, arrowsize=15, ax=ax)

    # Labels
    labels = {i: str(i) for i in range(len(coords))}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold', ax=ax)

    ax.set_title(f'{title} - Coût: {cost:.2f}\nGraphe orienté avec NetworkX',
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    ax.legend()

    st.pyplot(fig)
    plt.close()

    # Statistiques du graphe
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Nœuds", G.number_of_nodes())

    with col2:
        st.metric("Arêtes", G.number_of_edges())

    with col3:
        st.metric("Routes", len(solution))

    with col4:
        avg_route_length = np.mean([len(r) - 2 for r in solution])  # -2 pour enlever dépôts
        st.metric("Clients/route (moy)", f"{avg_route_length:.1f}")


def display_route_details(instance, solution):
    """Affiche les détails de chaque route."""

    for route_idx, route in enumerate(solution):
        st.markdown(f"**Route {route_idx + 1}**")

        # Calcul du coût et de la charge
        route_cost = sum(instance['edge_weight'][route[i]][route[i+1]]
                        for i in range(len(route) - 1))
        route_demand = sum(instance['demand'][c] for c in route if c != 0)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.text(f"Séquence: {' → '.join(map(str, route))}")

        with col2:
            st.text(f"Coût: {route_cost:.2f}")

        with col3:
            capacity_usage = (route_demand / instance['capacity']) * 100
            st.text(f"Charge: {route_demand}/{instance['capacity']} ({capacity_usage:.1f}%)")


def format_solution_for_export(solution, cost):
    """Formate la solution au format VRPLIB."""

    lines = [f"Cost {int(cost)}"]

    for route in solution:
        # Retirer les dépôts pour le format standard
        route_without_depot = [str(c) for c in route[1:-1]]
        lines.append("Route: " + " ".join(route_without_depot))

    return "\n".join(lines)


def format_routes_to_csv(instance, solution):
    """Formate les routes en CSV."""

    lines = ["Route,Sequence,Cost,Demand,Capacity_Usage"]

    for route_idx, route in enumerate(solution):
        sequence = " -> ".join(map(str, route))
        route_cost = sum(instance['edge_weight'][route[i]][route[i+1]]
                        for i in range(len(route) - 1))
        route_demand = sum(instance['demand'][c] for c in route if c != 0)
        capacity_usage = (route_demand / instance['capacity']) * 100

        lines.append(f"{route_idx + 1},{sequence},{route_cost:.2f},{route_demand},{capacity_usage:.1f}%")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
