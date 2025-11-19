"""
Module de visualisation avancée pour les solutions VRP.
Génère des graphes NetworkX professionnels pour la présentation.

Ce module fournit des fonctions de visualisation de haute qualité pour :
- Afficher les instances VRP (dépôt + clients)
- Visualiser les solutions avec routes colorées
- Créer des graphes orientés avec NetworkX
- Générer des visualisations comparatives
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.collections import LineCollection


def create_vrp_graph(instance, solution, title="Solution VRP", show_edge_labels=True):
    """
    Crée un graphe NetworkX professionnel pour visualiser une solution VRP.

    Cette fonction génère un graphe orienté (DiGraph) représentant la solution
    d'un problème de tournées de véhicules. Chaque route est colorée différemment
    et les arêtes sont orientées pour montrer le sens de parcours.

    Args:
        instance (dict): Instance VRP contenant :
            - 'node_coord': Coordonnées des nœuds (numpy array)
            - 'demand': Demandes de chaque client (numpy array)
            - 'capacity': Capacité des véhicules (int)
            - 'edge_weight': Matrice des distances (numpy array)

        solution (list): Liste des routes, chaque route est une liste d'indices
                        Exemple: [[0, 1, 3, 0], [0, 2, 4, 5, 0]]

        title (str): Titre du graphe

        show_edge_labels (bool): Afficher les distances sur les arêtes

    Returns:
        tuple: (figure, axes, graph) - Figure matplotlib, axes et graphe NetworkX

    Exemple:
        >>> instance = {...}  # Instance VRP
        >>> solution = [[0, 1, 2, 0], [0, 3, 4, 0]]
        >>> fig, ax, G = create_vrp_graph(instance, solution, "Ma Solution")
        >>> plt.show()
    """
    # Création du graphe orienté
    G = nx.DiGraph()

    # Extraction des données
    coords = instance['node_coord']
    demands = instance['demand']
    capacity = instance['capacity']
    distances = instance['edge_weight']

    # Position des nœuds (pour le layout)
    pos = {i: (coords[i][0], coords[i][1]) for i in range(len(coords))}

    # Ajout des nœuds avec attributs
    G.add_node(0, label='Dépôt', node_type='depot', demand=0, pos=pos[0])

    for i in range(1, len(coords)):
        G.add_node(i,
                  label=f'C{i}',
                  node_type='client',
                  demand=demands[i],
                  pos=pos[i])

    # Palette de couleurs pour les routes
    colors = plt.cm.tab20(np.linspace(0, 1, len(solution)))

    # Ajout des arêtes par route avec couleurs
    edge_colors = {}
    edge_routes = {}

    for route_idx, route in enumerate(solution):
        route_color = colors[route_idx]

        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]

            # Distance de l'arête
            distance = distances[u][v]

            # Ajout de l'arête
            G.add_edge(u, v,
                      route=route_idx + 1,
                      distance=distance,
                      color=route_color)

            edge_colors[(u, v)] = route_color
            edge_routes[(u, v)] = route_idx + 1

    # Création de la figure
    fig, ax = plt.subplots(figsize=(16, 12))

    # Séparation des nœuds par type
    depot_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'depot']
    client_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'client']

    # Dessin des nœuds
    # Dépôt - Grande étoile rouge
    nx.draw_networkx_nodes(G, pos,
                          nodelist=depot_nodes,
                          node_color='darkred',
                          node_size=1000,
                          node_shape='s',
                          edgecolors='black',
                          linewidths=3,
                          ax=ax,
                          label='Dépôt')

    # Clients - Cercles bleus avec bordure
    nx.draw_networkx_nodes(G, pos,
                          nodelist=client_nodes,
                          node_color='lightblue',
                          node_size=400,
                          node_shape='o',
                          edgecolors='darkblue',
                          linewidths=2,
                          ax=ax,
                          label='Clients')

    # Dessin des arêtes avec couleurs par route
    for route_idx, route in enumerate(solution):
        edge_list = [(route[i], route[i+1]) for i in range(len(route) - 1)]

        nx.draw_networkx_edges(G, pos,
                              edgelist=edge_list,
                              edge_color=[edge_colors[e] for e in edge_list],
                              width=3,
                              alpha=0.7,
                              arrows=True,
                              arrowsize=20,
                              arrowstyle='-|>',
                              connectionstyle='arc3,rad=0.1',
                              ax=ax,
                              label=f'Route {route_idx + 1}')

    # Labels des nœuds
    node_labels = {}
    for node in G.nodes():
        if G.nodes[node]['node_type'] == 'depot':
            node_labels[node] = 'Dépôt\n0'
        else:
            demand = G.nodes[node]['demand']
            node_labels[node] = f'{node}\n({demand})'

    nx.draw_networkx_labels(G, pos,
                           node_labels,
                           font_size=9,
                           font_weight='bold',
                           font_color='black',
                           ax=ax)

    # Labels des arêtes (distances)
    if show_edge_labels:
        edge_labels = {}
        for u, v in G.edges():
            distance = G[u][v]['distance']
            edge_labels[(u, v)] = f'{distance:.0f}'

        nx.draw_networkx_edge_labels(G, pos,
                                     edge_labels,
                                     font_size=7,
                                     font_color='darkred',
                                     bbox=dict(boxstyle='round,pad=0.3',
                                             facecolor='white',
                                             edgecolor='gray',
                                             alpha=0.8),
                                     ax=ax)

    # Calcul des statistiques
    total_cost = sum(distances[route[i]][route[i+1]]
                    for route in solution
                    for i in range(len(route) - 1))

    # Titre et informations
    ax.set_title(f'{title}\n' +
                f'Coût total: {total_cost:.2f} | ' +
                f'Routes: {len(solution)} | ' +
                f'Clients: {len(coords) - 1}',
                fontsize=16,
                fontweight='bold',
                pad=20)

    ax.set_xlabel('Coordonnée X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coordonnée Y', fontsize=12, fontweight='bold')

    # Légende personnalisée
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(solution)+2], labels[:len(solution)+2],
             loc='upper left',
             bbox_to_anchor=(1.02, 1),
             fontsize=10,
             frameon=True,
             shadow=True)

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')

    plt.tight_layout()

    return fig, ax, G


def create_comparison_graphs(instance, initial_solution, final_solution,
                             initial_cost, final_cost, algorithm_name):
    """
    Crée une visualisation comparative avant/après optimisation.

    Cette fonction génère deux graphes côte à côte pour comparer visuellement
    la solution initiale et la solution optimisée, avec leurs statistiques respectives.

    Args:
        instance (dict): Instance VRP
        initial_solution (list): Solution initiale (routes)
        final_solution (list): Solution optimisée (routes)
        initial_cost (float): Coût de la solution initiale
        final_cost (float): Coût de la solution finale
        algorithm_name (str): Nom de l'algorithme utilisé

    Returns:
        tuple: (figure, axes) - Figure matplotlib avec 2 sous-graphiques

    Exemple:
        >>> fig, axes = create_comparison_graphs(
        ...     instance, sol_init, sol_final, 1000, 850, "ALNS"
        ... )
        >>> plt.show()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    # Graphe 1 : Solution initiale
    G1 = nx.DiGraph()
    coords = instance['node_coord']
    distances = instance['edge_weight']
    pos = {i: (coords[i][0], coords[i][1]) for i in range(len(coords))}

    # Construction graphe initial
    for i in range(len(coords)):
        G1.add_node(i, pos=pos[i])

    colors1 = plt.cm.Set3(np.linspace(0, 1, len(initial_solution)))

    for route_idx, route in enumerate(initial_solution):
        for i in range(len(route) - 1):
            G1.add_edge(route[i], route[i+1],
                       color=colors1[route_idx],
                       route=route_idx)

    # Dessin graphe initial
    depot = [0]
    clients = list(range(1, len(coords)))

    nx.draw_networkx_nodes(G1, pos, nodelist=depot,
                          node_color='red', node_size=800,
                          node_shape='s', ax=ax1)
    nx.draw_networkx_nodes(G1, pos, nodelist=clients,
                          node_color='lightblue', node_size=300,
                          ax=ax1)

    for route_idx, route in enumerate(initial_solution):
        edges = [(route[i], route[i+1]) for i in range(len(route)-1)]
        nx.draw_networkx_edges(G1, pos, edgelist=edges,
                              edge_color=[colors1[route_idx]]*len(edges),
                              width=2.5, alpha=0.7, arrows=True,
                              arrowsize=15, ax=ax1)

    labels = {i: str(i) for i in range(len(coords))}
    nx.draw_networkx_labels(G1, pos, labels, font_size=10,
                           font_weight='bold', ax=ax1)

    improvement = ((initial_cost - final_cost) / initial_cost) * 100

    ax1.set_title(f'Solution Initiale\n' +
                 f'Coût: {initial_cost:.2f} | Routes: {len(initial_solution)}',
                 fontsize=14, fontweight='bold', color='darkblue')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Graphe 2 : Solution optimisée
    G2 = nx.DiGraph()

    for i in range(len(coords)):
        G2.add_node(i, pos=pos[i])

    colors2 = plt.cm.tab20(np.linspace(0, 1, len(final_solution)))

    for route_idx, route in enumerate(final_solution):
        for i in range(len(route) - 1):
            G2.add_edge(route[i], route[i+1],
                       color=colors2[route_idx],
                       route=route_idx)

    # Dessin graphe final
    nx.draw_networkx_nodes(G2, pos, nodelist=depot,
                          node_color='darkgreen', node_size=800,
                          node_shape='s', ax=ax2)
    nx.draw_networkx_nodes(G2, pos, nodelist=clients,
                          node_color='lightgreen', node_size=300,
                          ax=ax2)

    for route_idx, route in enumerate(final_solution):
        edges = [(route[i], route[i+1]) for i in range(len(route)-1)]
        nx.draw_networkx_edges(G2, pos, edgelist=edges,
                              edge_color=[colors2[route_idx]]*len(edges),
                              width=2.5, alpha=0.7, arrows=True,
                              arrowsize=15, ax=ax2)

    nx.draw_networkx_labels(G2, pos, labels, font_size=10,
                           font_weight='bold', ax=ax2)

    ax2.set_title(f'Solution Optimisée ({algorithm_name})\n' +
                 f'Coût: {final_cost:.2f} | Routes: {len(final_solution)} | ' +
                 f'Amélioration: {improvement:.2f}%',
                 fontsize=14, fontweight='bold', color='darkgreen')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    plt.tight_layout()

    return fig, (ax1, ax2)


def visualize_graph_statistics(graph, solution, instance):
    """
    Génère des statistiques graphiques sur le graphe de solution.

    Cette fonction analyse le graphe NetworkX et génère des visualisations
    statistiques : distribution des degrés, centralité, etc.

    Args:
        graph (nx.DiGraph): Graphe NetworkX de la solution
        solution (list): Liste des routes
        instance (dict): Instance VRP

    Returns:
        figure: Figure matplotlib avec statistiques

    Exemple:
        >>> G = nx.DiGraph()
        >>> # ... construction du graphe
        >>> fig = visualize_graph_statistics(G, solution, instance)
        >>> plt.show()
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Distribution des degrés
    ax1 = axes[0, 0]
    degrees = dict(graph.degree())
    degree_values = list(degrees.values())

    ax1.hist(degree_values, bins=range(max(degree_values)+2),
            color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Degré', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Nombre de nœuds', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution des degrés des nœuds', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. Longueur des routes
    ax2 = axes[0, 1]
    route_lengths = [len(route) - 2 for route in solution]  # -2 pour enlever dépôts
    route_labels = [f'Route {i+1}' for i in range(len(solution))]

    colors_bar = plt.cm.viridis(np.linspace(0, 1, len(solution)))
    ax2.bar(route_labels, route_lengths, color=colors_bar, edgecolor='black', alpha=0.8)
    ax2.set_xlabel('Routes', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Nombre de clients', fontsize=11, fontweight='bold')
    ax2.set_title('Clients par route', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Charge des véhicules
    ax3 = axes[1, 0]
    demands = instance['demand']
    capacity = instance['capacity']

    route_demands = []
    for route in solution:
        demand = sum(demands[c] for c in route if c != 0)
        route_demands.append(demand)

    capacity_usage = [(d/capacity)*100 for d in route_demands]

    ax3.barh(route_labels, capacity_usage, color=colors_bar, edgecolor='black', alpha=0.8)
    ax3.axvline(x=100, color='red', linestyle='--', linewidth=2, label='Capacité max')
    ax3.set_xlabel('Utilisation de la capacité (%)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Routes', fontsize=11, fontweight='bold')
    ax3.set_title('Taux d\'utilisation des véhicules', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='x')

    # 4. Coût des routes
    ax4 = axes[1, 1]
    distances = instance['edge_weight']

    route_costs = []
    for route in solution:
        cost = sum(distances[route[i]][route[i+1]] for i in range(len(route)-1))
        route_costs.append(cost)

    ax4.bar(route_labels, route_costs, color=colors_bar, edgecolor='black', alpha=0.8)
    ax4.set_xlabel('Routes', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Coût', fontsize=11, fontweight='bold')
    ax4.set_title('Coût par route', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')

    # Statistiques globales
    total_cost = sum(route_costs)
    avg_route_length = np.mean(route_lengths)
    avg_capacity_usage = np.mean(capacity_usage)

    fig.suptitle(f'Statistiques du Graphe\n' +
                f'Coût total: {total_cost:.2f} | ' +
                f'Longueur moyenne: {avg_route_length:.1f} clients/route | ' +
                f'Capacité moyenne: {avg_capacity_usage:.1f}%',
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def create_instance_preview_graph(instance, title="Instance VRP"):
    """
    Crée un graphe de prévisualisation d'une instance VRP (avant résolution).

    Affiche uniquement les nœuds (dépôt et clients) sans les routes.
    Utile pour visualiser la disposition géographique des clients.

    Args:
        instance (dict): Instance VRP
        title (str): Titre du graphe

    Returns:
        tuple: (figure, axes, graph)

    Exemple:
        >>> instance = generator.generate_instance(n_clients=30)
        >>> fig, ax, G = create_instance_preview_graph(instance)
        >>> plt.show()
    """
    G = nx.Graph()

    coords = instance['node_coord']
    demands = instance['demand']

    pos = {i: (coords[i][0], coords[i][1]) for i in range(len(coords))}

    # Ajout des nœuds
    for i in range(len(coords)):
        node_type = 'depot' if i == 0 else 'client'
        G.add_node(i,
                  pos=pos[i],
                  type=node_type,
                  demand=demands[i])

    # Création figure
    fig, ax = plt.subplots(figsize=(12, 10))

    depot = [0]
    clients = list(range(1, len(coords)))

    # Dessin
    nx.draw_networkx_nodes(G, pos,
                          nodelist=depot,
                          node_color='darkred',
                          node_size=1000,
                          node_shape='s',
                          edgecolors='black',
                          linewidths=3,
                          ax=ax,
                          label='Dépôt')

    # Coloration des clients selon la demande
    demands_clients = [demands[i] for i in clients]
    nx.draw_networkx_nodes(G, pos,
                          nodelist=clients,
                          node_color=demands_clients,
                          cmap='YlOrRd',
                          node_size=300,
                          edgecolors='black',
                          linewidths=1.5,
                          ax=ax,
                          label='Clients')

    # Labels
    labels = {i: f'{i}\n({demands[i]})' if i > 0 else 'Dépôt\n0'
             for i in range(len(coords))}
    nx.draw_networkx_labels(G, pos, labels,
                           font_size=9,
                           font_weight='bold',
                           ax=ax)

    # Colorbar pour les demandes
    sm = plt.cm.ScalarMappable(cmap='YlOrRd',
                               norm=plt.Normalize(vmin=min(demands_clients),
                                                 vmax=max(demands_clients)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Demande client', fontsize=11, fontweight='bold')

    ax.set_title(f'{title}\n' +
                f'Clients: {len(coords)-1} | ' +
                f'Capacité: {instance["capacity"]} | ' +
                f'Demande totale: {sum(demands):.0f}',
                fontsize=14,
                fontweight='bold',
                pad=15)

    ax.set_xlabel('Coordonnée X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coordonnée Y', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')

    plt.tight_layout()

    return fig, ax, G
