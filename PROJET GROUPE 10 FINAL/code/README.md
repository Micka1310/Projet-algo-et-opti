## Description

Ce projet implémente des solutions pour résoudre les problèmes de tournées de véhicules (VRP) en utilisant des heuristiques d'insertion séquentielle gloutonne. Le projet supporte deux types de problèmes :

- **CVRP** (Capacitated Vehicle Routing Problem) : Problème de tournées avec contrainte de capacité
- **VRPTW** (Vehicle Routing Problem with Time Windows) : Problème de tournées avec contraintes de capacité et fenêtres temporelles

## Fonctionnalités

- Lecture et parsing des fichiers d'instances VRP au format VRPLIB et Solomon
- Génération de solutions initiales avec heuristique d'insertion séquentielle gloutonne
- Calcul du coût total des solutions
- Calcul des émissions de CO2
- Visualisation graphique des solutions avec matplotlib
- Support des contraintes de capacité (CVRP)
- Support des fenêtres temporelles (VRPTW)

## Structure du projet

```
code/
├── test_cvrp_based_version.py          # Script principal pour CVRP
├── test_vrptw_advanced_version.py      # Script principal pour VRPTW
├── (les autres fichiers peuvent etre tester dans l'interface (le dossier interface(bonus)))
├── vrplib/                              # Bibliothèque vrplib
│   ├── __init__.py
│   ├── parse/                           # Modules de parsing
│   │   ├── parse_distances.py
│   │   ├── parse_solomon.py
│   │   ├── parse_solution.py
│   │   ├── parse_utils.py
│   │   └── parse_vrplib.py
│   ├── read/                            # Modules de lecture
│   │   ├── read_instance.py
│   │   └── read_solution.py
│   └── write/                           # Modules d'écriture
│       ├── write_instance.py
│       └── write_solution.py
└── tests/                               # Tests et données
    └── data/                            # Fichiers d'instances et solutions
```

## Installation

### Prérequis

- Python 3.8 ou supérieur

### Installation des dépendances

```bash
pip install -r requirements.txt
```

## Utilisation

### Résolution d'un problème CVRP

```bash
python test_cvrp_based_version.py
```

Ce script :
1. Lit une instance CVRP depuis `tests/data/B-n31-k5.vrp`
2. Génère une solution initiale avec l'heuristique gloutonne
3. Affiche les routes et le coût total
4. Calcule les émissions de CO2
5. Visualise la solution graphiquement
6. Compare avec la solution optimale

### Résolution d'un problème VRPTW

```bash
python test_vrptw_advanced_version.py
```

Ce script :
1. Lit une instance Solomon depuis `tests/data/cvrplib/Vrp-Set-Solomon/C201.txt`
2. Génère une solution initiale en respectant les fenêtres temporelles
3. Affiche les routes et le coût total
4. Calcule les émissions de CO2
5. Visualise la solution graphiquement
6. Compare avec la solution optimale

## Algorithmes

### Heuristique d'insertion séquentielle gloutonne

L'algorithme construit les routes de manière itérative :

1. **Initialisation** : Sélectionner le client le plus proche du dépôt
2. **Construction de route** :
   - Pour chaque client non visité, tester toutes les positions d'insertion possibles
   - Calculer le coût d'insertion : `coût(i → client) + coût(client → j) - coût(i → j)`
   - Insérer le client avec le meilleur coût si les contraintes sont respectées
3. **Contraintes** :
   - Capacité du véhicule (CVRP et VRPTW)
   - Fenêtres temporelles (VRPTW uniquement)
4. **Itération** : Créer une nouvelle route si nécessaire

## Calcul des émissions de CO2

Le projet calcule automatiquement les émissions de CO2 pour chaque solution :
- Taux d'émission : 900 g CO2/km
- Formule : `CO2 (kg) = Distance totale (km) × 0.9`

## Visualisation

Les solutions sont visualisées avec matplotlib :
- Dépôt en rouge (carré)
- Clients en bleu (cercles)
- Routes colorées distinctement
- Numérotation des sommets
- Affichage du coût total

## Format des données

### Format VRPLIB (CVRP)
```
NAME : B-n31-k5
COMMENT : Instance CVRP
TYPE : CVRP
DIMENSION : 31
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 100
NODE_COORD_SECTION
...
DEMAND_SECTION
...
DEPOT_SECTION
...
```

### Format Solomon (VRPTW)
```
C201

VEHICLE
NUMBER     CAPACITY
  25         200

CUSTOMER
CUST NO.  XCOORD.   YCOORD.   DEMAND   READY TIME  DUE DATE  SERVICE TIME
    0      40        50          0         0        236          0
...
```

## Auteurs

Projet Groupe 10

## Licence

Ce projet est développé dans un cadre éducatif.

## Contact

Pour toute question ou amélioration, veuillez contacter les membres du groupe.
