# EF_Hermitte — Éléments finis de poutre d'Hermite (Jupyter Notebooks)

Exécution en ligne des notebooks via Binder, sans rien installer.
Deux versions sont disponibles, une par branche du dépôt :

## 1. Version standard — branche `main`

[![Binder main](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sylvain-drapier/EF_Hermitte/main?labpath=main_EF_Hermitte.ipynb)

Ouvre `main_EF_Hermitte.ipynb` (affichages matplotlib classiques).

## 2. Version avec affichages interactifs — branche `feat/affichage-interactif`

[![Binder affichage interactif](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sylvain-drapier/EF_Hermitte/feat/affichage-interactif?labpath=EF_Hermitte_Visualisation_interactive.ipynb)

Ouvre `EF_Hermitte_Visualisation_interactive.ipynb` : survol des valeurs, diagrammes N/T/M,
déformée d'Hermite, modes animés (module `affichage_interactif.py`, Plotly + ipywidgets).
Les autres notebooks (`main_EF_Hermitte.ipynb`, etc.) sont aussi disponibles dans cette version.

---

Le premier lancement d'une branche peut prendre quelques minutes (construction de l'image Binder).
