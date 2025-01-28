# AI Classification Image 

## Description :
Ce projet est une application de classification d'images basée sur l'IA. Le site web permet de prédire la classe d'une image uploadée par l'utilisateur. L'application utilise Flask pour le backend.

## Installation :

Pour exécuter le site en local, suivez les étapes ci-dessous :

1. Installez Flask dans l'environnement virtuel du projet en exécutant la commande suivante dans le terminal à la racine du projet :
   ```bash
   pip install flask

2. Lancer la commande pour ouvrir le serveur flask a l'adresse http://127.0.0.1:5000/ en etant toujours a la racine du projet:
   ```bash
   python app.py

## Remarque :

Les images contenues dans le dossier data ne sont pas chargées dans ce projet en raison de leur grand nombre.
Étant donné que l'accuracy du modèle est relativement basse, il est possible que le site affiche fréquemment la même classe, quelle que soit l'image soumise.
