# Labo 3 

## Introduction
Dans ce laboratoire, nous allons appliquer et analyser un ensemble de modélisations data-driven sur un jeu de données de classification.
Dans un cadre plus général, le but de ce laboratoire est de pouvoir classifier l'état de sommeil d'une souris.

Nous avons à dispsition deux jeux de données avec le résultat souhaité: EEG_mouse_data_1.csv et EEG_mouse_data_2.csv ainsi qu'un jeu 
de données sans le résultat recherché: EEG_mouse_data_test.csv.

//TODO parler plus des données et ce qu'on utilise

Nos données d'entrainement seront utilisées pour entrainer nos modèles ainsi que faire de la validation croisée.
De plus, nous allons faire une matrice de confusion pour chaque modèle ainsi que leur F-score respectif.
## Partie 1

Dans cette première partie, le seul but est de pouvoir reconnaître l'état d'une souris dormant ou éveillée.

### Choix du modèle

En ce qui concerne le modèle, nous avons utilisé la fonction d'activation **tahn**. 
Pour cela, les données ont été normalisées avec -1 pour les valeurs "dorment" (rem/non-rem) et 1 pour les valeurs "éveillées"(awake).

Pour le nombre de neurones nous avons essayé différentes configurations avec une couche.
Le nombre de neurones essayés ont été, 2,4,8, 20, 30. Nous avons remarqué une augmentation d'overfitting à partir de plus de **8 neurones** et avons décidé de s'arrêter à 8.

Concernant le nombre d'épochs, la plupart des tests ont été faits avec 20 epochs afin de pouvoir supposer sur la direction que prenait le modèle.
Nous avons cependant remarqué qu'un nombre d'epochs plus grand continuait à apporter des meilleurs résultats mais à un rythme logarithmique.
C'est pour cela que nous avons décidé de nous arrêter à **100 epochs**.

La loss function utilisée est  **mse**, l'optimizer est **SGD**. Les paramètres pour l'optimizer sont les suivants:

- learning rate: 0.001
- momentum: 0.99

Nous avons essayé avec d'autres paramètres mais ceux-ci ont donné les meilleurs résultats.

### Résultats

Nous avons obtenu un mean f-score de 0.88663 et la matrice de confusion suivante:

![Matrice de confusion](./figures/p1_matrix.png)

On peut remarquer qu'il y a encore de l'overfitting mais le modèle est capable de bien classifier les données de tests.

Voici le plot de la loss function:

![Loss function](./figures/p1_history.png)


