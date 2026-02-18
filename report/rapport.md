# 🐧 Rapport Technique : Architecture Big Data Distribuée & MLOps

**Projet :** Classification Distribuée des Manchots (Palmer Penguins) avec Architecture Multi-NoSQL

**Date :** 17 Février 2026

**Auteur :** Semenov Illia

**Stack Technique :** Docker, Apache Spark, MongoDB, Cassandra, Redis, FastAPI, Streamlit

**URL du Dashboard :** [http://91.134.143.137:8501/]()

---

## 📅 Table des Matières

1. **Introduction & Objectifs**
2. **Architecture du Système**
3. **Modélisation des Données (NoSQL)**

   3.1 MongoDB (Document Store)
   3.2 Cassandra (Column-Family Store)
   3.3 Redis (Cache Key-Value)

4. **Analyse Statistique & Régression**

   4.1 Statistiques Descriptives
   4.2 Modélisation par Régression Linéaire

5. **Pipeline Machine Learning (Spark MLlib)**

   5.1 Prétraitement et Feature Engineering
   5.2 Algorithme de Classification
   5.3 Inférence Hybride (Batch vs Temps-réel)

6. **Observabilité MLOps**

   6.1 Détection de Drift (Dérive de données)
   6.2 Monitoring de la Performance

7. **Benchmarks & Analyse de Performance**
8. **Conclusion & Perspectives de Scalabilité**

---

## 1. Introduction & Objectifs

Ce projet vise à concevoir une **architecture Big Data résiliente et scalable** capable de gérer le cycle de vie complet d'un modèle de Machine Learning : de l'ingestion de données brutes à l'inférence en temps réel.

L'objectif principal est de classifier les espèces de manchots de l'archipel Palmer (*Adelie, Chinstrap, Gentoo*) en utilisant leurs mesures biométriques. Au-delà de la simple classification, ce projet répond aux exigences d'analyse exploratoire et de modélisation prédictive (régression), tout en démontrant l'orchestration de plusieurs moteurs NoSQL pour satisfaire les **3 V du Big Data** :

* **Variété** : Gestion de données semi-structurées via MongoDB.
* **Volume** : Stockage distribué et partitionné via Cassandra.
* **Vélocité** : Inférence temps-réel (< 1ms) via Redis et FastAPI.

---

## 2. Architecture du Système

Le système repose sur une architecture conteneurisée (Docker) composée de quatre services interconnectés :

1. **Couche d'Ingestion (ETL Python) :**

   * Récupération du dataset brut (CSV).
   * Nettoyage (gestion des `NA`, normalisation des noms de colonnes).
   * Dispatch vers les bases de données (MongoDB pour le "Data Lake", Cassandra pour le "Data Warehouse").

2. **Couche de Traitement (Apache Spark 3.5 & Scikit-Learn) :**

   * Lecture distribuée depuis MongoDB.
   * Transformation des features (`VectorAssembler`, `StandardScaler`).
   * Entraînement du modèle de classification `RandomForestClassifier` (Spark).
   * Modélisation de régression linéaire interactive (Scikit-Learn).

3. **Couche de Service (API & Cache) :**

   * **Redis :** Cache "Hot Path" pour servir les prédictions déjà calculées instantanément.
   * **FastAPI :** Microservice exposant deux endpoints (Lookup et Inférence Custom). Il héberge un "Shadow Model" pour traiter les nouvelles données à la volée.

4. **Couche de Visualisation (Streamlit) :**

   * Dashboard interactif accessible publiquement sur **[http://91.134.143.137:8501/]()**. Il inclut le monitoring MLOps, le laboratoire de régression et les statistiques descriptives.

---

## 3. Modélisation des Données (NoSQL)

Le choix d'une architecture polyglotte (Multi-NoSQL) est justifié par le besoin d'optimiser chaque étape du traitement.

### 3.1 MongoDB (Document Store)

**Rôle :** Data Lake opérationnel & flexibilité de schéma.

Nous avons opté pour une modélisation **orientée document** avec imbrication (`embedding`). Contrairement à un modèle relationnel normalisé (3NF), nous regroupons toutes les mesures biométriques dans un sous-document `features`.

* **Structure JSON :**

```json
{
  "_id": "ObjectId(...)",
  "penguin_id": "P1024",
  "species": "Gentoo",
  "island": "Biscoe",
  "features": {  // Encapsulation pour lecture atomique
    "bill_length": 45.2,
    "bill_depth": 14.8,
    "flipper_length": 212,
    "body_mass": 5200
  },
  "sex": "FEMALE",
  "year": 2020
}
```

* **Justification Scalabilité :** Ce modèle permet d'ajouter de nouvelles features (ex: `gps_coordinates` ou `blood_sample`) sans casser le schéma existant ni nécessiter de migration lourde (Schema Evolution).

### 3.2 Cassandra (Column-Family Store)

**Rôle :** Stockage analytique haute disponibilité & "Source of Truth".

Cassandra est optimisé pour les écritures massives et la tolérance aux pannes. La modélisation repose sur les requêtes (Query-Driven Modeling).

* **Schéma CQL :**

```sql
CREATE TABLE penguin_ks.penguins_by_island (
    island text,          -- PARTITION KEY
    species text,         -- CLUSTERING KEY 1
    penguin_id text,      -- CLUSTERING KEY 2
    bill_length float,
    body_mass int,
    prediction double,
    PRIMARY KEY ((island), species, penguin_id)
);
```

* **Stratégie de Partitionnement (`island`) :**
  * Le choix de l'île comme **Partition Key** garantit que toutes les données d'une zone géographique sont stockées sur les mêmes nœuds physiques. Cela permet des requêtes d'agrégation géographique extrêmement rapides.
  * La **Clustering Key** (`species`, `penguin_id`) trie les données sur le disque, optimisant les recherches par plage (Range Scans).

### 3.3 Redis (Key-Value Store)

**Rôle :** Cache de prédiction temps-réel (Low Latency).

* **Structure :** Clé = `penguin_id`, Valeur = `prediction_float`.
* **TTL (Time-To-Live) :** Configuré à 1 heure (3600s) pour garantir la fraîcheur des données tout en déchargeant les bases persistantes.

---

## 4. Analyse Statistique & Régression

Pour répondre aux besoins d'analyse exploratoire, une section dédiée a été intégrée au Dashboard.

### 4.1 Statistiques Descriptives

Le système calcule automatiquement les indicateurs clés (Moyenne, Médiane, Écart-type, Min/Max) pour l'ensemble des variables numériques (`bill_length`, `body_mass`, etc.).

* Visualisation de la répartition par Île et par Sexe.
* Scatter Plots spécifiques : *Longueur vs Profondeur du bec* (par Espèce) et *Longueur nageoire vs Masse* (par Sexe).

### 4.2 Modélisation par Régression Linéaire

Un laboratoire interactif permet d'exécuter des régressions (Simples et Multiples) pour prédire la masse corporelle (`body_mass_g`).

* **Méthodologie :** Séparation Train/Test (80/20), entraînement d'un modèle OLS.
* **Métriques :** Affichage du coefficient  et de l'erreur quadratique moyenne (MSE).
* **Résultats :** L'analyse des coefficients confirme que la `flipper_length` est la variable la plus corrélée positivement à la masse corporelle.

---

## 5. Pipeline Machine Learning (Spark MLlib)

Le cœur du traitement de classification est un job Spark distribué (`spark_ml.py`).

### 5.1 Prétraitement et Feature Engineering

L'étape critique identifiée a été la **mise à l'échelle des données**.

* **Problème :** La variable `body_mass` (~4000 g) a une magnitude 200x supérieure à `bill_depth` (~18 mm).
* **Solution :** Application d'un `StandardScaler` (Z-Score normalization) dans le pipeline Spark pour ramener toutes les features à une moyenne de 0 et un écart-type de 1.

### 5.2 Algorithme de Classification

* **Modèle :** `RandomForestClassifier`
* **Hyperparamètres :** `numTrees=20` (stabilisation de la variance), `maxDepth=5` (limitation du sur-apprentissage).
* **Résultats :** Le modèle atteint une précision élevée (>95%), distinguant clairement les *Gentoo* (massifs) des autres espèces.

### 5.3 Inférence Hybride

Pour une application pleinement fonctionnelle, deux modes d'inférence coexistent :

1. **Batch Inference (Spark) :** Calcul de nuit sur l'historique, résultats poussés dans Cassandra.
2. **Real-time Inference (API) :** Un modèle "Shadow" (Scikit-Learn) entraîné au démarrage de l'API permet de prédire l'espèce d'un manchot *inconnu* en < 50ms.

---

## 6. Observabilité MLOps

Un système Big Data en production doit être surveillé.

### 6.1 Détection de Drift (Dérive de données)

Les modèles ML se dégradent quand les données réelles changent.

* **Méthode :** Test de Kolmogorov-Smirnov (KS Test).
* **Implémentation :** Comparaison de la distribution de `body_mass` en base (Training) vs flux entrants (Production simulée).
* **Alerte :** Si `P-Value < 0.05`, le Dashboard affiche "🔴 DRIFT DETECTED".

### 6.2 Monitoring de la Performance

Visualisation en temps réel de la **Matrice de Confusion**, comparant les étiquettes réelles (`label`) stockées dans MongoDB aux prédictions (`prediction`).

---

## 7. Benchmarks & Analyse de Performance

Latence moyenne de lecture mesurée sur 1 000 itérations séquentielles :

| Technologie | Type | Latence Moyenne | Throughput Estimé | Cas d'usage idéal |
| --- | --- | --- | --- | --- |
| **Redis** | In-Memory | **0.18 ms** | ~5,600 req/s | Cache API, Session utilisateur |
| **MongoDB** | Document | **0.35 ms** | ~2,800 req/s | Backend Web, Profils utilisateurs |
| **Cassandra** | Columnar | **1.52 ms** | ~650 req/s | Historique, Time-Series, IoT |

**Analyse :** Redis domine grâce à l'absence d'I/O disque, essentiel pour la couche "Vélocité". Cassandra, bien que plus lent en lecture unitaire, offre une scalabilité linéaire en écriture pour les volumes massifs.

---

## 8. Conclusion & Perspectives de Scalabilité

Ce projet valide une architecture Lambda complète, allant de l'ingestion à l'analyse prédictive.

**Points Forts :**

1. **Couverture Complète :** Intègre Statistiques Descriptives, Régression, et Classification.
2. **Résilience :** Architecture découplée (l'API survit si Spark tombe).
3. **Automatisation :** Déploiement `docker-compose` et interface "No-Code" (Streamlit).
