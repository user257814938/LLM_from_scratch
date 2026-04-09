# LLM From Scratch - Cœur Pédagogique

## Architecture Workflow
![Architecture Workflow](architecure workflow.png)

Ce projet illustre de façon minimale, mais fonctionnelle, le mécanisme interne d'un grand modèle de langage (Large Language Model) de type Transformer. 

Cette documentation se concentre exclusivement sur les trois fichiers fondateurs de l'architecture, codés de zéro, qui dictent le comportement mathématique de l'IA.

---

## 1. La Préparation Textuelle (`tokenizer.py`)
Un réseau de neurones ne peut ni lire ni comprendre l'alphabet humain. La classe `MeriamTokenizer` a pour rôle exclusif de faire l'intermédiaire de traduction numérique.

- **`build_vocab()`** : Parcourt un texte d'entraînement pour identifier tous les symboles/caractères uniques, et attribue à chacun un numéro d'identification exclusif.
- **`encode()`** : L'action de prendre du texte envoyé par un humain et de remplacer chaque caractère par son numéro respectif. Cette liste d'entiers sera la donnée d'entrée du modèle.
- **`decode()`** : L'action inverse permettant de retraduire les numéros générés par l'IA en texte lisible pour l'utilisateur.

---

## 2. L'Analyse du Contexte (`attention.py`)
Ce fichier contient la classe `MeriamAttention`. C'est le cœur cognitif du Transformer moderne : plutôt que de lire les mots un à un successivement de gauche à droite, ce mécanisme permet de superposer tout le contenu d'une phrase pour en tirer le sens global.

- **Projections Q, K, V** : Le modèle multiplie chaque token d'entrée par trois matrices pour générer une Requête (Q), une Clé (K) et une Valeur sémantique (V).
- **Scores \& Softmax** : Le modèle calcule la compatibilité entre les mots en multipliant les Requêtes (Q) aux Clés (K). La fonction **Softmax** convertit ensuite le résultat algébrique en un pourcentage strict compris entre 0% et 100%. 
- **L'Attention finale** : Les pourcentages obtenus agissent comme des coefficients multiplicateurs sur les Valeurs (V). Le résultat est un nouveau vecteur, désormais enrichi de tout le contexte de la phrase (par exemple, liant grammaticalement un pronom au sujet lointain auquel il se rapporte).

---

## 3. L'Orchestration du Modèle (`model.py`)
Ce fichier assemble les briques dans la classe `MeriamTransformer`. Il définit la route obligatoire que va emprunter la donnée depuis l'entrée vers la prédiction finale de la machine.

- **`nn.Embedding` (Plongement)** : Couche initiale qui convertit l'identifiant (le numéro d'un mot) issu du *tokenizer* en un vecteur composé de plusieurs dimensions mathématiques (de taille `dim`).
- **Passe Avant (`forward()`)** : Définit le flux strict des opérations algorithmiques. Le tenseur passe à travers l'Embedding, puis transite dans le bloc d'Attention pour le croisement contextuel.
- **`nn.Linear` (Couche Linéaire)** : C'est l'entonnoir inverse final. Il multiplie le tenseur obtenu par la précédente étape avec une matrice spécifique pour forcer sa taille finale à correspondre exactement au nombre de caractères de notre *vocabulaire*.
- **Génération des Logits** : Le réseau recrache les *Logits*. Il s'agit des scores de prédiction bruts, caractère par caractère, indiquant lequel la machine détermine mathématiquement comme étant logiquement le suivant.

