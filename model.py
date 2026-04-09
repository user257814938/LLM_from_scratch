import torch
import torch.nn as nn
from attention import MeriamAttention

# ============================================================
#  MeriamTransformer : Modèle de type GPT simplifié
#
# DÉFINITIONS TECHNIQUES :
# - Transformer : Architecture de réseau de neurones basée sur 
#   l'attention. Contrairement aux anciens réseaux, elle traite les
#   données en parallèle et capte mieux le contexte global.
# - Modèle GPT : Generative Pre-trained Transformer, un modèle dont le 
#   but strict est d'apprendre à prédire la suite d'une séquence de texte.
# ============================================================

class MeriamTransformer(nn.Module):
    def __init__(self, vocab_size, dim=32):
        super().__init__()

        # Embedding (Plongement) : Matrice qui convertit un nombre entier 
        # (l'identifiant du token) en un "vecteur dense" (une liste de nombres
        # de taille 'dim'). Cela positionne les mots dans un espace mathématique.
        self.embedding = nn.Embedding(vocab_size, dim)

        # Le bloc d’attention déjà créé
        # Attention : Mécanisme de pondération qui ajuste la valeur mathématique
        # de chaque token en observant tous les autres tokens de la séquence.
        self.attn = MeriamAttention(dim)

        # Couche finale : prédire le prochain caractère
        # "fc" = Fully Connected, son rôle est de faire le "bilan final". Le réseau a analysé la phrase, 
        # il a calculé les vecteurs dimensionnels (dim), 
        # il a croisé le contexte (grâce à l'Attention). 
        # À ce stade, la machine ne manipule encore que des vecteurs abstraits de taille dim. 
        # La couche Linéaire a pour but de traduire cette abstraction mathématique en une véritable réponse 
        # utilisable pour notre texte.

        self.fc = nn.Linear(dim, vocab_size)

    def forward(self, x):
        """
        Passe avant (Forward) : C'est le flux d'exécution. Il dicte l'ordre
        exact des transformations des données, de l'entrée vers la sortie.
        """
        # Entrée (x)  : Tenseur contenant les indices (entiers) des tokens
        # Transformation en vecteurs de taille 'dim'
        x = self.embedding(x)      
        
        # Croisement des informations contextuelles avec l'Attention
        x = self.attn(x)           
        
        # Le réseau produit des "Logits" : ce sont les scores bruts générés
        # par le modèle pour chaque caractère, avant d'être convertis en probabilités.
        x = self.fc(x)             
       
        return x


# ------------------------------------------------------------
# Test rapide
# ------------------------------------------------------------
if __name__ == "__main__":
    vocab_size = 50 # Taille du vocabulaire (nombre de tokens uniques)
    seq_len = 8 # Longueur de la séquence (nombre de tokens)

    # 1 exemple : phrase de 8 symboles
    x = torch.randint(0, vocab_size, (1, seq_len))

    model = MeriamTransformer(vocab_size)
    out = model(x)

    print("Shape entrée :", x.shape)
    print("Shape sortie :", out.shape)
