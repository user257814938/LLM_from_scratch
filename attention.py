import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# MeriamAttention : un petit bloc d'attention (cerveau de l'IA)
#
# LEXIQUE POUR DÉBUTANT :
# - L'Attention : C'est ce qui permet à l'IA de comprendre quels
#   mots sont importants par rapport aux autres dans une phrase.
#   Par ex, dans "Le chat mange sa pâtée", l'attention relie "sa" à "chat".
# ============================================================

class MeriamAttention(nn.Module):
    def __init__(self, dim: int):
        """
        dim = taille des vecteurs (dimension du "cerveau").
        En informatique, un mot n'est pas vu comme du texte, 
        mais comme une liste de nombres (un vecteur). "dim" est 
        la quantité de nombres dans cette liste.
        """
        super().__init__()
        
        # On crée 3 couches linéaires : Q, K, V (Query, Key, Value)
        # L'attention utilise un système inspiré d'une bibliothèque :
        # - Q (Query / Question) : Ce que le mot actuel cherche à savoir.
        # - K (Key / Clé) : L'étiquette sur le livre (ce que les autres proposent).
        # - V (Value / Valeur) : Le contenu du livre (le sens réel du mot).
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : tenseur de forme [batch, longueur_phrase, dim]
        x : tenseur (un tableau d'entiers/nombres) de forme [batch, longueur_phrase, dim]
        - batch : Le nombre de phrases traitées en même temps (pour aller plus vite).
        - longueur_phrase : Le nombre de mots (ou caractères) de la phrase.
        - dim : La dimension détaillée du mot (mentionnée plus haut).
        
        retour : même forme, mais "enrichie" par l'attention
        """

        # 1) On projette x en Q, K, V
        Q = self.to_q(x)
        K = self.to_k(x)
        V = self.to_v(x)

        # 2) On calcule les scores d'attention : Q x K^T
        #    (on divise par sqrt(dim) pour stabiliser)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)

        # 3) On applique un softmax pour obtenir des poids entre 0 et 1
        # C'est une formule mathématique de "vote" qui transforme une liste de 
        # scores quelconques en pourcentages (des poids entre 0 et 1).
        # La somme de ces pourcentages fait toujours 100% (ou 1.0).
        attention = F.softmax(scores, dim=-1)

        # 4) On combine les valeurs V avec ces poids
        output = torch.matmul(attention, V)

        return output


# ----------------------------------------------
# Petit test rapide quand on lance attention.py
# ----------------------------------------------
if __name__ == "__main__":
    # On crée une "fausse phrase" :
    # batch = 1 phrase, longueur = 5 tokens, dimension = 16
    batch_size = 1
    seq_len = 5
    dim = 16

    x = torch.randn(batch_size, seq_len, dim)

    attn = MeriamAttention(dim)
    out = attn(x)

    print("Shape entrée  :", x.shape)
    print("Shape sortie  :", out.shape)


