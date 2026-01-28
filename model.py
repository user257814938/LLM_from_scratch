import torch
import torch.nn as nn
from attention import MeriamAttention

# ============================================================
#  MeriamTransformer : un mini modèle GPT simplifié
# ============================================================

class MeriamTransformer(nn.Module):
    def __init__(self, vocab_size, dim=32):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dim)

        # Ton bloc d’attention déjà créé
        self.attn = MeriamAttention(dim)

        # Couche finale : prédire le prochain caractère
        self.fc = nn.Linear(dim, vocab_size)

    def forward(self, x):
        # x : [batch, longueur_phrase]
        x = self.embedding(x)      # -> [batch, long, dim]
        x = self.attn(x)           # -> attention
        x = self.fc(x)             # -> logits prédiction
        return x


# ------------------------------------------------------------
# Test rapide
# ------------------------------------------------------------
if __name__ == "__main__":
    vocab_size = 50
    seq_len = 8

    # 1 exemple : phrase de 8 symboles
    x = torch.randint(0, vocab_size, (1, seq_len))

    model = MeriamTransformer(vocab_size)
    out = model(x)

    print("Shape entrée :", x.shape)
    print("Shape sortie :", out.shape)
