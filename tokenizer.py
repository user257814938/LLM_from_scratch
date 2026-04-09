# === MeriamTokenizer : mon traducteur texte <-> nombres ===
#
# LEXIQUE POUR DÉBUTANT :
# - Tokenizer (ou "Tokeniseur") : Une machine ne comprend pas les lettres
#   naturelles (A, B, C...). Le tokenizer est le dictionnaire bilingue qui 
#   traduit notre texte humain en séries de nombres mathématiques.
# - Token : C'est la plus petite unité de texte découpée. 
#   Dans ce mini-projet, un "token" correspond tout simplement à une lettre unique.
# ==========================================================

class MeriamTokenizer:
    def __init__(self):
        # vocab (vocabulaire) : Le dictionnaire qui traduit "Lettre" -> "Nombre" (ex: 'A' -> 1)
        # inverse_vocab : Le dictionnaire inverse pour "Nombre" -> "Lettre" (ex: 1 -> 'A')
        self.vocab = {}
        self.inverse_vocab = {}

    def build_vocab(self, text):
        """
        Crée le vocabulaire à partir d'un texte de base d'entraînement.
        On prend tout le texte, on garde uniquement les lettres/symboles uniques,
        et on attribue un numéro (identifiant) à chaque symbole.
        Exemple pour "chat" : { 'a': 0, 'c': 1, 'h': 2, 't': 3 }
        """
        chars = sorted(list(set(text)))
        self.vocab = {ch: i for i, ch in enumerate(chars)}
        self.inverse_vocab = {i: ch for ch, i in self.vocab.items()}
        print("Vocabulaire créé :", len(self.vocab), "tokens")

    def encode(self, text):
        """
        L'Encodage : Transforme une phrase écrite par un humain 
        en une liste de nombres (les fameux "tokens") lisibles par l'IA.
        """
        return [self.vocab[c] for c in text if c in self.vocab]

    def decode(self, ids):
        """
        Le Décodage : Fait l'opération inverse. Quand l'IA donne sa réponse 
        en nombres mathématiques, on re-traduit ces nombres en vrai texte lisible pour nous.
        """
        return "".join(self.inverse_vocab[i] for i in ids)


# === Petit test (qui se lance seulement si on exécute ce fichier avec Python) ===
if __name__ == "__main__":
    t = MeriamTokenizer()
    exemple = "Bonjour je suis Meriam et je crée une IA."
    t.build_vocab(exemple)

    codes = t.encode("Bonjour")
    print("Encode 'Bonjour' :", codes)

    texte = t.decode(codes)
    print("Decode ->", texte)
