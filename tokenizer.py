# === MeriamTokenizer : mon traducteur texte <-> nombres ===

class MeriamTokenizer:
    def __init__(self):
        # vocab : caractère -> numéro
        # inverse_vocab : numéro -> caractère
        self.vocab = {}
        self.inverse_vocab = {}

    def build_vocab(self, text):
        """
        Crée le vocabulaire à partir d'un texte.
        Chaque caractère unique reçoit un numéro.
        """
        chars = sorted(list(set(text)))
        self.vocab = {ch: i for i, ch in enumerate(chars)}
        self.inverse_vocab = {i: ch for ch, i in self.vocab.items()}
        print("Vocabulaire créé :", len(self.vocab), "tokens")

    def encode(self, text):
        """
        Transforme un texte en liste de nombres.
        """
        return [self.vocab[c] for c in text if c in self.vocab]

    def decode(self, ids):
        """
        Transforme une liste de nombres en texte.
        """
        return "".join(self.inverse_vocab[i] for i in ids)


# === Petit test ===
if __name__ == "__main__":
    t = MeriamTokenizer()
    exemple = "Bonjour je suis Meriam et je crée une IA."
    t.build_vocab(exemple)

    codes = t.encode("Bonjour")
    print("Encode 'Bonjour' :", codes)

    texte = t.decode(codes)
    print("Decode ->", texte)
