
# train_ia.py
# Entraîne une petite IA locale (mini modèle type GPT) sur ton fichier data.txt

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------- CONFIG ----------
MODELE_DE_BASE = "dbddv01/gpt2-french-small"

      # petit modèle GPT allégé
TAILLE_SEQUENCE = 64              # nombre de tokens par exemple
BATCH_SIZE = 16
N_STEPS = 2000                  # plus tu montes, plus ça apprend (mais c'est plus long)
LR = 5e-5                         # learning rate (vitesse d'apprentissage)
DOSSIER_SORTIE = "mon_modele_ia"  # où on va sauvegarder ton modèle
# -----------------------------


print("📂 Lecture de data.txt ...")
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

if len(text.strip()) == 0:
    raise ValueError("⚠️ Ton fichier data.txt est vide, mets du texte dedans avant d'entraîner l'IA.")

print(" Chargement du tokenizer et du modèle de base :", MODELE_DE_BASE)
tokenizer = AutoTokenizer.from_pretrained(MODELE_DE_BASE)
model = AutoModelForCausalLM.from_pretrained(MODELE_DE_BASE)

# Certains modèles n'ont pas de token pad → on le force
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("🧮 Tokenisation du texte...")
encodings = tokenizer(text, return_tensors="pt")

input_ids = encodings["input_ids"][0]  # (longueur_totale,)
n_tokens = input_ids.size(0)

if n_tokens <= TAILLE_SEQUENCE + 1:
    raise ValueError("⚠️ Il n'y a pas assez de texte dans data.txt. Ajoute plus de phrases / texte.")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

def get_batch():
    """
    Prend des morceaux aléatoires du texte pour faire un batch.
    """
    ix = torch.randint(0, n_tokens - TAILLE_SEQUENCE - 1, (BATCH_SIZE,))
    x_batch = []
    y_batch = []
    for i in ix:
        i = int(i)
        x = input_ids[i : i + TAILLE_SEQUENCE]
        y = input_ids[i + 1 : i + 1 + TAILLE_SEQUENCE]
        x_batch.append(x)
        y_batch.append(y)

    x_batch = torch.stack(x_batch).to(device)  # (BATCH_SIZE, TAILLE_SEQUENCE)
    y_batch = torch.stack(y_batch).to(device)
    return x_batch, y_batch

print("🚀 Début de l'entraînement sur ton texte...")
model.train()

for step in range(1, N_STEPS + 1):
    x, y = get_batch()
    outputs = model(input_ids=x, labels=y)
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Étape {step}/{N_STEPS} - Perte : {loss.item():.4f}")

print("✅ Entraînement terminé.")

print(f"💾 Sauvegarde du modèle dans le dossier : {DOSSIER_SORTIE}")
model.save_pretrained(DOSSIER_SORTIE)
tokenizer.save_pretrained(DOSSIER_SORTIE)

print("🎉 Ton mini modèle d'IA est prêt ! Tu peux maintenant l'utiliser dans chat_ia.py")
# ----- SAUVEGARDE DU MODELE -----
DOSSIER_SORTIE = "mon_modele_ia"

model.save_pretrained(DOSSIER_SORTIE)
tokenizer.save_pretrained(DOSSIER_SORTIE)

print(f"✅ Modèle sauvegardé dans : {DOSSIER_SORTIE}")
DOSSIER_SORTIE = "mon_modele_ia"

model.save_pretrained(DOSSIER_SORTIE)
tokenizer.save_pretrained(DOSSIER_SORTIE)

