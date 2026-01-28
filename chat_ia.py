import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Dossier où ton modèle entraîné est enregistré
DOSSIER_MODELE = "mon_modele_ia"
TAILLE_MAX_REPONSE = 120

print("📂 Chargement de ton modèle local :", DOSSIER_MODELE)
tokenizer = AutoTokenizer.from_pretrained(DOSSIER_MODELE)
model = AutoModelForCausalLM.from_pretrained(DOSSIER_MODELE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("🤖 IA locale prête ! (tape 'quit' pour quitter)\n")

historique = ""

while True:
    user_input = input("Toi : ")
    
    if user_input.lower() == "quit":
        print("👋 À bientôt !")
        break

    # Construire un prompt propre
    prompt = (
        "Tu es une intelligence artificielle qui répond en français "
        "de façon claire, simple et gentille.\n\n"
        f"Utilisateur : {user_input}\n"
        "IA :"
    )

    # Encoder le prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Génération
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    # Décodage
    texte_genere = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Récupérer uniquement la partie après "IA :"
    if "IA :" in texte_genere:
        reponse = texte_genere.split("IA :")[-1].strip()
    else:
        reponse = texte_genere.strip()

    print("IA :", reponse, "\n")
