import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Chargement du modèle local
DOSSIER_MODELE = "mon_modele_ia"

tokenizer = AutoTokenizer.from_pretrained(DOSSIER_MODELE)
model = AutoModelForCausalLM.from_pretrained(DOSSIER_MODELE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Fonction IA
def joms_chat(message, historique):
    prompt = ""
    for user, ai in historique:
        prompt += f"Utilisateur : {user}\nIA : {ai}\n"

    prompt += f"Utilisateur : {message}\nIA :"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )

    texte = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extraire seulement la réponse après "IA :"
    if "IA :" in texte:
        reponse = texte.split("IA :")[-1].strip()
    else:
        reponse = texte.strip()

    historique.append((message, reponse))
    return historique, historique

# Interface GRADIO PRO
with gr.Blocks(css="""
#title { 
    text-align:center; 
    font-size:40px; 
    font-weight:bold; 
    margin-bottom:20px; 
    color:#6C63FF;
}
""") as joms_ui:

    gr.HTML("<div id='title'>✨ JOMS — Ton IA Personnelle ✨</div>")

    with gr.Row():
        with gr.Column(scale=1):
            historique = gr.Chatbot(label="Historique des discussions")

        with gr.Column(scale=3):
            message = gr.Textbox(label="Écris ton message ici…", placeholder="Pose une question à JOMS...")
            bouton = gr.Button("Envoyer")

    bouton.click(joms_chat, inputs=[message, historique], outputs=[historique, historique])

joms_ui.launch()


