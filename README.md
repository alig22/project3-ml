# project3-ml

# 🧠 Mini-projet : Entraîner un modèle Image-to-Text dans Google Colab

Ce notebook vous permet de :
- Charger un mini-dataset image + texte
- Visualiser les images et légendes
- Entraîner un modèle ViT + GPT-2
- Générer une légende automatiquement à partir d'une image

---
# Étape 1 : Installer les bibliothèques nécessaires
!pip install transformers datasets torchvision --quiet
# Étape 2 : Télécharger manuellement le fichier ZIP
from google.colab import files

print("⬆️ Veuillez téléverser le fichier 'dataset_caption_demo.zip'")
uploaded = files.upload()

import zipfile
import os

zip_path = "dataset_caption_demo.zip"
extract_path = "dataset_caption_demo"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("✅ Fichiers extraits :", os.listdir(extract_path))
# Étape 3 : Charger et afficher les données
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

df = pd.read_csv(f"{extract_path}/captions.csv")
print(df.head())

# Affichage d'une image avec légende
img_path = f"{extract_path}/{df.iloc[0]['image_path']}"
caption = df.iloc[0]['caption']
img = Image.open(img_path)
plt.imshow(img)
plt.title(caption)
plt.axis('off')
plt.show()
# Étape 4 : Préparer le modèle et les données
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

# Initialiser modèle et tokenizer
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained("google/vit-base-patch16-224", "gpt2")
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.vocab_size = model.config.decoder.vocab_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Dataset personnalisé
class CaptionDataset(Dataset):
    def __init__(self, df, transform, tokenizer, base_path):
        self.df = df
        self.transform = transform
        self.tokenizer = tokenizer
        self.base_path = base_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.base_path, self.df.iloc[idx]['image_path'])).convert('RGB')
        image = self.transform(image)
        caption = self.df.iloc[idx]['caption']
        labels = self.tokenizer(caption, padding="max_length", truncation=True, max_length=64, return_tensors="pt").input_ids.squeeze()
        return image, labels

# Transformations pour les images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = CaptionDataset(df, transform, tokenizer, extract_path)
loader = DataLoader(dataset, batch_size=2, shuffle=True)
# Étape 5 : Entraîner brièvement le modèle (1 epoch)
from torch import nn, optim
from tqdm import tqdm

model.train()
optimizer = optim.Adam(model.parameters(), lr=5e-5)

for epoch in range(1):
    loop = tqdm(loader, desc="Entraînement")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        outputs = model(pixel_values=images, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loop.set_postfix(loss=loss.item())
# Étape 6 : Générer une légende pour une image
model.eval()

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(image, return_tensors="pt").pixel_values.to(device)
    output_ids = model.generate(pixel_values, max_length=64)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# Exemple
test_image = f"{extract_path}/{df.iloc[1]['image_path']}"
print("📸 Image test :", test_image)
Image.open(test_image).show()
print("📝 Légende générée :", generate_caption(test_image))
