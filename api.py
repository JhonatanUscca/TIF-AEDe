from fastapi import FastAPI, UploadFile, File
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64

# Importar el modelo desde el archivo archs.py de tu carpeta
import archs

app = FastAPI(title="UNeXt Segmentation API")

# 1. Configurar el dispositivo y cargar el modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Iniciar la arquitectura (ajusta num_classes si en tu config.yml usaste otro valor)
# En este repositorio, la clase normalmente se llama UNext
model = archs.UNext(num_classes=1, input_channels=3) 

# Cargar los pesos desde tu carpeta de experimentos
model.load_state_dict(torch.load('models/primer_experimento/model.pth', map_location=device))
model.to(device)
model.eval() 

# Transformaciones de imagen
transform = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 2. Leer la imagen
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 3. Preprocesar
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 4. Inferencia
    with torch.no_grad():
        output = model(input_tensor)
        preds = torch.sigmoid(output) 
        preds = (preds > 0.5).float()
    
    # 5. Postprocesar
    pred_mask = preds[0].cpu().numpy().squeeze()
    mask_image = Image.fromarray((pred_mask * 255).astype('uint8'))
    
    # 6. Devolver Base64
    buffered = io.BytesIO()
    mask_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {"filename": file.filename, "segmentation_mask_base64": img_str}