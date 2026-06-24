from fastapi import FastAPI, UploadFile, File
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import numpy as np

# Importar el modelo desde el archivo archs.py de tu carpeta
import archs

app = FastAPI(title="UNeXt Multiclass Segmentation API")

# =====================================================================
# CONFIGURACIÓN DEL EXPERIMENTO
# Cambia este número según cuántas carpetas de máscaras (0, 1, 2...) usaste en el entrenamiento
NUM_CLASSES = 1 
# =====================================================================

# 1. Configurar el dispositivo y cargar el modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Iniciar la arquitectura con el número de clases configurado
model = archs.UNext(num_classes=NUM_CLASSES, input_channels=3) 

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
    # 2. Leer la imagen que sube el usuario
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 3. Preprocesar para la red neuronal
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 4. Inferencia del modelo UNeXt
    with torch.no_grad():
        output = model(input_tensor) # Salida de la red: [1, NUM_CLASSES, 256, 256]
        
        if NUM_CLASSES > 1:
            # LÓGICA MULTICLASE:
            probs = torch.softmax(output, dim=1) # Obtiene probabilidades para cada clase
            preds = torch.argmax(probs, dim=1)   # Elige la clase con mayor valor [1, 256, 256]
            
            # Extraemos la probabilidad más alta de cada píxel para saber la certeza
            max_probs = torch.max(probs, dim=1)[0]
            certeza_promedio = float(max_probs.mean().item()) * 100
        else:
            # LÓGICA BINARIA (Por si vuelves a 1 sola clase):
            probs = torch.sigmoid(output)
            preds = (probs > 0.5).float()
            lesion_pixels = probs[preds == 1]
            certeza_promedio = float(lesion_pixels.mean().item()) * 100 if len(lesion_pixels) > 0 else 0.0
    
    # 5. Postprocesar la máscara para convertirla en imagen
    pred_mask = preds[0].cpu().numpy().squeeze() # Matriz con valores de clases (0, 1, 2...)
    
    # Para poder visualizar las clases en la imagen PNG:
    # Si es binario, multiplicamos por 255 (0=negro, 1=blanco)
    # Si es multiclase, escalamos los valores para que cada clase tenga un tono de gris diferente
    if NUM_CLASSES > 1:
        factor_escala = 255 // (NUM_CLASSES - 1)
        mask_visual = (pred_mask * factor_escala).astype('uint8')
    else:
        mask_visual = (pred_mask * 255).astype('uint8')
        
    mask_image = Image.fromarray(mask_visual)
    
    # 6. Convertir la imagen resultante a una cadena Base64
    buffered = io.BytesIO()
    mask_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # 7. Identificar qué clases principales se detectaron en la imagen
    clases_detectadas = np.unique(pred_mask).tolist()
    # Removemos el 0 si asumimos que 0 es siempre el fondo (piel sana)
    enfermedades_detectadas = [int(c) for c in clases_detectadas if c != 0]
    
    # 8. Devolver los resultados en formato JSON
    return {
        "filename": file.filename, 
        "porcentaje_certeza_modelo": f"{round(certeza_promedio, 2)}%",
        "clases_detectadas_en_píxeles": enfermedades_detectadas,
        "segmentation_mask_base64": img_str
    }