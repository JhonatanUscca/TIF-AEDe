
# TIF - UNeXt: Segmentación Rápida de Imágenes Médicas mediante Redes Basadas en MLP

Repositorio oficial del Trabajo de Investigación Formativa (TIF) para la **Universidad Nacional de San Agustín de Arequipa (UNSA)**, Escuela Profesional de Ciencias de la Computación.

Basado en el modelo original: [UNeXt: MLP-based Rapid Medical Image Segmentation Network](https://arxiv.org/abs/2203.04967).

## Introducción

La segmentación de imágenes médicas tradicional con arquitecturas como UNet requiere una gran capacidad computacional. En este proyecto evaluamos e implementamos **UNeXt**, una red basada en perceptrones multicapa (MLP) y convoluciones, diseñada para aplicaciones *point-of-care*. UNeXt reduce significativamente la cantidad de parámetros y la complejidad computacional mediante el uso de bloques *Tokenized MLP* y operaciones de desplazamiento espacial, obteniendo resultados altamente competitivos frente a modelos basados en Transformers.

### Autores

* USCCA GIRALDO, JHONATAN BILBAO
* CERPA GARCÍA, RANDÚ JEAN FRANCO
* LOPEZ ZEGARRA, IVAN ALEXANDER

**Docente:** MSc. Jhon F. Bernedo González

---

## Requisitos e Instalación

El código ha sido probado en entornos con `Python 3.6+` y soporte para `CUDA >= 10.1`.

1. Clona este repositorio:
```bash
git clone [https://github.com/tu-usuario/TIF-UNeXt-UNSA.git](https://github.com/tu-usuario/TIF-UNeXt-UNSA.git)
cd TIF-UNeXt-UNSA

```

2. (Opcional) Crea un entorno virtual e instala las dependencias usando `pip`:

```bash
pip install -r requirements.txt

```

*Dependencias principales: `torch`, `torchvision`, `opencv-python`, `timm`.*

---

## Datasets Utilizados

Para nuestra investigación estrictamente cuantitativa, utilizamos fuentes secundarias validadas:

1. **ISIC 2018** (Lesiones Cutáneas) - [Enlace al Dataset](https://challenge.isic-archive.com/data/)


### Formato de Datos

Asegúrate de estructurar las carpetas de los datasets de la siguiente manera:

```text
inputs
└── isic2018
    ├── images
    |   ├── 001.jpg
    │   ├── 002.jpg
    │   └── ...
    └── masks
        └── 0
            ├── 001.png
            ├── 002.png
            └── ...

```

*(Nota: Para problemas de segmentación binaria, solo se utiliza la carpeta `0` dentro de `masks`).*

---

## 🚀 Entrenamiento y Validación

La configuración del experimento (`primer_experimento`) está parametrizada mediante el archivo `config.yml` utilizando la función de pérdida `BCEDiceLoss` y el optimizador `Adam`.

**1. Entrenar el modelo:**
Ejecuta el siguiente comando con los hiperparámetros establecidos en la investigación:

```bash
python train.py --dataset isic2018 --arch UNext --name primer_experimento --img_ext .jpg --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --b 4

```

**2. Evaluar el modelo:**

```bash
python val.py --name primer_experimento

```

---

## 📊 Resultados Preliminares

Durante el entrenamiento de 100 épocas (registrado en `log.csv`), el modelo demostró una rápida convergencia:

* **Loss inicial (Época 0):** 0.6595
* **Loss final (Época 86):** 0.1737
* **IoU Validación Máximo:** ~82.91%

---

