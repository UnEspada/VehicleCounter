# 🚗 VehicleCounter 🚗


Este repositorio contiene un modelo de detección de vehículos desarrollado utilizando **YOLOv8**. El objetivo principal es detectar y clasificar vehículos en imágenes o videos con alta precisión y eficiencia.

## 📜 Descripción

El modelo está entrenado para identificar diferentes tipos de vehículos en diversos tipos de escenarios, ya sea autopistas, carreteras, etc.
- Monitoreo del trafico.
- Conteo de vehiculos en el trafico.

## 📂 Estructura del proyecto
├── dataset/ # Dataset utilizado para el entrenamiento 
├── models/ # Modelos preentrenados y optimizados 
├── runs/ # Resultados de entrenamientos (si se requiere)
├── main.py # Codigo de ejecucion principal
├── sort.py # Codigo del cual depende main.py (NO TOCAR)
├── yolov8n.pt # Modelo base de yolov8
├── README.md # Este archivo 
├── requirements.txt # Dependencias del proyecto 
└── dataset.yaml # Configuración del dataset (por si se requiere entrenar, el dataset no esta en repositorio, el archivo dataset.yaml puede ser modificado para otro dataset)


## 🚀 Cómo empezar 🚀

### 1. Clonar el repositorio
```bash
git clone https://github.com/UnEspada/VehicleDetection-YOLOv8.git
cd VehicleDetection

pip install -r requirements.txt

```

Las coordenadas son de la linea que se usa para detectar cada vehiculo, es decir, cada vehiculo que pase por esa linea sera detectado, esto dependera de cada camara, agulo y posicionamiento, pues dependera de donde esta la carretera, calle o autopista.

```
python3 main.py (Coordenadas x1 y1 x2 y2 x3 y3 x4 y4)

```

📚 Dataset
El modelo se entrenó utilizando un dataset personal, sin embargo, si se requiere reentrenar el modelo, ¡puede hacerlo con total libertad!

🛠 Tecnologías utilizadas
YOLOv8: Framework de detección de objetos.
Python: Lenguaje principal del proyecto.
OpenCV: Procesamiento de imágenes.
PyTorch: Framework de entrenamiento de modelos.


