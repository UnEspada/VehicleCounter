import sys
from ultralytics import YOLO
import cv2
import numpy as np
from sort import *

# Verificar si se han proporcionado las coordenadas como argumentos
if len(sys.argv) != 9:
    print("Uso: python3 main.py x1 y1 x2 y2 x3 y3 x4 y4")
    exit()

# Obtener las coordenadas desde los argumentos
x1, y1 = int(sys.argv[1]), int(sys.argv[2])
x2, y2 = int(sys.argv[3]), int(sys.argv[4])
x3, y3 = int(sys.argv[5]), int(sys.argv[6])
x4, y4 = int(sys.argv[7]), int(sys.argv[8])

# Cargar el modelo
model = YOLO('vehicle_detector.pt')

# Definir las clases del modelo
classes = ['Car', 'Motorcycle', 'Truck', 'Bus', 'Bicycle']

# Leer el video
cap = cv2.VideoCapture('./assets/videos/video4.mp4')  # Cambia a la ruta de tu video

if not cap.isOpened():
    print("Error al abrir el video")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Cargar y ajustar la máscara
mask = cv2.imread('./assets/images/mask2.jpg')  # Cambia a la ruta de tu máscara
mask = cv2.resize(mask, (width, height))

# Configurar el tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.4)

# Definir el área rectangular usando las coordenadas proporcionadas
limit_polygon = np.array([
    [x1, y1],
    [x2, y2],
    [x3, y3],
    [x4, y4]
], np.int32)

total_count = []
count = 0

target_width = 1920
target_height = 1080

mask = cv2.resize(mask, (1920, 1080))

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, (target_width, target_height))

    # Ajustar la resolución de la máscara al frame del video
    mask_region = cv2.bitwise_and(mask, img)
    results = model(mask_region, stream=True)

    detections = np.empty((0, 6))  # Ahora incluye la clase en la última columna

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = int(box.cls[0])  # Clase predicha
            conf = box.conf[0]

            # Filtrar por confianza y clases relevantes
            if conf > 0.4 and label < len(classes):
                current_stat = np.array([x1, y1, x2, y2, conf.cpu().numpy(), label])
                detections = np.vstack((detections, current_stat))

    # Actualizar el tracker con las detecciones
    result_tracker = tracker.update(detections[:, :5])

    # Dibujar el área de conteo
    cv2.polylines(img, [limit_polygon], isClosed=True, color=(0, 0, 255), thickness=4)

    for trk, det in zip(result_tracker, detections):
        x1, y1, x2, y2, id = map(int, trk)
        label = int(det[5])  # Obtener la clase correspondiente
        class_name = classes[label]

        # Dibujar el rectángulo, ID y clase
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'ID: {id}, {class_name}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)
        
        # Verificar si el centroide está dentro del área
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        if cv2.pointPolygonTest(limit_polygon, (cx, cy), False) >= 0:
            if id not in total_count:
                total_count.append(id)
                count += 1

    # Mostrar el conteo total
    cv2.putText(img, f'Total Count: {count}', (40, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
    cv2.imshow('result', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
