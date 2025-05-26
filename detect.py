# ===========================
# SORT Tracking Interno (sin dependencias externas)
# ===========================
import numpy as np
import cv2
from ultralytics import YOLO
from twilio.rest import Client

# ===========================
# Configuración de Twilio (reemplaza con tus credenciales)
# ===========================
TWILIO_ACCOUNT_SID = 'AC1499daca606f015c58fcde9c87549da3'
TWILIO_AUTH_TOKEN  = 'ca2b5fe8acb4fc3224509b1f6d8dde5a'
TWILIO_FROM        = '+19788003568'     # Tu número de Twilio
TWILIO_TO          = '+573054004214'     # Número destino (Colombia +57)

# Inicializar cliente de Twilio
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Tracker por IOU sin dependencias externas
class Tracker:
    def __init__(self):
        self.id_count = 0
        self.trackers = []
        self.max_age = 20
        self.min_hits = 1
        self.iou_threshold = 0.2

    def iou(self, bb_test, bb_gt):
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
                  (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
        return o

    def update(self, dets):
        updated_tracks = []
        for det in dets:
            matched = False
            for trk in self.trackers:
                if self.iou(det[:4], trk['bbox']) > self.iou_threshold:
                    trk['bbox'] = det[:4]
                    trk['hits'] += 1
                    trk['no_losses'] = 0
                    matched = True
                    break
            if not matched:
                self.trackers.append({
                    'bbox': det[:4],
                    'hits': 1,
                    'id': self.id_count,
                    'no_losses': 0
                })
                self.id_count += 1

        for trk in self.trackers:
            trk['no_losses'] += 1
        self.trackers = [t for t in self.trackers if t['no_losses'] <= self.max_age and t['hits'] >= self.min_hits]

        for trk in self.trackers:
            updated_tracks.append((*trk['bbox'], trk['id']))
        return updated_tracks

# ===========================
# Código principal
# ===========================

model = YOLO('yolov8s.pt')
det_tracker = Tracker()
LINE_Y = 300
LINE_OFFSET = 5
count_vehicles = set()
count_motorbikes = set()

def detectar_y_filtrar(frame):
    results = model(frame)[0]
    dets = []
    for box in results.boxes:
        cls_id = int(box.cls.cpu().numpy()[0])
        conf = float(box.conf.cpu().numpy()[0])
        if cls_id in [2, 3]:
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            dets.append((x1, y1, x2, y2, cls_id, conf))
    return dets

def procesa_frame(frame):
    global count_vehicles, count_motorbikes
    h, w, _ = frame.shape
    if w > 1080:
        new_w = 1080
        new_h = int(h * (1080 / w))
        frame = cv2.resize(frame, (new_w, new_h))
        h, w = new_h, new_w

    dets = detectar_y_filtrar(frame)
    dets_array = np.array([[x1, y1, x2, y2, conf] for x1, y1, x2, y2, cls, conf in dets])
    tracks = det_tracker.update(dets_array)

    for x1, y1, x2, y2, tid in tracks:
        for bx1, by1, bx2, by2, cls_id, conf in dets:
            if abs(bx1 - x1) < 8 and abs(by1 - y1) < 8:
                cy = (y1 + y2) // 2
                if cy < LINE_Y <= cy + LINE_OFFSET:
                    if cls_id == 2:
                        count_vehicles.add(tid)
                    elif cls_id == 3:
                        count_motorbikes.add(tid)
                break

    for x1, y1, x2, y2, cls_id, conf in dets:
        color = (255, 0, 0) if cls_id == 2 else (0, 0, 255)
        label = f"Carro {conf:.1f}" if cls_id == 2 else f"Moto {conf:.1f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.line(frame, (0, LINE_Y), (w, LINE_Y), (0, 255, 0), 2)
    cv2.putText(frame, f"Vehiculos: {len(count_vehicles)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Motos: {len(count_motorbikes)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def main():
    cap = cv2.VideoCapture('multimedia/final3.mp4')
    if not cap.isOpened():
        print("No se pudo abrir el video")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        salida = procesa_frame(frame)
        cv2.imshow("Conteo Car Wash", salida)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Resumen diario:\n Vehiculos: {len(count_vehicles)}\n Motos: {len(count_motorbikes)}")

    # Enviar resumen diario al final
    resumen = f"A continuación el Resumen del día en el CarWash fue:\nVehículos detectados: {len(count_vehicles)}\nMotos detectadas: {len(count_motorbikes)}"
    try:
        client.messages.create(body=resumen, from_=TWILIO_FROM, to=TWILIO_TO)
        print("Mensaje de resumen enviado")
    except Exception as e:
        print("Error al enviar mensaje de resumen:", e)

if __name__ == "__main__":
    main()
