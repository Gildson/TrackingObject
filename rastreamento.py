import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Carregar o modelo YOLOv8
model = YOLO('yolov8n.pt')  # YOLOv8 nano model, para outros modelos consulta a documentação

# Inicializar o rastreador DeepSORT
tracker = DeepSort()

# Abrir a captura de vídeo (0 para webcam ou o caminho de um vídeo)
cap = cv2.VideoCapture('carros_2.mp4')

# Contadores
count_in = 0  # Contagem de objetos que cruzam a linha de cima para baixo
count_out = 0  # Contagem de objetos que cruzam a linha de baixo para cima

# Dicionário para armazenar a posição anterior dos objetos
object_states = {}

# Classe YOLOv8 para carros
VEHICLE_CLASSES = [2]  # Se precisar de outras classes olhar a documentação e alterar ou adicionar a essa lista

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar o frame para melhorar o desempenho
    frame = cv2.resize(frame, (640, 480))

    # Realizar a detecção de objetos usando YOLOv8
    results = model(frame)

    # Calcular a posição da linha no centro da tela
    frame_height = frame.shape[0]
    line_position = frame_height // 2  # Linha no centro da tela

    # Desenhar o objeto detectado
    frame = results[0].plot()

    # Converter as detecções para o formato [x, y, w, h, conf, class]
    detections = results[0].boxes.xywh.cpu().numpy()  # Caixas delimitadoras (x, y, w, h)
    confidences = results[0].boxes.conf.cpu().numpy()  # Confiança
    class_ids = results[0].boxes.cls.cpu().numpy()  # Classes dos objetos

    # Verificar se há detecções
    if len(detections) > 0:

        # Converter as detecções para o formato esperado pelo DeepSORT
        deepsort_input = []
        for i in range(len(detections)):
            x, y, w, h = detections[i]
            conf = confidences[i]
            cls = int(class_ids[i])

            # Verificar se o objeto detectado é um veículo
            if cls in VEHICLE_CLASSES:
                deepsort_input.append([[x, y, w, h], conf])  # Não passar a classe para o DeepSORT, apenas a caixa e confiança

        # Atualizar o rastreador DeepSORT com as novas detecções
        tracks = tracker.update_tracks(deepsort_input, frame=frame)

        # Desenhar a linha delimitadora
        cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (255, 0, 0), 2)

        # Iterar sobre os objetos rastreados
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  # Retorna [left, top, right, bottom]
            x, y, w, h = ltrb
            center_y = (y + h) // 2  # Ponto central da caixa no eixo Y

            # Inicializar o estado do objeto se ele ainda não foi visto
            if track_id not in object_states:
                if center_y < line_position:
                    object_states[track_id] = {"previous_y": center_y, "crossed_line": False, "entrace":True}
                else:
                    object_states[track_id] = {"previous_y": center_y, "crossed_line": False, "entrace":False}                    

            # Verificar se o objeto cruzou a linha
            if not object_states[track_id]["crossed_line"]:
                # Verificar se o objeto cruzou a linha de cima para baixo
                if object_states[track_id]["previous_y"] > line_position and object_states[track_id]["entrace"]:
                    count_in += 1
                    object_states[track_id]["crossed_line"] = True  # Marcar como contado
                    print(f'Objeto {track_id} cruzou de cima para baixo. Contagem IN: {count_in}')

                # Verificar se o objeto cruzou a linha de baixo para cima
                elif object_states[track_id]["previous_y"] < line_position and not object_states[track_id]["entrace"]:
                    count_out += 1
                    object_states[track_id]["crossed_line"] = True  # Marcar como contado
                    print(f'Objeto {track_id} cruzou de baixo para cima. Contagem OUT: {count_out}')

            # Atualizar a posição anterior do objeto
            object_states[track_id]["previous_y"] = center_y

    # Exibir o vídeo com as caixas de detecção e linha delimitadora
    cv2.putText(frame, f'IN: {count_in}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, f'OUT: {count_out}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Object Tracking', frame)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
