import cv2
import mediapipe as mp
import serial
import time

# Configura la conexión serial (ajusta el puerto COM y la velocidad según tu Arduino)
arduino = serial.Serial('COM4', 9600)
time.sleep(2)  # Da tiempo para que el Arduino reinicie

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Inicializar captura de video
cap = cv2.VideoCapture(0)

# Para llevar el control del último número enviado
last_finger_count = -1

def count_fingers(hand_landmarks):
    fingers = []

    # Thumb (pulgar)
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Fingers: Index, Middle, Ring, Pinky
    tips_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]
    pip_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP,
    ]

    for tip, pip in zip(tips_ids, pip_ids):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Voltear imagen para efecto espejo
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                finger_count = count_fingers(hand_landmarks)

                # Solo enviamos al Arduino si el número cambia
                if finger_count != last_finger_count:
                    print(f"Dedos detectados: {finger_count}")
                    arduino.write(f"{finger_count}\n".encode())
                    last_finger_count = finger_count

                # Mostrar número en pantalla
                cv2.putText(frame, f'Dedos: {finger_count}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv2.imshow('Detector de Dedos', frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Programa interrumpido manualmente.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()