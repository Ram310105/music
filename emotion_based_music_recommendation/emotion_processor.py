import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

class EmotionProcessor:
    def __init__(self, model_path, label_path):
        self.model = load_model(model_path)
        self.label = np.load(label_path)

        # Initialize MediaPipe Holistic for face and hand detection
        self.holis = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.drawing = mp.solutions.drawing_utils

    def recv(self, frame):
        # No need to call to_ndarray() since 'frame' is already a NumPy array
        frm = frame  
        frm = cv2.flip(frm, 1)

        res = self.holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1,-1)

            pred = self.label[np.argmax(self.model.predict(lst))]

            cv2.putText(frm, pred, (50,50), cv2.FONT_ITALIC, 1, (255,0,0), 2)

            np.save("emotion.npy", np.array([pred]))

        self.drawing.draw_landmarks(frm, res.face_landmarks, mp.solutions.holistic.FACEMESH_TESSELATION,
                                    landmark_drawing_spec=self.drawing.DrawingSpec(color=(0,0,255), thickness=-1, circle_radius=1),
                                    connection_drawing_spec=self.drawing.DrawingSpec(thickness=1))
        self.drawing.draw_landmarks(frm, res.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        self.drawing.draw_landmarks(frm, res.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        return frm
