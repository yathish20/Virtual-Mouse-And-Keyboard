import cv2
import mediapipe as mp
import pyautogui
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)

prev_index_x, prev_index_y = 0, 0
alpha = 0.8

def smooth_position(new_pos, prev_pos, alpha):
    return alpha * prev_pos + (1 - alpha) * new_pos

window_name = 'Virtual Mouse'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    hand_landmarks = result.multi_hand_landmarks
    
    if hand_landmarks:
        for hand_landmark in hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)
            
            index_tip = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmark.landmark[mp_hands.HandLandmark.THUMB_TIP]
            
            index_x = int(index_tip.x * frame.shape[1])
            index_y = int(index_tip.y * frame.shape[0])
            thumb_x = int(thumb_tip.x * frame.shape[1])
            thumb_y = int(thumb_tip.y * frame.shape[0])
            
            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 255), cv2.FILLED)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 255), cv2.FILLED)
            
            screen_index_x = np.interp(index_x, (0, frame.shape[1]), (0, screen_width))
            screen_index_y = np.interp(index_y, (0, frame.shape[0]), (0, screen_height))
            
            smoothed_index_x = smooth_position(screen_index_x, prev_index_x, alpha)
            smoothed_index_y = smooth_position(screen_index_y, prev_index_y, alpha)
            
            smoothed_index_x = np.clip(smoothed_index_x, 0, screen_width - 1)
            smoothed_index_y = np.clip(smoothed_index_y, 0, screen_height - 1)
            
            pyautogui.moveTo(smoothed_index_x, smoothed_index_y)
            
            prev_index_x, prev_index_y = smoothed_index_x, smoothed_index_y
            
            distance = np.hypot(index_x - thumb_x, index_y - thumb_y)
            
            if distance < 25:
                pyautogui.click()
                pyautogui.sleep(1)
            
    cv2.imshow(window_name, frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()