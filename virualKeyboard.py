import cv2
import numpy as np
from time import time
from pynput.keyboard import Controller, Key
import pyautogui
import mediapipe as mp
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

screen_width, screen_height = pyautogui.size()

window_width = screen_width // 2
window_height = screen_height // 2

window_name = 'Virtual Keyboard'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

cv2.resizeWindow(window_name, window_width, window_height)
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


keyboard_keys = [
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "<"],
    ["CAPS", "SPACE", "ENTER", "BACKSPACE"]
]


keyboard = Controller()

class Button:
    def __init__(self, pos, text, size=(85, 85)):
        self.pos = pos
        self.size = size
        self.text = text

def draw_buttons(img, button_list, caps_on):
    for button in button_list:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, button.pos, (x + w, y + h), (255, 255, 255), cv2.FILLED)
        if button.text == "CAPS":
            text = "CAPS" if caps_on else "caps"
        else:
            text = button.text.upper() if caps_on else button.text.lower()
        cv2.putText(img, text, (x+20, y+65), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
    return img

button_list = []

padding = 10
caps_button_padding = 10
enter_button_padding = 10
backspace_button_padding = 130

for k in range(len(keyboard_keys)):
    for x, key in enumerate(keyboard_keys[k]):
        xpos = (100 + padding) * x + 80
        ypos = (100 + padding) * k + 80
        if key != "SPACE" and key != "ENTER" and key != "BACKSPACE" and key != "CAPS":
            button_list.append(Button((xpos, ypos), key))
        elif key == "ENTER":
            button_list.append(Button((xpos + enter_button_padding - 30, ypos), key, (220, 85))) 
        elif key == "SPACE":
            button_list.append(Button((xpos + 750, ypos), key, (220, 85)))  
        elif key == "BACKSPACE":
            button_list.append(Button((xpos + enter_button_padding + backspace_button_padding - 30, ypos), key, (400, 85)))  
        elif key == "CAPS":
            button_list.append(Button((xpos + caps_button_padding - 30, ypos), key, (200, 85)))  

caps_on = False
last_click_time = time()
click_delay = 0.4
sensitivity = 25

while True:
    success, img = cap.read()
    if not success:
        break
    
    img = cv2.flip(img, 1) 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lm_list = [(int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for lm in hand_landmarks.landmark]
    else:
        lm_list = []

    img = draw_buttons(img, button_list, caps_on)

    if lm_list:
        for button in button_list:
            x, y = button.pos
            w, h = button.size

            if x < lm_list[8][0] < x + w and y < lm_list[8][1] < y + h:
                cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 0), cv2.FILLED)
                text = button.text.upper() if caps_on else button.text.lower()
                cv2.putText(img, text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)

                distance = np.sqrt((lm_list[8][0] - lm_list[4][0])**2 + (lm_list[8][1] - lm_list[4][1])**2)

                if distance < sensitivity and (time() - last_click_time) > click_delay:
                    last_click_time = time()
                    if button.text == "CAPS":
                        caps_on = not caps_on
                    elif button.text not in ['ENTER', "BACKSPACE", "SPACE"]:
                        key = button.text.upper() if caps_on else button.text.lower()
                        keyboard.press(key)
                        keyboard.release(key)
                    else:
                        if button.text == "SPACE":
                            keyboard.press(Key.space)
                            keyboard.release(Key.space)
                        elif button.text == "ENTER":
                            keyboard.press(Key.enter)
                            keyboard.release(Key.enter)                          
                        elif button.text == "BACKSPACE":
                            keyboard.press(Key.backspace)
                            keyboard.release(Key.backspace)

                    cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)

    cv2.imshow(window_name, img)

    if cv2.waitKey(1) & 0xFF == ord('q'):                  
        break

cap.release()
cv2.destroyAllWindows()