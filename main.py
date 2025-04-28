import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import math
import sounddevice as sd
from scipy.io.wavfile import write
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# 🎵 Initialize Pygame for Music
pygame.mixer.init()
playlist = ["your_song.mp3", "song2.mp3", "song3.mp3","song1.mp3"] # Replace with actual paths
current_song_index = 0
pygame.mixer.music.load(playlist[current_song_index])
pygame.mixer.music.play(-1)
pygame.mixer.music.pause()

# ✋ Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)  # Only one hand

# 🔊 Get System Audio Control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
min_vol, max_vol = volume.GetVolumeRange()[0], volume.GetVolumeRange()[1]

# 🎼 Initialize Variables
music_playing = False
last_gesture_time = 0
gesture_cooldown = 1.5  # Prevents repeated triggers
last_clap_time = 0
clap_count = 0
song_changed = False
muted = False

# 🎤 Sound Detection Function
def detect_clap(indata, frames, time, status):
    global last_clap_time, clap_count, music_playing

    volume_norm = np.linalg.norm(indata) * 10  # Normalize sound level
    if volume_norm > 50:  # Adjust sensitivity
        current_time = time.inputBufferAdcTime  # Get timestamp
        if current_time - last_clap_time < 0.6:  # Double Clap Detection
            clap_count += 1
        else:
            clap_count = 1  # Reset if timeout exceeded
        last_clap_time = current_time

        if clap_count == 1:  # Single Clap = Stop
            pygame.mixer.music.stop()
            print("🛑 Music Stopped")
        elif clap_count == 2:  # Double Clap = Play
            pygame.mixer.music.play()
            print("🎵 Double Clap - Music Playing")

# Start Sound Listener
stream = sd.InputStream(callback=detect_clap)
stream.start()

# 🎥 Start Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_time = time.time()
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = hand_landmarks.landmark
            wrist = landmarks[mp_hands.HandLandmark.WRIST]
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]

            h, w, _ = frame.shape
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
            wx, wy = int(wrist.x * w), int(wrist.y * h)

            # 🎚 Volume Control
            distance = math.hypot(tx - ix, ty - iy)
            volume_level = np.interp(distance, [20, 150], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(volume_level, None)
            volume_percent = int(np.interp(volume_level, [min_vol, max_vol], [0, 100]))
            cv2.putText(frame, f'Volume: {volume_percent}%', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 🔇 Mute Audio (Wrist Below Index Finger)
            if wy > iy and not muted:
                volume.SetMasterVolumeLevel(min_vol, None)
                muted = True
                print("🔇 Muted")
            elif wy < iy and muted:
                muted = False  # Unmute when wrist goes up

            # 🎵 Play Music (Open Hand)
            if (index_tip.y < wrist.y and middle_tip.y < wrist.y and ring_tip.y < wrist.y and pinky_tip.y < wrist.y):
                if not music_playing:
                    pygame.mixer.music.unpause()
                    music_playing = True
                    print("🎵 Music Playing...")

            # ⏸ Pause Music (Fist)
            elif (index_tip.y > wrist.y and middle_tip.y > wrist.y and ring_tip.y > wrist.y and pinky_tip.y > wrist.y):
                if music_playing:
                    pygame.mixer.music.pause()
                    music_playing = False
                    print("⏸ Music Paused.")

            # ⏭ Switching Songs (Swipe Left/Right)
            if current_time - last_gesture_time > gesture_cooldown:  # Prevent multiple triggers
                if wx - ix > 120 and not song_changed:  # Swipe Right (Next Song)
                    current_song_index = (current_song_index + 1) % len(playlist)
                    pygame.mixer.music.load(playlist[current_song_index])
                    pygame.mixer.music.play(-1)
                    print("▶ Next Song")
                    song_changed = True
                elif ix - wx > 120 and not song_changed:  # Swipe Left (Previous Song)
                    current_song_index = (current_song_index - 1) % len(playlist)
                    pygame.mixer.music.load(playlist[current_song_index])
                    pygame.mixer.music.play(-1)
                    print("◀ Previous Song")
                    song_changed = True
                else:
                    song_changed = False  # Reset flag if no swipe detected

                last_gesture_time = current_time

    cv2.imshow("Hand Gesture Music Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
stream.stop()
stream.close()
