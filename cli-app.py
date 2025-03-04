import cv2
import face_recognition
import numpy as np
import os
import datetime
import requests
import serial
import keyboard  # Library for detecting key presses

# --- Bluetooth Setup ---
try:
    ser = serial.Serial("COM7", 9600, timeout=1)
    print("Bluetooth serial connection established.")
except Exception as e:
    print("Bluetooth Serial Error:", e)
    ser = None

# --- Known Face Recognition Setup ---
known_faces_path = r"C:\Users\arsha\OneDrive\Desktop\new-bot\images"
known_images = []
classnames = []
mylist = os.listdir(known_faces_path)
print("Known image list:", mylist)

for cl in mylist:
    img_path = os.path.join(known_faces_path, cl)
    crimg = cv2.imread(img_path)
    if crimg is None:
        continue
    known_images.append(crimg)
    classnames.append(os.path.splitext(cl)[0])

print("Known class names:", classnames)

def find_encodings(images):
    encodelist = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img_rgb)[0]
        except IndexError:
            continue  # Skip images with no detectable face
        encodelist.append(encode)
    return encodelist

encodelist_known = find_encodings(known_images)
print("Encoding complete...")

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Set SIMULATION_MODE to False to use Bluetooth communication.
SIMULATION_MODE = False

def send_command(command):
    # Send the command via Bluetooth if available.
    if ser is not None:
        try:
            ser.write((command + "\n").encode('utf-8'))
        except Exception as e:
            print("Error sending command:", e)
    print(f"Sent command: {command}")

# Use your ESP32-CAM stream URL.
stream_url = "http://192.168.1.8/mjpeg/1"  # Adjust if necessary

def process_video_stream():
    r = requests.get(stream_url, stream=True)
    bytes_data = bytes()
    for chunk in r.iter_content(chunk_size=1024):
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            data = np.frombuffer(jpg, dtype=np.uint8)
            if data.size == 0:
                continue
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # --- Face Detection & Recognition ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            recognized = False  # Flag to check if at least one face is recognized
            for (x, y, w, h) in faces:
                # Convert Haar detection coordinates to face_recognition format: (top, right, bottom, left)
                top, right, bottom, left = y, x+w, y+h, x
                face_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(face_rgb, [(top, right, bottom, left)])
                name = "Unknown"
                if face_encodings:
                    encoding = face_encodings[0]
                    matches = face_recognition.compare_faces(encodelist_known, encoding)
                    face_distances = face_recognition.face_distance(encodelist_known, encoding)
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = classnames[best_match_index].upper()
                            recognized = True  # A recognized face is found
                print(f"Detected face: {name} at ({x}, {y})")
            # If a recognized face is found, send the "Open" command via Bluetooth
            if recognized:
                send_command("Open")
            # --- End Face Detection & Recognition ---

# --- Keyboard Teleoperation ---
def teleop():
    print("Starting teleoperation. Use Arrow Keys to control the robot.")
    print("Press 'Q' to quit.")

    # Map keys to commands
    key_command_map = {
        "up": "Forward",
        "down": "Backward",
        "left": "Left",
        "right": "Right",
    }

    while True:
        # Check for key presses
        for key, command in key_command_map.items():
            if keyboard.is_pressed(key):
                send_command(command)
                break  # Send only one command at a time
        else:
            # If no movement key is pressed, send "Stop"
            send_command("Stop")

        # Exit on 'Q' key press
        if keyboard.is_pressed("q"):
            print("Exiting teleoperation.")
            break

def main():
    parser = argparse.ArgumentParser(description="SCOUTX - Surveillance Robot Control")
    parser.add_argument("--teleop", action="store_true", help="Start keyboard teleoperation")
    args = parser.parse_args()

    if args.teleop:
        teleop()
    else:
        print("Starting video stream processing...")
        process_video_stream()

if __name__ == "__main__":
    main()