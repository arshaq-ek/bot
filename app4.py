import cv2
import tkinter as tk
from tkinter import ttk, Label, Frame, simpledialog
from PIL import Image, ImageTk
import threading
import time
from ttkthemes import ThemedTk
import face_recognition
import numpy as np
import os
import datetime
import requests

# --- Known Face Recognition Setup ---
# Update the file path to your known face images directory.
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

# --- End Known Face Recognition Setup ---

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

SIMULATION_MODE = True

def send_command(command):
    if SIMULATION_MODE:
        print(f"Simulated sending command: {command}")
        status_label.config(text=f"Status: {command}")
    else:
        # Insert actual Bluetooth/serial command-sending code here.
        pass

# Global variable to store the latest frame for capture.
latest_frame = None

def capture_face():
    global latest_frame, known_images, classnames, encodelist_known
    if latest_frame is None:
        status_label.config(text="Status: No frame available yet.")
        return

    # Convert latest frame to grayscale and detect faces using Haar cascade
    gray = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        status_label.config(text="Status: No face detected in captured frame.")
        print("No face detected in captured frame.")
        return

    # For simplicity, capture the first detected face region
    (x, y, w, h) = faces[0]
    face_img = latest_frame[y:y+h, x:x+w]

    name = simpledialog.askstring("Input", "Enter name for captured face:", parent=root)
    if name is None or name.strip() == "":
        status_label.config(text="Status: Capture canceled or invalid name.")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name.strip()}_{timestamp}.jpg"
    file_path = os.path.join(known_faces_path, filename)
    cv2.imwrite(file_path, face_img)
    
    # Update the known faces lists
    known_images.append(face_img.copy())
    classnames.append(name.strip())
    new_encoding = face_recognition.face_encodings(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    if new_encoding:
        encodelist_known.append(new_encoding[0])
    
    status_label.config(text=f"Status: Captured face for {name.strip()}")
    print(f"Captured face for {name.strip()} and saved to {file_path}")

# Create the main Tkinter window with a modern dark theme
root = ThemedTk(theme="arc")
root.title("SCOUTX - Surveillance Robot Control")
root.geometry("800x600")
root.configure(bg="#2c3e50")

# Setup custom ttk style for GitHub green buttons
style = ttk.Style(root)
style.theme_use("clam")
style.configure("GitHub.TButton",
                font=("Helvetica", 12),
                foreground="white",
                background="#2ea44f",
                borderwidth=0,
                focusthickness=3,
                focuscolor="none")
style.map("GitHub.TButton",
          background=[("active", "#2c974b")],
          relief=[("pressed", "sunken"), ("!pressed", "flat")])

# Video frame
video_frame = Frame(root, bg="#2c3e50")
video_frame.pack(pady=10)
video_label = Label(video_frame, bg="#2c3e50")
video_label.pack()

# Control frame with buttons
control_frame = Frame(root, bg="#2c3e50")
control_frame.pack(pady=10)
ttk.Button(control_frame, text="Forward", width=15, command=lambda: send_command("Forward"), style="GitHub.TButton").grid(row=0, column=1, padx=5, pady=5)
ttk.Button(control_frame, text="Left", width=15, command=lambda: send_command("Left"), style="GitHub.TButton").grid(row=1, column=0, padx=5, pady=5)
ttk.Button(control_frame, text="Stop", width=15, command=lambda: send_command("Stop"), style="GitHub.TButton").grid(row=1, column=1, padx=5, pady=5)
ttk.Button(control_frame, text="Right", width=15, command=lambda: send_command("Right"), style="GitHub.TButton").grid(row=1, column=2, padx=5, pady=5)
ttk.Button(control_frame, text="Backward", width=15, command=lambda: send_command("Backward"), style="GitHub.TButton").grid(row=2, column=1, padx=5, pady=5)
ttk.Button(control_frame, text="Capture Face", width=15, command=capture_face, style="GitHub.TButton").grid(row=3, column=1, padx=5, pady=5)

# Status bar
status_label = Label(root, text="Status: Idle", bd=1, relief=tk.SUNKEN,
                     anchor=tk.W, font=("Helvetica", 10), bg="#34495e", fg="white")
status_label.pack(side=tk.BOTTOM, fill=tk.X)

# Use your ESP32-CAM stream URL.
stream_url = "http://192.168.1.8/mjpeg/1"  # Adjust if necessary

# We'll use requests to read the MJPEG stream.
def update_video():
    global latest_frame
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
            latest_frame = frame.copy()
            
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
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            # Print "True" only if at least one face is recognized in this frame
            if recognized:
                print("True")
            # --- End Face Detection & Recognition ---
            
            display_frame = cv2.resize(frame, (640, 480))
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(display_frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk  # Prevent garbage collection
            video_label.configure(image=imgtk)
            time.sleep(0.03)

video_thread = threading.Thread(target=update_video, daemon=True)
video_thread.start()

def on_closing():
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
