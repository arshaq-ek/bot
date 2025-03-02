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

# --- Face Recognition Setup ---
# Update the file path to your face images directory.
face_images_path = r"C:\Users\arsha\OneDrive\Desktop\new-bot\images"
images = []
classnames = []
mylist = os.listdir(face_images_path)
print("Image list:", mylist)

for cl in mylist:
    img_path = os.path.join(face_images_path, cl)
    crimg = cv2.imread(img_path)
    if crimg is None:
        continue
    images.append(crimg)
    classnames.append(os.path.splitext(cl)[0])

print("Class names:", classnames)

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

encodelist_known = find_encodings(images)
print("Encoding complete...")

# --- End Face Recognition Setup ---

# Simulation mode flag - set to True to simulate hardware behavior.
SIMULATION_MODE = True

def send_command(command):
    if SIMULATION_MODE:
        print(f"Simulated sending command: {command}")
        status_label.config(text=f"Status: {command}")
    else:
        # Insert actual Bluetooth/serial command-sending code here.
        pass

# Global variable to store the most recent frame from the stream.
latest_frame = None

# Function to capture a face from the current frame of the ESP32-CAM stream
def capture_face():
    global latest_frame, encodelist_known, classnames, images
    if latest_frame is None:
        status_label.config(text="Status: No frame available yet.")
        return

    # Convert the latest frame to RGB for face recognition
    rgb_frame = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    if len(face_encodings) == 0:
        status_label.config(text="Status: No face detected in captured frame.")
        print("No face detected in captured frame.")
        return

    # Use the first detected face
    new_encoding = face_encodings[0]
    name = simpledialog.askstring("Input", "Enter name for captured face:", parent=root)
    if name is None or name.strip() == "":
        status_label.config(text="Status: Capture canceled or invalid name.")
        return

    # Save the captured frame to disk
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name.strip()}_{timestamp}.jpg"
    file_path = os.path.join(face_images_path, filename)
    cv2.imwrite(file_path, latest_frame)
    
    # Update global lists
    images.append(latest_frame.copy())
    classnames.append(name.strip())
    encodelist_known.append(new_encoding)
    
    status_label.config(text=f"Status: Captured face for {name.strip()}")
    print(f"Captured face for {name.strip()} and saved to {file_path}")

# Create the main Tkinter window with a modern dark theme
root = ThemedTk(theme="arc")
root.title("SCOUTX - Surveillance Robot Control")
root.geometry("800x600")
root.configure(bg="#2c3e50")  # Dark background

# Set up a custom ttk style for GitHub green buttons
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

# Create a frame for the video feed with matching dark background
video_frame = Frame(root, bg="#2c3e50")
video_frame.pack(pady=10)
video_label = Label(video_frame, bg="#2c3e50")
video_label.pack()

# Create a frame for control buttons with dark background
control_frame = Frame(root, bg="#2c3e50")
control_frame.pack(pady=10)
ttk.Button(control_frame, text="Forward", width=15, command=lambda: send_command("Forward"), style="GitHub.TButton").grid(row=0, column=1, padx=5, pady=5)
ttk.Button(control_frame, text="Left", width=15, command=lambda: send_command("Left"), style="GitHub.TButton").grid(row=1, column=0, padx=5, pady=5)
ttk.Button(control_frame, text="Stop", width=15, command=lambda: send_command("Stop"), style="GitHub.TButton").grid(row=1, column=1, padx=5, pady=5)
ttk.Button(control_frame, text="Right", width=15, command=lambda: send_command("Right"), style="GitHub.TButton").grid(row=1, column=2, padx=5, pady=5)
ttk.Button(control_frame, text="Backward", width=15, command=lambda: send_command("Backward"), style="GitHub.TButton").grid(row=2, column=1, padx=5, pady=5)
ttk.Button(control_frame, text="Capture Face", width=15, command=capture_face, style="GitHub.TButton").grid(row=3, column=1, padx=5, pady=5)

# Create a status bar with dark theme styling
status_label = Label(root, text="Status: Idle", bd=1, relief=tk.SUNKEN,
                     anchor=tk.W, font=("Helvetica", 10), bg="#34495e", fg="white")
status_label.pack(side=tk.BOTTOM, fill=tk.X)

# Use your ESP32-CAM stream URL.
esp32_cam_url = "http://192.168.1.8/mjpeg/1"  # Now using the root URL which serves the webpage & stream
cap = cv2.VideoCapture(esp32_cam_url)

def update_video():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from stream.")
            time.sleep(0.1)
            continue

        # Debug: Print frame dimensions
        print("Frame shape:", frame.shape)
        latest_frame = frame.copy()  # Update the latest frame for capture
        
        # --- Face Recognition Processing ---
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(small_frame_rgb)
        face_encodings = face_recognition.face_encodings(small_frame_rgb)
        
        # Debug: Print number of faces detected
        print("Detected faces:", len(face_locations))
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(encodelist_known, face_encoding)
            face_distances = face_recognition.face_distance(encodelist_known, face_encoding)
            if len(face_distances) > 0:
                match_index = np.argmin(face_distances)
                if matches[match_index]:
                    name = classnames[match_index].upper()
                    top, right, bottom, left = face_location
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (left + 6, bottom - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # --- End Face Recognition Processing ---
        
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
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
