import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import cv2
import os
import run_model
from datetime import datetime
import threading
import queue
from prometheus_client import Counter, Gauge
import GPUtil
import psutil
import paho.mqtt.client as mqtt
import json
import socket
import time

file_path = None
video_player = None #za predvajanje videa
video_playing = False
image_player = None #prikaz slike 
video_label = None #referenca za box, kjer se bo prikazal video/slika
log_box = None #referenca za log box
video_active = False
model_ready = False

model_queue = queue.Queue() #omogoca komunikacijo med thread in UI

# Prometheus
frame_count_hand = Counter('processed_frames_hand_total', 'Stevilo obdelanih frames modela glave')
frame_count_head = Counter('processed_frames_head_total', 'Stevilo obdelanih frames modela roke')
cpu_usage = Gauge('cpu_usage_percent', 'Obremenitev CPU')
gpu_usage = Gauge('gpu_usage_percent', 'Obremenitev GPU')

ip_address = socket.gethostbyname(socket.gethostname())
mqtt_client = mqtt.Client(client_id=ip_address)
mqtt_client.connect("localhost", 1883, 60)
mqtt_client.loop_start()

def monitor_usage():
    global frame_count_hand, frame_count_head

    while True:
        if not video_active:
            time.sleep(0.5)
            continue

        cpu = psutil.cpu_percent(interval=None)
        try:
            gpus = GPUtil.getGPUs()
            gpu = gpus[0].load * 100 if gpus else 0
        except:
            gpu = 0

        data = {
            "cpu usage": cpu,
            "gpu usage": gpu,
            "frames_hand_total": frame_count_hand._value.get(),
            "frames_head_total": frame_count_head._value.get(),
        }
        mqtt_client.publish("/metrike", json.dumps(data), qos=1)

        #log(f"[DEBUG] skupno: roka={frame_count_hand}, glava={frame_count_head}")

        time.sleep(1) #posilja na eno sekundo

def update_boxes_on_image(prob_array):
    try:
        updated_img = Image.open("tocke.png").resize((721, 282))
        draw = ImageDraw.Draw(updated_img, "RGBA")

        for box_id, point_ids in boxes.items():
            xs = [points[i][0] for i in point_ids]
            ys = [points[i][1] for i in point_ids]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # Sum of probabilities for points in this box
            prob_sum = sum(prob_array[i - 1] for i in point_ids if 0 <= i - 1 < len(prob_array))
            alpha = int(min(255, max(0, prob_sum * 255)))  # Clamp between 0 and 255

            draw.rectangle(
                [min_x - 5, min_y - 5, max_x + 5, max_y + 5],
                fill=(0, 255, 0, alpha),
                outline=(0, 128, 0)
            )

        tocke_img_tk_updated = ImageTk.PhotoImage(updated_img)
        image_label.configure(image=tocke_img_tk_updated)
        image_label.image = tocke_img_tk_updated

    except Exception as e:
        log(f"ERROR: Failed to update image: {str(e)}")


#klice ob kliku gumba load
def load_file():
    global video_player, video_playing, image_player, file_path, model_ready, frame_count_hand, frame_count_head #spreminjanje globalih spremenljivk
    frame_count_hand._value.set(0)
    frame_count_head._value.set(0)

    path = filedialog.askopenfilename(filetypes=[("Media files", "*.png *.jpg *.jpeg *.mp4 *.avi *.mov *.mkv")]) #odpremo file explorer
    if not path: #close file explorer
        return

    file_path_var.set(path) #zapis patha v textbox
    file_path = path
    model_ready = False

    ext = os.path.splitext(path)[-1].lower() #extension v lowercase za lazji comparison, predvajanje videa/prikaz slike
    if ext in [".mp4", ".avi", ".mov", ".mkv"]: #ce je datoteka video
        log(f"\"{path}\" loaded, type: video")
        start_video(path)

        # Samo zaženi thread, ki bo frame-e procesiral v ozadju
        model_thread = threading.Thread(target=worker, daemon=True) #zagon modelov
        model_thread.start()

        model_output()
    else: #ce je datoteka slika
        log(f"\"{path}\" loaded, type: picture")
        stop_video()
        display_image(path)
        frame_count_hand = 1
        frame_count_head = 1

        head_probabilities = [0.0] * 27

        _, model_hand_output = run_model.run_model("./Models/model-21-05-2025.pt", image_path=path) #zazene se model za roke in v gui se izpise rezultat
        hand_output.config(state="normal")
        hand_output.delete(1.0, tk.END)
        hand_output.insert(tk.END, model_hand_output)
        hand_output.config(state="disabled")

        _, model_head_output = run_model.run_model("./Models/face_30_epochs.pt", image_path=path, prob_array=head_probabilities) #zazene se model za glavo in rezultate zapise v gui
        update_boxes_on_image(head_probabilities)

        #izpise samo max 5 tock z najvecjim probability
        head_lines = model_head_output.splitlines()
        prediction_line = head_lines[0]
        inference_line = head_lines[-1]
        prob_lines = head_lines[2:-1][:5]

        head_output_text = f"{prediction_line}\n> TOP 5 PROBABILITIES:\n" + "\n".join(prob_lines) + f"\n{inference_line}"

        head_output.config(state="normal")
        head_output.delete(1.0, tk.END)
        head_output.insert(tk.END, head_output_text)
        head_output.config(state="disabled")

#ko nalozimo video se ga runna
def start_video(path):
    global video_player, video_playing, video_active
    video_active = True
    stop_video()
    video_player = cv2.VideoCapture(path)
    if not video_player.isOpened():
        log(f"ERROR: Failed to open video \"{path}\"")
        return
    video_playing = True
    update_video.running = False #najprej pocakamo da se modeli inicializirajo, ker cene video runna in outputov ni

#ustavimo predvajanje videa
def stop_video():
    global video_player, video_playing
    video_playing = False

    if video_player:
        video_player.release()
        video_player = None

#predvajanje videa
def update_video():
    global video_player, image_player, video_active
    if not video_playing or not video_player:
        return

    ret, frame = video_player.read() #branje naslednjega frama iz videa
    if not ret: #ko doseze zadnji frame gre nazaj na zacetek
        stop_video()
        video_active = False
        return
    if ret:
        frame = cv2.resize(frame, (640, 360)) #resize da se lepo prilega
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #bgr -> rgb
        img = Image.fromarray(frame) #numpy array -> pillow pic
        image_player = ImageTk.PhotoImage(img) #pillow pic -> format, ki ga tkinter razume
        video_label.configure(image=image_player, text="") #posodabljanje labela z novim framom
    root.after(33, update_video) #prikazujemo 30 fps

#prikaz slike, isti postopek kot prej
def display_image(path):
    global image_player
    try:
        img = Image.open(path).resize((640, 360))
        image_player = ImageTk.PhotoImage(img)
        video_label.configure(image=image_player, text="")
    except Exception as e:
        log(f"ERROR: Failed to open image \"{path}\": {str(e)}")

#izpis logov v log
def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_box.config(state="normal")
    log_box.insert(tk.END, f"[{timestamp}] {message}\n")
    log_box.config(state="disabled")
    log_box.see(tk.END)

def worker():
    global model_ready, video_active
    cap = cv2.VideoCapture(file_path)
    global frame_count_hand, frame_count_head

    prvi_output = True  #video se zacne predvajat ko model vrne prvi output
    head_probabilities = [0.0] * 27

    while video_playing and cap.isOpened(): #beremo frame po frame iz videa
        ret, frame = cap.read()
        if not ret:
            video_active = False
            break

        _, hand_out = run_model.run_model("./Models/model-21-05-2025.pt", image=frame) #klicanje modela za roke
        if hand_out is not None:
            frame_count_hand.inc()

        _, full_head_output = run_model.run_model("./Models/face_30_epochs.pt", image=frame, prob_array=head_probabilities) #klicanje modela za head
        if full_head_output is not None:
            frame_count_head.inc()

        update_boxes_on_image(head_probabilities)

        #izpise samo max 5 tock z najvecjim probability za head model
        head_lines = full_head_output.splitlines()
        prediction_line = head_lines[0]
        inference_line = head_lines[-1]
        prob_lines = head_lines[2:-1][:5]

        head_output = f"{prediction_line}\n> TOP 5 PROBABILITIES:\n" + "\n".join(prob_lines) + f"\n{inference_line}"

        model_queue.put((hand_out, head_output)) #rezultat damo v queue

        if prvi_output: #model_ready damo na true, video se zacne predvajat
            model_ready = True
            threading.Thread(target=monitor_usage, daemon=True).start()
            prvi_output = False

    cap.release()

def model_output():
    global model_ready
    try:
        while True:
            hand_out, head_out = model_queue.get_nowait() #preverimo, ce je kej v queue, ce je, izpisemo

            #izpis v UI
            hand_output.config(state="normal")
            hand_output.delete(1.0, tk.END)
            hand_output.insert(tk.END, hand_out)
            hand_output.config(state="disabled")

            head_output.config(state="normal")
            head_output.delete(1.0, tk.END)
            head_output.insert(tk.END, head_out)
            head_output.config(state="disabled")

            if not update_video.running and model_ready: #ko dobimo prvi output, zazenemo video
                update_video.running = True
                update_video()
    except queue.Empty:
        pass
    root.after(33, model_output) #klicemo vsakih 33ms

#---------------------WINDOW SETUP---------------------

root = tk.Tk() #ustvarimo 
root.title("Nadzor pozornosti")
root.geometry("1200x800")

file_path_var = tk.StringVar() #"textbox" kjer se bo izpisal path

#------ZGORNJI DEL------
#select file text + "textbox" + button load
top_frame = tk.Frame(root)
top_frame.pack(pady=10)

tk.Label(top_frame, text="Select file:").pack(side=tk.LEFT, padx=5) #select file label
file_entry = tk.Entry(top_frame, textvariable=file_path_var, width=80, state="readonly") #textbox z potjo do fajla
file_entry.pack(side=tk.LEFT, padx=5)
tk.Button(top_frame, text="Load", command=load_file).pack(side=tk.LEFT, padx=5) #button load

#------SREDNJI DEL------
#prikazovalnik videa + output od modelov
middle_frame = tk.Frame(root)
middle_frame.pack(pady=20)

#video player
video_container = tk.Frame(middle_frame, width=640, height=360, bg="black")
video_container.pack(side=tk.LEFT, padx=10)
video_container.pack_propagate(False)

video_label = tk.Label(video_container, text="NO VIDEO/PICTURE LOADED", bg="lightgray")
video_label.pack(fill="both", expand=True)

#head model output box
head_frame = tk.Frame(middle_frame)
head_frame.pack(side=tk.LEFT, padx=10)

#head model output text
tk.Label(head_frame, text="HEAD MODEL OUTPUT", font=('Arial', 10, 'bold')).pack()
head_output = tk.Text(head_frame, width=30, height=20, bg="lightgray", state="disabled")
head_output.pack()
head_output.config(state="normal")
head_output.config(state="disabled")

#hand model output box
hand_frame = tk.Frame(middle_frame)
hand_frame.pack(side=tk.LEFT, padx=10)

#hand model output text
tk.Label(hand_frame, text="HAND MODEL OUTPUT", font=('Arial', 10, 'bold')).pack()
hand_output = tk.Text(hand_frame, width=30, height=20, bg="lightgray", state="disabled")
hand_output.pack()
hand_output.config(state="normal")
hand_output.config(state="disabled")

#------SPODNJI DEL------
bottom_frame = tk.Frame(root)
bottom_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)

# left: tocke image
image_frame = tk.Frame(bottom_frame)
image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

points = {
    1: (30, 180), 19: (100, 60),
    15: (550, 90), 20: (600, 165),
    2: (130, 15), 3: (240, 15), 4: (380, 45), 18: (420, 85),
    5: (220, 80), 6: (290, 105), 7: (340, 140), 8: (160, 150), 9: (250, 150), 21: (395, 70),
    10: (140, 155), 11: (275, 155), 12: (180, 190), 13: (400, 170), 14: (270, 240),
    22: (435, 150), 23: (430, 170), 24: (550, 55), 25: (470, 180), 26: (490, 180),
    16: (70, 220), 17: (640, 220), 27: (360, 265)
}

# Groupings: box number -> list of point indices
boxes = {
    1: [1, 19],
    2: [15, 20],
    3: [2, 3, 4, 18],
    4: [5, 6, 7, 8, 9, 21],
    5: [10, 11, 12, 13, 14],
    6: [22, 23, 24, 25, 26],
    7: [16],
    8: [17],
    9: [27]
}

# Open and draw image
try:
    tocke_img = Image.open("tocke.png").resize((721, 282))
    draw = ImageDraw.Draw(tocke_img, "RGBA")

    for box_points in boxes.values():
        xs = [points[i][0] for i in box_points]
        ys = [points[i][1] for i in box_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        draw.rectangle([min_x - 5, min_y - 5, max_x + 5, max_y + 5], fill=(0, 255, 0, 100), outline=(0, 128, 0))

    tocke_img_tk = ImageTk.PhotoImage(tocke_img)
    image_label = tk.Label(image_frame, image=tocke_img_tk)
    image_label.image = tocke_img_tk
    image_label.pack()
except Exception as e:
    image_label = tk.Label(image_frame, text=f"Failed to load tocke.png\n{e}", fg="red")
    image_label.pack()

# right: log box
log_frame = tk.Frame(bottom_frame)
log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

tk.Label(log_frame, text="LOG", anchor='w', font=('Arial', 10, 'bold')).pack(fill=tk.X)

log_box = tk.Text(log_frame, height=8, bg="lightgray", state="disabled")
log_box.pack(fill=tk.BOTH, expand=True)

root.mainloop()