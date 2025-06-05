import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import os
import run_model
from datetime import datetime
import threading
import queue

file_path = None
video_player = None #za predvajanje videa
video_playing = False
image_player = None #prikaz slike 
video_label = None #referenca za box, kjer se bo prikazal video/slika
log_box = None #referenca za log box

model_ready = False
model_queue = queue.Queue() #omogoca komunikacijo med thread in UI

#klice ob kliku gumba load
def load_file():
    global video_player, video_playing, image_player, file_path, model_ready #spreminjanje globalih spremenljivk
    path = filedialog.askopenfilename(filetypes=[("Media files", "*.png *.jpg *.jpeg *.mp4 *.avi *.mov *.mkv")]) #odpremo file explorer
    if not path: #close file explorer
        return

    file_path_var.set(path) #zapis patha v textbox
    file_path = path
    model_ready = False

    #zapis v log
    ext = os.path.splitext(path)[-1].lower() #extension v lowercase za lazji comparison, predvajanje videa/prikaz slike
    if ext in [".mp4", ".avi", ".mov", ".mkv"]: #ce je datoteka video
        log(f"\"{path}\" loaded, type: video")
        start_video(path)

        model_thread = threading.Thread(target=worker, daemon=True) #zagon modelov
        model_thread.start()

        model_output()
    else: #ce je datoteka slika
        log(f"\"{path}\" loaded, type: picture")
        stop_video()
        display_image(path)

        _, model_hand_output = run_model.run_model("./Models/model-21-05-2025.pt", image_path=path) #zazene se model za roke in v gui se izpise rezultat
        hand_output.config(state="normal")
        hand_output.delete(1.0, tk.END)
        hand_output.insert(tk.END, model_hand_output)
        hand_output.config(state="disabled")

        _, full_head_output = run_model.run_model("./Models/face_30_epochs.pt", image_path=path) #zazene se model za glavo in rezultate zapise v gui

        #izpise samo max 5 tock z najvecjim probability
        head_lines = full_head_output.splitlines()
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
    global video_player, video_playing
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
    global video_player, image_player
    if not video_playing or not video_player:
        return

    ret, frame = video_player.read() #branje naslednjega frama iz videa
    if not ret: #ko doseze zadnji frame gre nazaj na zacetek
        video_player.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = video_player.read()
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
    global model_ready
    cap = cv2.VideoCapture(file_path)
    prvi_output = True #video se zacne predvajat ko model vrne prvi output

    while video_playing and cap.isOpened(): #beremo frame po frame iz videa
        ret, frame = cap.read()
        if not ret:
            break

        _, hand_out = run_model.run_model("./Models/model-21-05-2025.pt", image=frame) #klicanje modela za roke
        _, full_head_output = run_model.run_model("./Models/face_30_epochs.pt", image=frame) #klicanje modela za head

        #izpise samo max 5 tock z najvecjim probability za head model
        head_lines = full_head_output.splitlines()
        prediction_line = head_lines[0]
        inference_line = head_lines[-1]
        prob_lines = head_lines[2:-1][:5]

        head_output = f"{prediction_line}\n> TOP 5 PROBABILITIES:\n" + "\n".join(prob_lines) + f"\n{inference_line}"


        model_queue.put((hand_out, head_output)) #rezultat damo v queue

        if prvi_output: #model_ready damo na true, video se zacne predvajat
            model_ready = True
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
#log text
tk.Label(root, text="LOG", anchor='w', font=('Arial', 10, 'bold')).pack(fill=tk.X, padx=12, pady=(10, 0))

#log box
log_box = tk.Text(root, height=8, bg="lightgray", state="disabled")
log_box.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)

root.mainloop()