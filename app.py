import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import os
import run_model
from datetime import datetime

file_path = None
video_player = None #za predvajanje videa
video_playing = False
image_player = None #prikaz slike 
video_label = None #referenca za box, kjer se bo prikazal video/slika
log_box = None #referenca za log box


#klice ob kliku gumba load
def load_file():
    global video_player, video_playing, image_player, file_path #spreminjanje globalih spremenljivk

    path = filedialog.askopenfilename(filetypes=[("Media files", "*.png *.jpg *.jpeg *.mp4 *.avi *.mov *.mkv")]) #odpremo file explorer
    if not path: #close file explorer
        return

    file_path_var.set(path) #zapis patha v textbox
    file_path = path

    #zapis v log
    ext = os.path.splitext(path)[-1].lower() #extension v lowercase za lazji comparison, predvajanje videa/prikaz slike
    if ext in [".mp4", ".avi", ".mov", ".mkv"]:
        log(f"\"{path}\" loaded, type: video")
        start_video(path)
    else:
        log(f"\"{path}\" loaded, type: picture")
        stop_video()
        display_image(path)

    # model za roke
    if ext in [".mp4", ".avi", ".mov", ".mkv"]:
        cap = cv2.VideoCapture(path)

        # Preveri, če je video naložen
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        while True:
            ret, frame = cap.read()

            # Ko se video konča
            if not ret:
                break

            # Preberi frame
            model_hand_prediction, model_hand_output = run_model.run_model("./Models/model-21-05-2025.pt", image=frame)
            hand_output.config(state="normal")
            hand_output.delete(1.0, tk.END)
            hand_output.insert(tk.END, model_hand_output)
            hand_output.config(state="disabled")

            model_head_prediction, model_head_output = run_model.run_model("./Models/face_30_epochs.pt", image=frame)
            head_output.config(state="normal")
            head_output.delete(1.0, tk.END)
            head_output.insert(tk.END, model_head_output)
            head_output.config(state="disabled")



    else:
        # Preberi sliko
        model_hand_prediction, model_hand_output = run_model.run_model("./Models/model-21-05-2025.pt", image_path=path)
        hand_output.config(state="normal")
        hand_output.delete(1.0, tk.END)
        hand_output.insert(tk.END, model_hand_output)
        hand_output.config(state="disabled")

        model_head_prediction, model_head_output = run_model.run_model("./Models/face_30_epochs.pt", image_path=path)
        #best.pt vrne error, ker je YOLOv5, pa naj bi mogu bit z novejsim yolo modelom, takda sm dau kr svojga.
        head_output.config(state="normal")
        head_output.delete(1.0, tk.END)
        head_output.insert(tk.END, model_head_output)
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
    update_video()


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

#---------------------WINDOW SETUP---------------------  
root = tk.Tk()
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