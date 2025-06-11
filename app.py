import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import cv2
import os
from run_model import run_model
from evaluate import *
from datetime import datetime
import threading
import queue

file_path = None
video_player = None #za predvajanje videa
video_playing = False
image_player = None #prikaz slike 
video_label = None #referenca za box, kjer se bo prikazal video/slika
log_box = None #referenca za log box
video_active = False
model_ready = False

model_queue = queue.Queue() #omogoca komunikacijo med thread in UI

def update_boxes_on_image(prob_array):
    try:
        # Load base image
        base_img = Image.open("tocke.jpg").resize((600, 282)).convert("RGBA")

        # Create transparent overlay
        overlay = Image.new("RGBA", base_img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        for box_id, point_ids in boxes.items():
            xs = [points[i][0] for i in point_ids]
            ys = [points[i][1] for i in point_ids]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            prob_sum = prob_array[box_id - 1]
            prob_clamped = min(1.0, max(0.0, prob_sum))

            alpha = int(prob_clamped * 150)  # max alpha = 150 (more see-through)

            draw.rectangle(
                [min_x - 5, min_y - 5, max_x + 5, max_y + 5],
                fill=(0, 255, 0, alpha),
                outline=(0, 128, 0, 255)
            )

        # Composite overlay onto base image
        combined = Image.alpha_composite(base_img, overlay)

        tocke_img_tk_updated = ImageTk.PhotoImage(combined)
        image_label.configure(image=tocke_img_tk_updated)
        image_label.image = tocke_img_tk_updated

    except Exception as e:
        log(f"ERROR: Failed to update image: {str(e)}")


#klice ob kliku gumba load
def load_file():
    global video_player, video_playing, image_player, file_path, model_ready, frame_count_hand, frame_count_head #spreminjanje globalih spremenljivk

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

        model_thread = threading.Thread(target=worker, daemon=True) #zagon modelov
        model_thread.start()

        model_output()
    else: #ce je datoteka slika
        log(f"\"{path}\" loaded, type: picture")
        stop_video()
        display_image(path)
        frame_count_hand = 1
        frame_count_head = 1

        head_probabilities = [0.0] * 9

        _, model_hand_output = run_model("./Models/model-21-05-2025.pt", image_path=path) #zazene se model za roke in v gui se izpise rezultat
        hand_output.config(state="normal")
        hand_output.delete(1.0, tk.END)
        hand_output.insert(tk.END, model_hand_output)
        hand_output.config(state="disabled")

        _, model_head_output = run_model.run_model("./Models/boxesmodel50epochs.pt", image_path=path, prob_array=head_probabilities) #zazene se model za glavo in rezultate zapise v gui
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

    head_probabilities = [0.0] * 9

    prvi_output = True #video se zacne predvajat ko model vrne prvi output
    head_pred = ""
    hand_pred = ""
    seconds_head_pred = 0
    seconds_hand_pred = 0
    i = 0
    while video_playing and cap.isOpened(): #beremo frame po frame iz videa
        ret, frame = cap.read()
        if not ret:
            video_active = False
            break

        _, hand_out = run_model("./Models/model-21-05-2025.pt", image=frame) #klicanje modela za roke

        _, full_head_output = run_model("./Models/boxesmodel50epochs.pt", image=frame, prob_array=head_probabilities) #klicanje modela za head

        update_boxes_on_image(head_probabilities)
        n_hand_pred, hand_out = run_model("./Models/model-21-05-2025.pt", image=frame) #klicanje modela za roke
        n_head_pred, full_head_output = run_model("./Models/boxesmodel50epochs.pt", image=frame) #klicanje modela za head

        if head_pred != n_head_pred: # handle napoved rok
            head_pred = n_head_pred
            seconds_head_pred = 0
        elif i % 30 == 0: # napoved je enaka kot ena sekunda nazaj
            seconds_head_pred += 1

        if hand_pred != n_hand_pred:
            hand_pred = n_hand_pred
            seconds_hand_pred = 0
        elif i % 30 == 0:
            seconds_hand_pred += 1

        # kliči evalvacijsko funkcijo
        assessment = evaluate(head_pred, hand_pred, seconds_head_pred, seconds_hand_pred)

        #izpise samo max 5 tock z najvecjim probability za head model
        head_lines = full_head_output.splitlines()
        prediction_line = head_lines[0]
        inference_line = head_lines[-1]
        prob_lines = head_lines[2:-1][:5]

        head_output = f"{prediction_line}\n> TOP 5 PROBABILITIES:\n" + "\n".join(prob_lines) + f"\n{inference_line}"

        model_queue.put((hand_out, head_output, str(assessment), str(100-assessment))) #rezultat damo v queue

        if prvi_output: #model_ready damo na true, video se zacne predvajat
            model_ready = True
            prvi_output = False

        i += 1
    cap.release()

def model_output():
    global model_ready
    try:
        while True:
            hand_out, head_out, assessment, penalty = model_queue.get_nowait() #preverimo, ce je kej v queue, ce je, izpisemo
            #izpis v UI
            hand_output.config(state="normal")
            hand_output.delete(1.0, tk.END)
            hand_output.insert(tk.END, hand_out)
            hand_output.config(state="disabled")

            head_output.config(state="normal")
            head_output.delete(1.0, tk.END)
            head_output.insert(tk.END, head_out)
            head_output.config(state="disabled")

            assessment_var.set(f"{assessment}%")
            penalty_var.set(f"{penalty}%")

            if not is_attentive(float(assessment)):
                attentive_var.set("INNATENTIVE")
                attentive_label.config(bg="red")
            else:
                attentive_var.set("ATTENTIVE")
                attentive_label.config(bg="lightgreen")

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

assessment_var = tk.StringVar()
penalty_var = tk.StringVar()
attentive_var = tk.StringVar()
attentive_var.set("ATTENTIVE")
assessment_var.set("0%")
penalty_var.set("0%")

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

#--------DEL OCEN-------
eval_frame = tk.Frame(root)
eval_frame.pack(fill=tk.BOTH, padx=10, pady=10)

# to naredi, da je enakomerno porazdeljeno po dolzini
eval_frame.grid_columnconfigure(0, weight=1)
eval_frame.grid_columnconfigure(1, weight=1)
eval_frame.grid_columnconfigure(2, weight=1)

# frame za pozornost
att_frame = tk.Frame(eval_frame)
att_frame.grid(row=0, column=0, padx=20)

tk.Label(att_frame, text="Attentiveness:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w')
eval_label = tk.Label(att_frame, textvariable=assessment_var, font=('Arial', 20, 'bold'))
eval_label.grid(row=1, column=0)

# frame za odštevek
penalty_frame = tk.Frame(eval_frame)
penalty_frame.grid(row=0, column=1, padx=20)

tk.Label(penalty_frame, text="Penalty:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w')
penalty_label = tk.Label(penalty_frame, textvariable=penalty_var, font=('Arial', 20, 'bold'))
penalty_label.grid(row=1, column=0)

# label pozornosti
attentive_label = tk.Label(eval_frame, textvariable=attentive_var, font=('Arial', 20, 'bold'), fg="black", bg="lightgreen", pady=2)
attentive_label.grid(row=0, column=2, padx=20)

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

boxesModel = {
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6],
    7: [7],
    8: [8],
    9: [9]
}

# Open and draw image
try:
    tocke_img = Image.open("tocke.jpg").resize((600, 282))
    draw = ImageDraw.Draw(tocke_img, "RGBA")

    for box_points in boxes.values():
        xs = [points[i][0] for i in box_points]
        ys = [points[i][1] for i in box_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        draw.rectangle([min_x - 5, min_y - 5, max_x + 5, max_y + 5], fill=(0, 255, 0, 40), outline=(0, 128, 0))

    tocke_img_tk = ImageTk.PhotoImage(tocke_img)
    image_label = tk.Label(image_frame, image=tocke_img_tk)
    image_label.image = tocke_img_tk
    image_label.pack()
except Exception as e:
    image_label = tk.Label(image_frame, text=f"Failed to load tocke.jpg\n{e}", fg="red")
    image_label.pack()

# right: log box
log_frame = tk.Frame(bottom_frame)
log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

tk.Label(log_frame, text="LOG", anchor='w', font=('Arial', 10, 'bold')).pack(fill=tk.X)

log_box = tk.Text(log_frame, height=8, bg="lightgray", state="disabled")
log_box.pack(fill=tk.BOTH, expand=True)

root.mainloop()