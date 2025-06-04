from ultralytics import YOLO

def run_model(model_path: str, image_path: str = None, image = None, prob_array: list = None) -> tuple:
    """Function runs YOLOv8 model and returns top prediction and debug string
        SHAPE:
        > PREDICTION: p
        > PROBABILITY:
        p1 : 0.n
        p2 : 0.m
        p3 : 0.o
        (where n > m > o)
        > INFERENCE TIME: T ms
    """

    model = YOLO(model_path) # set model

    if image is None:
        results = model(image_path) # run model

    if image_path is None:
        results = model.predict(image)

    r = results[0]

    class_names = r.names
    probs_tensor = r.probs.data # tensors (e.g '1', '20', '5' / 'one_hand', ...)

    if prob_array is not None and len(prob_array) == 27:
        for i in range(27):
            prob_array[i] = float(probs_tensor[i])

    # fetch probabilites
    prob_dict = {class_names[i]: float(p) for i, p in enumerate(probs_tensor)}

    # sort probabilites
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

    # get top prediction
    top_prediction = sorted_probs[0][0]

    # get inference time
    inference_time_ms = r.speed['inference']

    # make output
    output = f"> PREDICTION: {top_prediction}\n> PROBABILITY:\n"
    output += "\n".join([f"{k}: {v:.2f}" for k, v in sorted_probs])
    output += f"\n> INFERENCE TIME: {inference_time_ms:.1f} ms"

    return top_prediction, output
