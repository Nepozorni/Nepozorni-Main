from ultralytics import YOLO


def run_model(model_path: str, image_path: str) -> str:
    """Function runs YOLOv8 model and outputs debug string
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

    results = model(image_path) # run model

    r = results[0]

    class_names = r.names
    probs_tensor = r.probs.data # tensors (e.g '1', '20', '5' / 'one_hand')

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

    return output
