from flask import Flask, request, jsonify, render_template, Response, send_from_directory
import json
import requests
from flask_sqlalchemy import SQLAlchemy
from multiprocessing.pool import ThreadPool
import numpy as np

from PIL import Image
import torch
import torchvision
import  modules.utils as ut
from time import time
import io


# Function definitions
def save_to_db(data: dict):

    print("Saving to db!")
    image_data = Images()
    print("dada")
    for key, value in data.items():
        setattr(image_data, key, value)

    db.add(image_data)
    db.commit()

def handle_results(latest_results: dict, results: dict):
    
    # handles incoming results
    # if car is valid (no converitble)
    # -> save results to 
    # -> update screen
    # else delete picture

    print(f"Handling request results for {results['image_path']}")

    # save data and update front end
    if results["valid_car"] == 1:
        #save_to_db(results)
        latest_results = update_results(latest_results, results)

    # delete image
    else:
        print("Deleting image (not implemented)")

def update_results(latest_results: dict, results: dict):

    # update results
    for key, value in results.items():
        if key in latest_results[results["camera_name"]].keys():
            latest_results[results["camera_name"]][key] = value

    return latest_results

def event_stream():
    while True:
        yield f"data: {latest_results}"


# Initialize 

app = Flask(__name__, static_url_path='/static')
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///results.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
pool = ThreadPool(processes=8)

db = SQLAlchemy(app)
class Images(db.Model):

    _id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String, unique=True, nullable=False)
    camera_name = db.Column(db.String)
    timestamp = db.Column(db.DateTime)
    valid_car = db.Column(db.Integer)
    y_min = db.Column(db.Integer)
    y_max = db.Column(db.Integer)
    x_min = db.Column(db.Integer)
    x_max = db.Column(db.Integer)
    predicted = db.Column(db.Integer)
    probability = db.Column(db.Float)

    def __repr__(self):
       return f"{self.image_name}, {self.camera_name}, {self.valid_car}, {self.y_min}, {self.y_max}, {self.x_min}, {self.x_max}, {self.predicted}, {self.probability}"

latest_results = {"caml1c1": {"image_path": None, "prediction": None, "id": None, "timestamp": None, "probability": None},
                    "caml1c2": {"image_path": None, "prediction": None, "id": None, "timestamp": None, "probability": None},
                    "caml2c2": {"image_path": None, "prediction": None, "id": None, "timestamp": None, "probability": None},
                    "picamera": {"image_path": None, "prediction": None, "id": None, "timestamp": None, "probability": None}}

@app.route("/")
def index():
    return "Server running"


@app.route("/api/data")
def UI():
    return latest_results


@app.route("/stream")
def stream():
    Response(event_stream ,mimetype="text/event-stream")


@app.route("/api/<camera_name>")
def provide_images(camera_name):
    return render_template("images.html", response="../" + latest_results[camera_name]["image_path"])


@app.route("/image/<path:filename>")
def test(filename):

    image_path = (f"static/base/{filename}")   # backend/ entfernt
    #result = threading.Thread(target=send_full_request, args=(image_path,)).start()
    async_result = pool.apply_async(eval_full, (image_path,))
    _ = pool.apply_async(handle_results, (latest_results, async_result.get()))

    return Response(status=200)


@app.route("/api/receive_images", methods=["POST"])
def receive_images():

    '''
    Get images from the client, save them locally and queue tasks:
    1. model inference
    2. make data accessible to UI & create db entry     
    '''

    # receive bytes image, decode and store it
    image_bytes = request.files["image"].read()
    json_bytes = request.files["json"].read()
    
    img_info = json.load(io.BytesIO(json_bytes))
    img = Image.open(io.BytesIO(image_bytes))
    image_path = f"static/send/{img_info['filename']}" # backend/ entfernt
    img.save(image_path)

    # queue tasks for further processing
    async_result = pool.apply_async(eval_full, (image_path,))
    _ = pool.apply_async(handle_results, (latest_results, async_result.get()))

    return Response(status=200)


'''ML Model & Functions'''

# define functions
def get_yolo_labels(points: torch.tensor) -> dict:

    #get all x and y points from the labels and take min and max
    x = np.array([points[0, 0], points[0, 2], points[1, 0], points[1, 2]])
    x = np.array([min(x), max(x)])
    y = np.array([points[0, 1], points[0, 3], points[1, 1], points[1, 3]])
    y = np.array([min(y), max(y)])
    
    # check if:
    #   1. exactly two points are found
    #   2. both have a different label

    if all((points.shape[0] == 2, torch.sum(points[:,-1]) == 1)):
        
        output = {"valid_car": 1,
                "x_min": int(x[0]),
                "x_max": int(x[1]),
                "y_min": int(y[0]),
                "y_max": int(y[1])}
    
    else:
        output = {"valid_car": 0,
                "x_min": -1,
                "x_max": -1,
                "y_min": -1,
                "y_max": -1}

    return output

def get_image_metadata(image_path: str)-> dict:
    
    camera_name = (image_path.split("/")[-1]).split("-")[0]
    data = {"image_path": image_path,
            "camera_name": camera_name,
            "timestamp": time()}

    return data

def eval_full(image_path):

    # Run yolo model
    output = model_yolo(image_path)
    result = get_yolo_labels(output.xyxy[0])


    # run resnet if the roi is found
    if result["valid_car"] == 1:

        img = Image.open(image_path)
        img = np.array(img)
        img = img[result["y_min"]:result["y_max"], result["x_min"]: result["x_max"]]

        preprocess = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img = preprocess(img)
        img = img.unsqueeze(0)
        img = img.to("cpu")

        output = model_resnet(img)

        softmax = torch.nn.Softmax(dim=1)
        probability = softmax(output.data)
        probability, prediction = torch.max(probability, 1)

        resnet_results = {"prediction": int(prediction),
                        "probability": float(probability)}

    else:
        resnet_results = {"prediction": -1,
                        "probability": -1}
    

    # concat the data and return it
    metadata = get_image_metadata(image_path)

    result = dict(metadata, **result)
    result = dict(result, **resnet_results)

    return result


# init models
device = "cpu"

MODEL_YOLO_PATH = "modules/models/yolov5"
WEIGHTS_YOLO_PATH = "modules/models/yolo_weights.pt"

WEIGHTS_RESNET_PATH = "modules/models/classifier_resnet18.pt"

model_yolo = ut.load_model(MODEL_YOLO_PATH, WEIGHTS_YOLO_PATH).to(device)
model_yolo.eval()

model_resnet = ut.ResnetModel(WEIGHTS_RESNET_PATH, False, 2).to(device)
model_resnet.eval()

'''Run App'''
if __name__ == "__main__":
    app.run(debug=1, threaded=True) #port=5001 rausgeschmissen, Fehler bei Start des Containers mit Docker - Kein Zugriff auf die Website