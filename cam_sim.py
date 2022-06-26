from flask import Flask, request, jsonify, render_template, Response, send_from_directory
import requests
import os
from numpy.random import randint
from time import sleep, time
from PIL import Image
from base64 import b64encode
import json

files = os.listdir("img/base")  # backend/ entfernt
i = 0
print(files)
while 0:
  for _ in range(20):
    start = time()
    file = files[randint(len(files)-1)]
    result = requests.get(f"http://127.0.0.1:5001/image/{file}")
    i += 1
    print(result, (time()-start), i)
    #sleep(0.1)
  sleep(10)


while 1:
  for _ in range(20):
    start = time()
    file = files[randint(len(files)-1)]
    image = open(f"img/base/{file}", "rb").read() # backend/ entfernt
    print(type(image))
    #img_enc = b64encode(image)
    img_info = json.dumps({"filename": file}).encode("UTF-8")
    result = requests.post("http://127.0.0.1:5001/api/receive_images", files={"image": image, "json": img_info}, headers={"content_type": "image/jpeg"})
    i += 1
    print(result, (time()-start), i)
    sleep(1)
  sleep(10)