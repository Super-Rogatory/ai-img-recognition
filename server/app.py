import os, sys

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/ml"
)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from keras.preprocessing import image
from io import BytesIO
from PIL import Image
from skimage.transform import resize
from utils import test_image


app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:5000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="./client"), name="client")


templates = Jinja2Templates(directory="./client")


@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    # return index.html
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/checkpicture")
async def check_picture(file: UploadFile):
    # read image from html post, convert to numpy array, make prediction with model
    img = await file.read()
    if not img:
        return {"status": "No image present."}
    img = np.array(Image.open(BytesIO(img)))  # convert from byte array to numpy array
    img = resize(img, (32, 32))  # resize
    img_tensor = image.img_to_array(img)  # (height, width, channels)
    img_tensor = np.expand_dims(
        img_tensor, axis=0
    )  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.0
    prediction_message = test_image(img_tensor)
    return {"status": prediction_message}
