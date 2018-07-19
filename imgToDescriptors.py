import numpy as np
import matplotlib
from matplotlib.patches import Rectangle
from dlib_models import download_model, download_predictor, load_dlib_models
download_model()
download_predictor()
from dlib_models import models
load_dlib_models()
face_detect = models["face detect"]
face_rec_model = models["face rec"]
shape_predictor = models["shape predict"]

def imgToDescriptors(img_array, name=None):
    detections = list(face_detect(img_array))
    shape = shape_predictor(img_array, detections[0])
    descriptor = np.array(face_rec_model.compute_face_descriptor(img_array, shape))
    return (name, descriptor)