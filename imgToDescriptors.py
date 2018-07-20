import numpy as np
import matplotlib
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from dlib_models import download_model, download_predictor, load_dlib_models
download_model()
download_predictor()
from dlib_models import models
load_dlib_models()
face_detect = models["face detect"]
face_rec_model = models["face rec"]
shape_predictor = models["shape predict"]

def imgToDescriptors(img_array, name=None):
    des_arrays = []
    name_arrays = []
    detections = list(face_detect(img_array))
    for i in range(len(detections)):
        shape = shape_predictor(img_array, detections[i])
        descriptor = np.array(face_rec_model.compute_face_descriptor(img_array, shape))
        if name is not None:
            name_arrays.append(name)
            des_arrays.append(descriptor)
        else:
            name_arrays.append(None)
            des_arrays.append(descriptor)

    print(len(detections))
    return (name_arrays, des_arrays, detections)
    
        
