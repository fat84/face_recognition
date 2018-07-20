import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dlib_models import download_model, download_predictor, load_dlib_models
download_model()
download_predictor()
from dlib_models import models

load_dlib_models()
face_detect = models["face detect"]
face_rec_model = models["face rec"]
shape_predictor = models["shape predict"]

def plot_rect(img, names=[]):
    detections = list(face_detect(img))
    
    fig,ax = plt.subplots()
    ax.imshow(img)
    
    for i, d in enumerate(detections):
        x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
        face = patches.Rectangle((x1, y1), w, h, fill=None, lw=1, color=(1, 1, 1))
        ax.add_patch(face)
	if names != []:
        	plt.text(x1, y1, names[i], color=(1, 1, 1))
