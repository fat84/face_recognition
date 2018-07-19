from camera import take_picture
import numpy as np
from PIL import Image
import matplotlib.image as mpimg

def cameraToArray(name):
    pic = take_picture()
    return pic

def jpgToArray(location):
    img = mpimg.imread(location)
    return img
