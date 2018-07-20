from camera import take_picture
import numpy as np
import matplotlib.image as mpimg

def cameraToArray():

    '''
    Takes a camera image and outputs it
    Input : none

    Output : 
    numpy array (H,W,C) of an image

    '''
    img = take_picture()
    return img

def jpgToArray(location):
    '''
    Takes a camera image and outputs it
    Input : 
    location : the absolute location of the JPEG image

    Output : 
    numpy array (H,W,C) of an image

    '''
    img = mpimg.imread(location)
    return img