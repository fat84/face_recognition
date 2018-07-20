import EncodeImage
import Database_Logging_and_Matching
import imgToDescriptors
from pathlib import Path
import pickle
import numpy as np
import plot_rect


def cameraToStorage(name):

    img = EncodeImage.cameraToArray()

    name_arrays, des_arrays, rects = imgToDescriptors.imgToDescriptors(img, name)

    Database_Logging_and_Matching.log_in_database(name_arrays, des_arrays)

    print("Stored image")

def cameraToMatch():
    
    img = EncodeImage.cameraToArray()

    name_arrays, des_arrays, rects = imgToDescriptors.imgToDescriptors(img)

    names = Database_Logging_and_Matching.match_against_database(des_arrays)

    plot_rect.plot_rect(img)

    print(names)

def seeDictionaryEntry(name = None):
    file_path = Path.home()
    with open(file_path/"names_and_faces.pkl", mode = "rb") as opened_file:
        names_and_faces = pickle.load(opened_file)
    print(names_and_faces)

def reset():
    file_path = Path.home()
    with open(file_path/"names_and_faces.pkl", mode = "wb") as opened_file:
        names_and_faces = {}
        pickle.dump(names_and_faces, opened_file)
        print("Reset Dictionary")


#cameraToStorage("Vedantha")
cameraToMatch()
#seeDictionaryEntry("Big V")
#reset()
