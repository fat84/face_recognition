from face_recognition import EncodeImage
from face_recognition import Database_Logging_and_Matching
from face_recognition import imgToDescriptors
from face_recognition import UnsupervisedClassify
from pathlib import Path
import pickle
import numpy as np
from face_recognition import plot_rect
import time


def cameraToStorage(name):

    img = EncodeImage.cameraToArray()

    name_arrays, des_arrays, rects = imgToDescriptors.imgToDescriptors(img, name)

    Database_Logging_and_Matching.log_in_database(name_arrays, des_arrays)

    print("Stored image")

def cameraToMatch():
    
    img = EncodeImage.cameraToArray()

    name_arrays, des_arrays, rects = imgToDescriptors.imgToDescriptors(img)

    names = Database_Logging_and_Matching.match_against_database(des_arrays)

    plot_rect.plot_rect(img, names)

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

def classify(imgs):
    vectors = []
    for img in range(imgs):
        print("Taking image in 1 seconds")
        time.sleep(1)
        n_img = EncodeImage.take_picture()
        names, descriptions, rects = imgToDescriptors.imgToDescriptors(n_img)
        print(len(descriptions[0]))
        vectors.append(descriptions[0])
        print("Took image")
    dists = UnsupervisedClassify.computeDists(vectors, imgs)
    graph = UnsupervisedClassify.createGraph(dists, .45)
    graph = UnsupervisedClassify.finalizeLabels(graph)

    for node in graph.nodes:
        print("ID: ", node.ID, "  Label: ", node.label)