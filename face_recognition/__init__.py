
from face_recognition import EncodeImage
from face_recognition import Database_Logging_and_Matching
from face_recognition import imgToDescriptors
from face_recognition import UnsupervisedClassify
from pathlib import Path
#from face_recognition import plot_rect
import pickle
import time

print("import finished")
def cameraToStorage(name):
    """
    Takes a picture of the person and stores it in the computer's pickle file.
    Parameters:
    --------
    name: str
        The name of the person whose image is to be stored.
    Returns:
    ---------
    truth: Boolean
        1 if a face was detected and 0 if a face was not.
    """

    img = EncodeImage.cameraToArray()

    name_arrays, des_arrays, rects = imgToDescriptors.imgToDescriptors(img, name)

    truth = Database_Logging_and_Matching.log_in_database(name_arrays, des_arrays)

    return truth
    #print("Stored image")

def cameraToMatch():
    """
    Match peoples' faces against an existing database.
    Parameters
    ---------
    Returns
    ---------
    names: np.ndarray
        The array of names being returned.
    """
    
    img = EncodeImage.cameraToArray()

    name_arrays, des_arrays, rects = imgToDescriptors.imgToDescriptors(img)

    names = Database_Logging_and_Matching.match_against_database(des_arrays)

    #plot_rect.plot_rect(img, names)

    #print(names)
    return names

def seeDictionaryEntry():
    """
    See all dictionary entries.
    Parameters:
    ---------
    Returns:
    ---------
    It does not return the dictionary, but rather prints it to the console.
    """
    file_path = Path.home()
    with open(file_path/"names_and_faces.pkl", mode = "rb") as opened_file:
        names_and_faces = pickle.load(opened_file)
    print(names_and_faces)

def reset():
    """
    Resets the dictionary (tabula rasa)
    No params or returns
    """
    file_path = Path.home()
    with open(file_path/"names_and_faces.pkl", mode = "wb") as opened_file:
        names_and_faces = {}
        pickle.dump(names_and_faces, opened_file)
        #print("Reset Dictionary")

def classify(imgs):
    """
    Classifies the images into distinct groups using the whisper algorithm.
    Parameters:
    ---------
    imgs: int
        The number of pictures the computer should take.
    Returns:
    ---------
    Does not return, but rather prints nodes and their values.
    """
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

    for node in graph.nodes.values():
        print("ID: ", node.ID, "  Label: ", node.label)


