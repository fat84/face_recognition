from pathlib import Path
import numpy as np
import pickle
from .Profile import Profile


file_path = Path.home()


def log_in_database(name, face):

    profile_names = []

    if file_path/"profiles.pkl".exists():

        with open(file_path/"profiles.pkl", mode="rb") as opened_file:

            profiles = pickle.load(opened_file)

        for i in range(len(profiles)):
            profile_names.append(profiles[i].return_name())
            if name == profile_names[i]:
                profiles[i].add_descriptor(face)
                added = True

        if added != True:
            profiles.append(Profile(name, face))


    else:
            profiles = [Profile(name, face)]

    with open(file_path/"profiles.pkl", mode = "wb") as opened_file:
        pickle.dump(profile, opened_file)
    
    #Need to:
    #1. Log the names and tuples into the database (a dictionary)
    #2. Assuming one-to-one correspondence with names and faces
#ABOVE IS OBJECT ORIENTED, BELOW IS DICT

def match_against_database(list_of_face_vectors):
    #Need to:
    #Make a new dictionary which averages the face vectors corresponding to the duplicated face names
    #And stores the average as part of a new (key, value) pair
    #
    #For each face vector:
    #






















    #Bring up the lists of vectors that correspond to the lists of faces
    #If there are multiple vectors that correspond to a face (recorded many times)
    #Then average the vectors to produce a better vector.
    #Now you have 
    len_list_of_faces = len(list_of_faces)
    pass
