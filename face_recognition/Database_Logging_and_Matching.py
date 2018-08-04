from pathlib import Path
import numpy as np
import pickle
#import collections

file_path = Path.home()

def log_in_database(name, face):
    """
    Logs a name and descriptor in the database.
    -------
    Parameters:
    name: str
        The name of the person whose face-descriptors are being logged in the database.
    face: numpy.ndarray
        The descriptor for the face. Typically shape (128,).
    """
    #FIX: Curtain name array and face array to the first element of each respective array.
    if len(name) != 0:
        faces_present = 1
        name = name[0]
        face = face[0]

        #Check if the names and faces pickle exists.
        if (file_path/"names_and_faces.pkl").exists():

            #Load the pickled dictionary.
            with open(file_path/"names_and_faces.pkl", mode = "rb") as opened_file:
                names_and_faces = pickle.load(opened_file)

                #If the person's name is already in the dictionary, then append the descriptor to the end of the value as part of a list.
                if name in names_and_faces.keys():
                    names_and_faces[name].append(face)

                #If the person's name is not in the dictionary, make a new dictionary entry.
                else:
                    names_and_faces[name] = [face]
        #If there is no dictionary, make a new dictionary.
        else:
            names_and_faces = {}
            names_and_faces[name] = [face]

        #Save the dictionary.
        with open(file_path/"names_and_faces.pkl", mode = "wb") as opened_file:
            pickle.dump(names_and_faces, opened_file)
    else:
        faces_present = 0
    return faces_present


def match_against_database(list_of_face_vectors):
    """
    Computes the list of most likely faces corresponding to the list of face vectors.

    Parameters
    -------
    list_of_face_vectors: list
        The list of face vectors. Each vector is a numpy.ndarray, most likely shape (128,)

    Returns
    -------
    names_to_return:
        The list of most likely faces corresponding to the list of face vectors.

    """

    threshold_of_similarity = 0.4

    if len(list_of_face_vectors) != 0:
        #Load the database, if it exists.
        if (file_path/"names_and_faces.pkl").exists():
            with open(file_path/"names_and_faces.pkl", mode = "rb") as opened_file:
                names_and_faces = pickle.load(opened_file)

                #Calculate the mean for each key.
                for key in names_and_faces:
                    names_and_faces[key] = np.array(names_and_faces[key]).mean(axis = 0)


                #Here comes the fun part! Iterate thru our list of face vectors to find the best candidate name for each face vector.
                face_vectors = np.array(list_of_face_vectors)
                #print(face_vectors.shape)
                names = np.array(list(names_and_faces.keys()))
                faces = np.array(list(names_and_faces.values()))
                candidates = L2_dists_vectorized(face_vectors, faces)

                minimum_indices = np.argmin(candidates, axis = 1)
                #print("candidates are", candidates)
                minimum_args = np.min(candidates, axis = 1)
                names_to_return = names[minimum_indices]
                names_to_return[minimum_args > threshold_of_similarity] = "unknown"


                return names_to_return


        else:
            return 0
    else:
        return 0


def L2_dists(x, y):
    """ Computing L2 distances using memory-efficient
        vectorization.

        Parameters
        ----------
        x : numpy.ndarray, shape=(D,)
        y : numpy.ndarray, shape=(D,)

        Returns
        -------
        numpy.ndarray, shape=(D,)
            The Euclidean distance between each pair of
            rows between `x` and `y`."""
    #print(x.shape)
    #print(y.shape)
    dists = -2 * np.matmul(x, y.T)
    dists +=  np.sum(x**2)[np.newaxis]
    dists += np.sum(y**2)
    return  np.sqrt(dists)

def L2_dists_vectorized(x, y):
    """ Computing pairwise distances using memory-efficient
        vectorization.

        Parameters
        ----------
        x : numpy.ndarray, shape=(M, D)
        y : numpy.ndarray, shape=(N, D)

        Returns
        -------
        numpy.ndarray, shape=(M, N)
            The Euclidean distance between each pair of
            rows between `x` and `y`."""
    dists = -2 * np.matmul(x, y.T)
    dists +=  np.sum(x**2, axis=1)[:, np.newaxis]
    dists += np.sum(y**2, axis=1)
    return  np.sqrt(dists)


def invert_dictionary(dictionary):
    """
    Invert the dictionary so that the keys become values and the values become keys.
    Parameters
    -------
    dictionary: a dictionary
        The dictionary to be inverted.
    
    Returns
    -------
    inverted: a dictionary
        The inverted dictionary.
    """
    inverted = {}
    for key in dictionary:
        inverted[dictionary[key]] = key
    return inverted   





