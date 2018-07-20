from pathlib import Path
import numpy as np
import pickle
import collections

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
    name = name[0]
    face = face[0]

    #Check if the names and faces pickle exists.
    if (file_path/"names_and_faces.pkl").exists():

        #Load the pickled dictionary.
        with open(file_path/"names_and_faces.pkl", mode = "rb") as opened_file:
            names_and_faces = pickle.load(opened_file)
            
            #If the person's name is already in the dictionary, then append the descriptor to the end of the value as part of a list.
            if name in names_and_faces.keys():

                if len(names_and_faces[name]) == 1:
                    names_and_faces[name] = [names_and_faces[name], face]
                else:
                    names_and_faces[name] = [*names_and_faces[name], face]
            #If the person's name is not in the dictionary, make a new dictionary entry.
            else:
                names_and_faces[name] = face
    #If there is no dictionary, make a new dictionary.
    else:
        names_and_faces = {}
        names_and_faces[name] = face
    
    #Save the dictionary.
    with open(file_path/"names_and_faces.pkl", mode = "wb") as opened_file:
        pickle.dump(names_and_faces, opened_file)


def match_against_database(list_of_face_vectors):
    """
    Computes the list of most likely faces corresponding to the list of face vectors.

    Parameters
    -------
    list_of_face_vectors: list
        The list of face vectors. Each vector is a numpy.ndarray, most likely shape (128,)

    Returns
    -------
    list_of_names:
        The list of most likely faces corresponding to the list of face vectors.

    """

    threshold_of_similarity = 0.24
    list_of_names = []

    #Load the database, if it exists.
    if file_path/"names_and_faces.pkl".exists():
        with open(file_path/"names_and_faces.pkl", mode = "rb") as opened_file:
            names_and_faces = pickle.load(opened_file)

            #Calculate the mean for each key.
            for key in names_and_faces:
                names_and_faces[key] = np.array(names_and_faces[key]).mean(axis = 0)

            #Here comes the fun part! Iterate thru our list of face vectors to find the best candidate name for each face vector.
            for i in range(len(list_of_face_vectors)):

                #Temporary L2 differences dictionary, to be cleared each iteration (i.e. each time a new face vector is introduced).
                temp_L2diffs = {}

                #Compute the L2 distances for each face vector.
                for key in names_and_faces:
                    temp_L2diffs[key] = L2_dists(list_of_face_vectors[i], names_and_faces[key])

                #Invert dictionary so L2 diffs become keys.
                inverted_dictionary = invert_dictionary(temp_L2diffs)

                #Make the keys a list.
                numerical_keys = list(inverted_dictionary.keys())

                #Make the keys a numpy array.
                numerical_keys = np.array(numerical_keys)

                #Filter keys only for ones lower than the threshold of similarity.
                numerical_keys = numerical_keys[numerical_keys <= threshold_of_similarity]

                #Compile a list of candidate names from the inverted dictionary values.
                list_of_candidate_names = [invert_dictionary[x] for x in numerical_keys]

                #Make a Counter object so that we can find the most frequent name.
                collection_of_names = collections.Counter(list_of_candidate_names)
                most_common_name = collection_of_names[0][0]
                 
                #If there is no decisive name, then None will be returned. 
                #Otherwise, the most likely name corresponding to the face vector will be returned.
                #The order of the list_of_face_vectors and list_of_names are the same, 
                #so list_of_face_vectors[0] corresponds to list_of_names[0].
                if type(most_common_name) == str:
                    list_of_names.append(most_common_name)
                else:
                    list_of_names.append(None)

            return list_of_names


    else:
        print("Dictionary does not exist. Please initialize dictionary.")
        pass


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

    dists = -2 * np.matmul(x, y.T)
    dists +=  np.sum(x**2)[np.newaxis]
    dists += np.sum(y**2)
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




