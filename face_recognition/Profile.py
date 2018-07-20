import numpy as np

class Profile:

    array_of_descriptors = np.array([])
    name = ""

    def __init__(self, name, face):
        """
        Parameters:
        name: String
            The string corresponding to the person whose picture is being taken
        face: numpy.ndarray
            The numpy array containing 128 descriptors of the person
        """
        self.name = name
        self.array_of_descriptors = self.array_of_descriptors.append(face)
       

    def add_descriptor(face):
        """
        Parameters:
        face: numpy.ndarray
        """
        self.array_of_descriptors.append(face)

    def return_mean_descriptor(self):
        """
        Returns:
        mean_descriptor: a numpy array of shape 128 with the mean descriptors
        """
        return np.mean(self.array_of_descriptors, axis = 0)

    def return_name(self):
        """
        Returns:
        name: The name of the profile
        """
        return self.name
        

