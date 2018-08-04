#Import required packages
from flask import Flask
from flask_ask import Ask, statement, question
from torchvision.models import resnet18
from torchvision import transforms
from pathlib import Path
from PIL import Image

import face_recognition as fr
import io
import requests
import torch
import camera
import matplotlib
import matplotlib.image
import numpy as np
import pickle as pkl

#Setup Flask and Flask-app
app = Flask(__name__)
ask = Ask(app, '/')

#Open resnet features pickle file and make it a list
with open(Path.home() / "resnet18_features_train.pkl", mode="rb") as opened_file:
    rf = pkl.load(opened_file)
    rf_list = list(rf.items())

#Open a custom-created dictionary containing all of the captions
with open(Path.home() / "sorted_dict.pkl", mode = "rb") as dic:
    sorted_dict = pkl.load(dic)

#Define a class IdentityModule to help set up the Resnet CNN
class IdentityModule(torch.nn.Module):
    def forward(self, inputs):
        return inputs

#Initialize Resnet, setting it in evaluation mode
model = resnet18(pretrained=True)
model.fc = IdentityModule()
model.eval()



def get_image(img_url):
    ''' Fetch an image from CoCo.

        Parameters
        ----------
        img_url : str
            The url of the image to fetch, in the format:
                http://images.cocodataset.org/--dataset--/---unique_image_id--.jpg

            `dataset` is the specific coco dataset you wish to use, such as train2014 or val2017
            `unique_image_id` is an alpha-numeric sequence specific to the image you want to fetch

        Returns
        -------
        PIL.Image
            The image.
        '''
    response = requests.get(img_url)
    return Image.open(io.BytesIO(response.content))

def take_and_return_resnet():
    """
    Takes a picture using the computer's camera, saves it,  and converts to Resnet array.
    Parameters:
    ---------
    Returns:
    ---------
    torch.tensor:
        A pytorch tensor of shape (1, 512) which is the output of the ResNet CNN.
    """
    pic = camera.take_picture()
    matplotlib.image.imsave(Path.home() / "saved.jpg", pic, format="jpg")
    with open(Path.home() / 'saved.jpg', 'rb') as inf:
        return model(preprocess(get_image_from_data(inf.read()))[np.newaxis])

def get_image_from_data(imgdata):
    """
    Get the JPEG / PNG image from the binary image data.
    Parameters:
    ---------
    imgdata: IO.StringBuffer object
        The object corresponding to the image data which needs to be read.
    Returns:
    ---------
    Image:
        The data of the image (needs preprocessing).
    """

    return Image.open(io.BytesIO(imgdata))


#The preprocessing (resize, centercrop, totensor, and normalize)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def find_closest_caption(resnet_embedding):
    """
    Parameters:
    ---------
    resnet_embedding: torch.tensor
        The 512-dimension resnet embedding of the image.
    Returns:
    ---------
    caption: str
        The first caption of image whose Resnet's L2 distance is closest to resnet_embedding.
    """

    L2_dists = torch.tensor([torch.sqrt(torch.sum((resnet_embedding - resnet_vector[1]) ** 2)) for resnet_vector in rf_list])
    id_min = rf_list[torch.argmin(L2_dists)][0]
    return sorted_dict[id_min]["annotations"][0]

#Set up homepage
@app.route('/')
def homepage():
    return "Hello, this is the webpage of Alexa's Face Recognition ability."

#Set up skill
@ask.launch
def start_skill():
    welcome_message = 'One. Add. Two. Recognize. Three. Describe.'
    return question(welcome_message)

#Query user's name
@ask.intent("query_name")
def query_name():
    ques = "What is your name?"
    return question(ques)

#Get user's name
@ask.intent("get_name")
def get_name(name):
    if fr.cameraToStorage(name):
        name = name + " successfully stored"
        return statement(name)
    else:
        return statement("Sorry, I'm having trouble seeing you.")

#Use fr module to recognize faces. Added familiarities to make Alexa seem more human.
@ask.intent("recognize_faces")
def recognize_faces():
    list_of_names = fr.cameraToMatch()

    s = ""

    familiarities = ["my buddy ", "my friend ", "my pal ", "my homie ", "mi amigo ", " ", " ", " ", " ", " "]
    index = np.random.randint(0, 10)

    if list_of_names != 0:
        if len(list_of_names) > 1:
            list_of_names.insert(-2, "and")
        for i in list_of_names:
            s = s + i + " "
        people = "I see " + familiarities[index] + s
        if "unknown" in list_of_names:
            people = people + ". . .  There are people I don't know. Try adding them to the database."
        return statement(people)
    else:
        return statement("Sorry, I'm having trouble recognizing faces.")

#Describe the scene using L2 distances method.
@ask.intent("describe_scene")
def describe_scene():
    seeing = "I see " + find_closest_caption(take_and_return_resnet())
    return statement(seeing)

@ask.intent("AMAZON.FallbackIntent")
def fallbackintent():
    return statement("Sorry, I don't understand.")

@ask.intent("AMAZON.StopIntent")
def stopintent():
    return statement("OK, goodbye!")

@ask.intent("AMAZON.HelpIntent")
def helpintent():
    return statement("Say one to take a picture and store your face in the database. Say two to recognize faces. Say three to recognize the scene.")

#Run app
if __name__ == '__main__':
    app.run(debug=True)