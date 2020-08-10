import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from django.conf import settings
from PIL import Image
import math

# Setting Unique Breeds
unique_breeds = np.array(['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
       'american_staffordshire_terrier', 'appenzeller',
       'australian_terrier', 'basenji', 'basset', 'beagle',
       'bedlington_terrier', 'bernese_mountain_dog',
       'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound',
       'bluetick', 'border_collie', 'border_terrier', 'borzoi',
       'boston_bull', 'bouvier_des_flandres', 'boxer',
       'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff',
       'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua',
       'chow', 'clumber', 'cocker_spaniel', 'collie',
       'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo',
       'doberman', 'english_foxhound', 'english_setter',
       'english_springer', 'entlebucher', 'eskimo_dog',
       'flat-coated_retriever', 'french_bulldog', 'german_shepherd',
       'german_short-haired_pointer', 'giant_schnauzer',
       'golden_retriever', 'gordon_setter', 'great_dane',
       'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael',
       'ibizan_hound', 'irish_setter', 'irish_terrier',
       'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound',
       'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier',
       'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier',
       'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
       'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
       'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
       'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog',
       'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian',
       'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler',
       'saint_bernard', 'saluki', 'samoyed', 'schipperke',
       'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier',
       'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier',
       'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
       'standard_poodle', 'standard_schnauzer', 'sussex_spaniel',
       'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier',
       'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel',
       'west_highland_white_terrier', 'whippet',
       'wire-haired_fox_terrier', 'yorkshire_terrier'])

# Define image size
IMG_SIZE = 224

# Create a function for preprocessing images
def process_images(image_path, img_size=IMG_SIZE):
    """
    Take an image file path and turns the image into Tensor.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3) # color channel (rgb)
    image = tf.image.convert_image_dtype(image, tf.float32)# Normalization
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image

# Batch Size
BATCH_SIZE = 32

# Create a function to turn data into batches
def create_data_batches(X, batch_size=BATCH_SIZE):
    """
    Create Batches of the test data
    """
    data = tf.data.Dataset.from_tensor_slices(tf.constant(X))
    data_batch = data.map(process_images).batch(BATCH_SIZE)
    return data_batch


# Turn predictions probablities into their respestive label
def get_pred_label(prediction_probabilities):
    return unique_breeds[np.argmax(prediction_probabilities)]


# Unbatch the data
def unbatchify(batched_data):
    unbatched_data = []
    for image in batched_data.unbatch().as_numpy_iterator():
        unbatched_data.append(image)
    return unbatched_data

# Plot the Predictions and Saving Figure
def plot_images(imgs, labels, preds):
    plt.figure(figsize=(10*2, 5*math.ceil(len(imgs)/2)))
    for i,image in enumerate(imgs):
        plt.subplot(math.ceil(len(imgs)/2), 2, i+1)
        plt.xticks([])
        plt.yticks([])
        if np.round(np.max(preds[i])*100, 2) < 50:
            plt.title(labels[i] + ' (' + str(np.round(np.max(preds[i])*100, 2)) + '%)', color='r', fontsize=30)
        else:
            plt.title(labels[i] + ' (' + str(np.round(np.max(preds[i])*100, 2)) + '%)', color='g', fontsize=30)
        plt.imshow(image)
    path = str(str(settings.MEDIA_ROOT).replace("media", "") + '\\static\\figure\\figure.png')
    print(path)
    # path.replace("media", "static")
    # print(path)
    plt.savefig(path)
    # Image.open(path).convert('RGB').save(str(str(settings.MEDIA_ROOT).replace('media', '') + '\\static\\figure\\figure.jpg','JPEG'))

def load_model(model_path):
    model = tf.keras.models.load_model(model_path,custom_objects={"KerasLayer":hub.KerasLayer})
    return model