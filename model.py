import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers.core import Flatten
from keras.preprocessing.image import flip_axis
from keras.layers import Input, Lambda, Convolution2D, Dense, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
from keras.models import Model, model_from_json
from scipy import misc
import math
import random
import json

# Import csv-data
csv_data = pd.read_csv("data/driving_log.csv")

# Remove data with throttle < 0.5
csv_data = csv_data.loc[csv_data["throttle"] > 0.5]

# Additionally add flipped data for data with strong steering angles
csv_data["to_be_flipped"] = np.zeros(len(csv_data))
extreem_steering = csv_data[csv_data["steering"]>0.15].copy()
extreem_steering["to_be_flipped"] = extreem_steering["to_be_flipped"].apply(lambda x: x +1)
csv_data = csv_data.append(extreem_steering,ignore_index=True)

# Shuffle csv-data
csv_data = shuffle(csv_data)

# Split csv_data
train, val = train_test_split(csv_data,test_size = 0.1)
train_samples = len(train)
validation_samples = len(val)
print("The final training and test set sizes are:")
print()
print("Train set size: {}".format(len(train)))
print("Validation set size: {}".format(len(val)))


#2.0 Hyperparameters (only for overall training, for model specific hyperparameters go to next "except:")
epochs = 5
batch_size = 256
camera_steering_adjustment = 0.15
color_channel = None
if color_channel:
    nb_color_channel = 1
else:
    nb_color_channel = 3
image_rescale_size = (64,64,3)
image_final_size = (64,64,nb_color_channel)
flip_prob = 0.3


#2.1 Import stored model and weights from previous training session (if available)
try:
    # Import model
    with open("model.json","r") as f:
        model = model_from_json(json.load(f))

    # Import weights
    model.load_weights("model.h5",by_name = False)
    print("Model and weights were loaded from the files 'model.json' and 'model.h5'")
    print()

    # Set imported-markers to "True" for future reference
    model_imported = True
    weights_imported = True



#2.2 Define model (if no model and weights available from previous sessions)
except:
    #2.2.1 Model hyperparameters
    #2.2.1.1 Convolution - Number of output filters
    nb_filter1 = 4
    nb_filter2 = 8
    nb_filter3 = 12
    nb_filter4 = 16

    #2.2.1.2 Kernel size
    kernel_size_conv = (3,3)
    kernel_size_pool = (2,2)

    #2.2.1.3 Dropout
    drop_prob = 0.3

    #2.2.2 Define model layers

    # Define input tensor
    inputs = Input(shape=image_final_size)

    # Layer
    layer0 = Lambda(lambda x: x/127.5 - 1.)(inputs)
    layer1 = Convolution2D(nb_filter1,kernel_size_conv[0],kernel_size_conv[1], border_mode = "same")(layer0)
    layer1 = ELU()(layer1)
    layer2 = Dropout(drop_prob)(layer1)
    layer3 = Convolution2D(nb_filter2,kernel_size_conv[0],kernel_size_conv[1],border_mode = "same")(layer2)
    layer3 = ELU()(layer3)
    #layer4 = MaxPooling2D(border_mode = "same", pool_size = kernel_size_pool)(layer3)
    layer5 = Convolution2D(nb_filter3,kernel_size_conv[0],kernel_size_conv[1],border_mode = "same")(layer3)
    layer5 = ELU()(layer5)
    #layer6 = Convolution2D(nb_filter4,kernel_size_conv[0],kernel_size_conv[1],border_mode = "same")(layer5)
    #layer6 = ELU()(layer6)
    layer7 = Flatten()(layer5)
    layer8 = Dense(32)(layer7)
    layer8 = ELU()(layer8)
    layer9 = Dropout(drop_prob)(layer8)
    layer10 = Dense(16)(layer9)
    layer10 = ELU()(layer10)
    layer11 = Dropout(drop_prob)(layer10)
    layer12 = Dense(8)(layer11)
    layer12 = ELU()(layer12)
    prediction = Dense(1)(layer11)

    #2.2.3 Initialize model
    model = Model(input = inputs, output = prediction)
    
    print("New model generated in this session according to specifications")
    print()

    # Set imported-markers to "True" for future reference
    model_imported = False
    weights_imported = False

#3 Compile model; if model.h5 available use these weigts to initialize, else random weigths
optimizer = Adam(lr = 0.0001)
model.compile(optimizer = optimizer, loss = "mse", metrics = ["accuracy"])


# Define function for loading single image
# Input: data -> Train or validation file; camera_position (left,right,center); 
# Camera_steering adjustment -> Tupel for steering adjustments (left_adj (typically positive as move to the right/center required), center_adj, right_adj(typically negative as move to the left/center required))
# data_position -> row number in train or validation data
def load_image_and_steering_angle(data, camera_position, camera_steering_adjustment, data_position):
    
    name = "data/{}".format(data[camera_position][data_position].strip())  
    X = misc.imread(name)
    X = X.astype("float32")

    if camera_position == "left": 
        steering_adj = camera_steering_adjustment
    if camera_position == "center":
        steering_adj = 0
    if camera_position == "right":
        steering_adj = -camera_steering_adjustment
    
    y = data["steering"][data_position] + steering_adj

    if data["to_be_flipped"][data_position]:
        X, y = flip_image(X,y)
    
    return X,y
        

# Define function for selecting relevant image regions
def select_relevant_image_regions(img):
    return np.concatenate((img[60:124,:120,:],img[60:124,200:,:]),axis = 1)

def resize_image(img, image_rescale_size):
    return misc.imresize(img,image_rescale_size)


# Define function to reduce image to one image channel
def select_color_channel(img,color_channel = None):
    if color_channel == "r":
        img = img[:,:,0]
        return img[:,:,np.newaxis]
    elif color_channel == "g":
        img = img[:,:,1]
        return img[:,:,np.newaxis]
    elif color_channel == "b":
        img = img[:,:,2]
        return img[:,:,np.newaxis]
    return img
    
def flip_image(img,angle):
    img = flip_axis(img,1)
    angle *= -1
    return img, angle

# Fit the model
# Create a batch generator
import math
import random

# Image rescaling to be implemented!!!
class BatchDataGenerator:
    def __init__(self, data, batch_size, image_rescale_size, image_final_size, camera_steering_adjustment = 0, color_channel = None, flip_prob = 0):
        self.data = data
        self.batch_size = batch_size
        self.camera_steering_adjustment = camera_steering_adjustment
        self.image_rescale_size = image_rescale_size
        self.image_final_size = image_final_size
        self.color_channel = color_channel
        self.flip_prob = flip_prob
        self.counter = 0
        self.X_batch = np.zeros((batch_size, *image_final_size))
        self.y_batch = np.zeros(batch_size)
        self.data = data.sample(frac=1).reset_index(drop=True)
        self.available_examples = len(data)
        self.available_batches = math.trunc(batch_size/len(data))
    def __iter__(self):
        return self
    def __next__(self):
        while True:              
            for batch in range(self.batch_size):
                # Ensure that set is reshuffeled once all data is seen
                if self.counter == self.available_examples:
                    self.data = self.data.sample(frac=1).reset_index(drop=True)
                    self.counter = 0
                image_choice_variable = random.randint(0,2)
                camera_position = "left"
                if image_choice_variable == 1:
                    camera_position = "center"
                if image_choice_variable == 2:
                    camera_position = "right"
                    
                X,y = load_image_and_steering_angle(self.data, camera_position, self.camera_steering_adjustment, data_position = self.counter)
                
                X = select_relevant_image_regions(X)

                X = resize_image(X, self.image_rescale_size)

                if np.random.choice((True,False), p=[self.flip_prob,1-self.flip_prob]):
                    X, y = flip_image(X,y)

                X = select_color_channel(X, self.color_channel)

                         
                self.X_batch[batch] = X
                self.y_batch[batch] = y         
                self.counter += 1
            return self.X_batch, self.y_batch


# Fit the model

train_generator = BatchDataGenerator(data = train, batch_size = batch_size, image_rescale_size = image_rescale_size, image_final_size = image_final_size,
                                     camera_steering_adjustment = camera_steering_adjustment, 
                                     color_channel = color_channel, flip_prob = flip_prob)
val_generator = BatchDataGenerator(data = val, batch_size = batch_size, image_rescale_size = image_rescale_size, image_final_size = image_final_size,
                                     color_channel = color_channel)

samples_per_epoch = int(len(train)/batch_size) * batch_size
print("Samples epoch:" + str(samples_per_epoch))
model.fit_generator(generator = train_generator,samples_per_epoch= samples_per_epoch, 
                    nb_epoch = epochs, validation_data = val_generator, nb_val_samples = validation_samples)


#5 Save model to model.json and weights to model.h5
#5.1 Save model to model.json
#5.1.1 If model was already imported model.json does not need to be updated
if model_imported == True:
    print("Model not saved as model was already imported from 'model.json'")
    print()

#5.1.2 Save model to model.json (only if model was newly created)
else:
    model_json = model.to_json()
    with open("model.json","w") as f:
        json.dump(model_json,f)
    print("Model was saved to 'model.json'")
    print()

#5.2 Save weights to model.h5 (if user wants to update them)
if weights_imported == True:
    save_weights = input("Do you want to save the trained weights [y/n]? If you don't save the weights the training effect of this run will be lost: ")
    while save_weights not in ["y","n"]:
        save_weights = input("Your input was not 'y' or 'n'. Please provide proper input: ")
    if save_weights == "y":
        model.save_weights("./model.h5")
        print("Updated model weights saved to 'model.h5'")
        print()
    else:
        print("Weights were not saved")
        print()
else:
    model.save_weights("./model.h5")
    print("As this was the first round of training weights were saved to 'model.h5'")
    print()