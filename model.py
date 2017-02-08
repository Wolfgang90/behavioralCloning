import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers.core import Flatten
from keras.preprocessing.image import flip_axis
from keras.layers import Input, Lambda, Convolution2D, Dense, Dropout
from keras.models import Model, model_from_json
from scipy import misc
import random
import json

#1 Import and preprocess csv-data in pandas dataframe
#1.1 Import csv-data to pandas dataframe
csv_data = pd.read_csv("data/driving_log.csv")

#1.2 Modify data

#1.2.1 Remove data with throttle < 0.5
csv_data = csv_data.loc[csv_data["throttle"] > 0.5]

#1.2.2 Create additional data for strong steering angles by duplicating data (to be flipped in later batch generation)

# Create column in existing dataframe to mark data to be flipped in later batch generation
csv_data["to_be_flipped"] = np.zeros(len(csv_data))

# Copy data to be duplicated into new dataframe
extreem_steering = csv_data[csv_data["steering"]>0.10].copy()

# Mark data as to be flipped during batch generation
extreem_steering["to_be_flipped"] = extreem_steering["to_be_flipped"].apply(lambda x: x +1)

# Append copied data to existing data
csv_data = csv_data.append(extreem_steering,ignore_index=True)

#1.2.3 Splis csv_data

# Shuffle csv-data to ensure random split
csv_data = shuffle(csv_data)

# Split csv_data
train, val = train_test_split(csv_data,test_size = 0.1)
train_samples = len(train)
validation_samples = len(val)

# Output results
print("The final training and test set sizes are:")
print()
print("Train set size: {}".format(len(train)))
print("Validation set size: {}".format(len(val)))


#2 Define model
#2.0 Hyperparameters

#2.0.1 General hyperparameters
epochs = 5
batch_size = 64
camera_steering_adjustment = 0.1

#2.0.2 General hyperparameters with direct influence on model architecture 
# !!! If the model is loaded from existing model.h5 and model.json, 
# these parameters need to match the model architecture in the files !!!
color_channel = None
if color_channel:
    nb_color_channel = 1
else:
    nb_color_channel = 3
image_rescale_size = (66,200,3)
image_final_size = (66,200,nb_color_channel)
flip_prob = 0.6

#2.0.3 Model specific hyperparameters
# !!! If the model is loaded from existing model.h5 and model.json, 
# these parameters will have no influence on the model architecture. The parameters are sourced from the files !!!

# Convolution (Number of output filters)
nb_filter1 = 24
nb_filter2 = 36
nb_filter3 = 48
nb_filter4 = 64
nb_filter5 = 64

# Convolution (Stride)
stride_1 = (2,2)
stride_2 = (2,2)
stride_3 = (2,2)
stride_4 = (1,1)
stride_5 = (1,1)

# Convolution (Kernel size)
kernel_size_conv_1 = (5,5)
kernel_size_conv_2 = (5,5)
kernel_size_conv_3 = (5,5)
kernel_size_conv_4 = (3,3)
kernel_size_conv_5 = (3,3)

#Dropout (Probability)
drop_prob = 0.3


#2.1 Import stored model and weights from previous training session (if model.json and model.h5 file available)
try:
    # Import model
    with open("model.json","r") as f:
        model = model_from_json(json.load(f))

    # Import weights
    model.load_weights("model.h5",by_name = False)
    print("Model and weights were loaded from the files 'model.json' and 'model.h5'. Please mind, that the model parameters in the file need to match the data input created by this program")
    print()

    # Set imported-markers to "True" for future reference
    model_imported = True
    weights_imported = True


#2.2 Define model (if no model and weights available from previous sessions in model.json and model.h5 files)
except:
    #2.2.1 Define model layers

    # Input tensor
    inputs = Input(shape=image_final_size)

    # Model layers
    layer0 = Lambda(lambda x: x/127.5 - 1.0)(inputs)
    layer1 = Convolution2D(nb_filter1,kernel_size_conv_1[0],kernel_size_conv_1[1],activation = "relu", border_mode = "valid", subsample = stride_1)(layer0)
    layer2 = Dropout(drop_prob)(layer1)
    layer3 = Convolution2D(nb_filter2,kernel_size_conv_2[0],kernel_size_conv_2[1],activation = "relu", border_mode = "valid", subsample = stride_2)(layer1)
    layer4 = Convolution2D(nb_filter3,kernel_size_conv_3[0],kernel_size_conv_3[1],activation = "relu", border_mode = "valid", subsample = stride_3)(layer3)
    layer5 = Convolution2D(nb_filter4,kernel_size_conv_4[0],kernel_size_conv_4[1],activation = "relu", border_mode = "valid", subsample = stride_4)(layer4)
    layer6 = Convolution2D(nb_filter5,kernel_size_conv_5[0],kernel_size_conv_5[1],activation = "relu", border_mode = "valid", subsample = stride_5)(layer5)
    layer7 = Flatten()(layer6)
    layer8 = Dense(100, activation = "relu")(layer7)  
    layer9 = Dense(50, activation = "relu")(layer8)
    layer10 = Dense(10)(layer9)
    prediction = Dense(1)(layer10)

    #2.2.2 Initialize the model
    model = Model(input = inputs, output = prediction)
    
    print("New model generated in this session according to specifications")
    print()

    # Set imported-markers to "True" for future reference
    model_imported = False
    weights_imported = False

#3 Compile model (if model.h5 was available weigts from this file are used to initialize the model, else random weigths are created)
model.compile(optimizer = "adam", loss = "mse", metrics = ["accuracy"])
model.summary()
print()


#4 Define functions to create model input batches 
#4.1 Define image loading and modification functions
#4.1.1 Define function to load images and corresponding steering angles

def load_image_and_steering_angle(data, camera_position, camera_steering_adjustment, data_position):
    
    """ 
    Input:
    data (pd.dataframe): Train or validation pandas dataframe
    camera_position (string): From where the image is taken in the simulator (Possible values: "left", "right", "center")
    camera_steering adjustment (float): Float for steering adjustments (should be positive) 
    data_position (int): Row number in pandas dataframes for train or validation data

    Output:
    X: Image
    y: Steering angle
    """
    
    # Import image from 'data/IMG' folder
    name = "data/{}".format(data[camera_position][data_position].strip())  
    X = misc.imread(name)
    X = X.astype("float32")

    # Create steering adjustment additive for camera position
    if camera_position == "left": 
        steering_adj = camera_steering_adjustment
    if camera_position == "center":
        steering_adj = 0
    if camera_position == "right":
        steering_adj = -camera_steering_adjustment
    
    # Take steering angle for image from pandas dataframe and add steering adjustment
    y = data["steering"][data_position] + steering_adj

    # Flip image and reverse steering angle if image data is a copy of existing data and therefore is marked as "to_be_flipped" in pandas dataframe
    if data["to_be_flipped"][data_position] == 1:
        X, y = flip_image(X,y)   

    return X,y
        

#4.1.2 Define function for selecting relevant image regions (cut horizon and car in foreground)
def select_relevant_image_regions(img):
    return img[54:120,:,:]

#4.1.3 Resize image
def resize_image(img, image_rescale_size):
    return misc.imresize(img,image_rescale_size)

#4.1.4 Flip image and reverse corresponding steering angle
def flip_image(img,angle):
    img = flip_axis(img,1)
    angle *= -1
    return img, angle

#4.1.5 Reduce image to one image channel or leave image untouched if input is 'None'
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



#5 Create batch generator

class BatchDataGenerator:
    def __init__(self, data, batch_size, image_rescale_size, image_final_size, camera_steering_adjustment = 0, color_channel = None, flip_prob = 0):
        
        """
        Input:
        data (pd.dataframe): Train or validation pandas dataframe
        batch_size (int): Number of samples in batch
        image_rescale_size (tuple): Size to which the image needs to be rescaled during image preprocessing as tuple of the form (n,n,3) (third dimension needs to be 3, no matter wheter 1 or all 3 color channels will be selected eventually)
        image_final_size (tuple): Final image size after all preprocessing steps (incl. color channel selection) as tuple of the form (n,n,n)
        camera_steering adjustment (float): Float for steering adjustments (should be positive)
        color_channel (string): Color channel to be selected from image. If 'None', all three color channels will be included in the batch. Else select 'r','g','b' for respective RGB-channel
        flip_prob (float): Probability with which image will be flipped around y axis and and steering angle will be reversed in order to address bias to left turns due to track design
        """

        self.data = data
        self.batch_size = batch_size
        self.camera_steering_adjustment = camera_steering_adjustment
        self.image_rescale_size = image_rescale_size
        self.image_final_size = image_final_size
        self.color_channel = color_channel
        self.flip_prob = flip_prob
        self.counter = 0

        # Define batch placeholder dataframes
        self.X_batch = np.zeros((batch_size, *image_final_size))
        self.y_batch = np.zeros(batch_size)

        # Shuffle data
        self.data = data.sample(frac=1).reset_index(drop=True)

        # Calculate data sample size
        self.available_examples = len(data)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            for batch in range(self.batch_size):

                # Ensure that data is reshuffeled once all data is seen
                if self.counter == self.available_examples:
                    self.data = self.data.sample(frac=1).reset_index(drop=True)
                    self.counter = 0

                # Randomly chose which camera position to take image input from
                image_choice_variable = random.randint(0,2)
                camera_position = "left"
                if image_choice_variable == 1:
                    camera_position = "center"
                if image_choice_variable == 2:
                    camera_position = "right"
                
                # Load images and corresponding steering angles    
                X,y = load_image_and_steering_angle(self.data, camera_position, self.camera_steering_adjustment, data_position = self.counter)
                
                # Select relevant image regions (cut horizon and car in foreground)
                X = select_relevant_image_regions(X)

                # Resize image
                X = resize_image(X, self.image_rescale_size)

                # Flip image to correct for bias towards one direction in turns due to round race track
                if np.random.choice((True,False), p=[self.flip_prob,1-self.flip_prob]):
                    X, y = flip_image(X,y)

                # Select color channel if required ('None'-color_channel input returns all color channels untouched)
                X = select_color_channel(X, self.color_channel)

                # Define batch output         
                self.X_batch[batch] = X
                self.y_batch[batch] = y

                # Increase utilizied image counter by one         
                self.counter += 1

            return self.X_batch, self.y_batch


#6 Fit the model
#6.1 Define data generators
train_generator = BatchDataGenerator(data = train, batch_size = batch_size, image_rescale_size = image_rescale_size, image_final_size = image_final_size,
                                     camera_steering_adjustment = camera_steering_adjustment, 
                                     color_channel = color_channel, flip_prob = flip_prob)
val_generator = BatchDataGenerator(data = val, batch_size = batch_size, image_rescale_size = image_rescale_size, image_final_size = image_final_size,
                                     color_channel = color_channel)

#6.2 Fit the model
samples_per_epoch = int(len(train)/batch_size) * batch_size
model.fit_generator(generator = train_generator,samples_per_epoch= samples_per_epoch, 
                    nb_epoch = epochs, validation_data = val_generator, nb_val_samples = validation_samples)


#7 Save model to model.json and weights to model.h5
#7.1 Save model to model.json
#7.1.1 If model was already imported model.json does not need to be updated
if model_imported == True:
    print("Model not saved as model was already imported from 'model.json'")
    print()

#7.1.2 Save model to model.json (only if model was newly created)
else:
    model_json = model.to_json()
    with open("model.json","w") as f:
        json.dump(model_json,f)

    print("Model was saved to 'model.json'")
    print()

#7.2 Save weights to model.h5 (if user wants to update them)
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
