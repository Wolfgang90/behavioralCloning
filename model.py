import tensorflow as tf
import img_preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers.core import Flatten
from keras.layers import Input, Convolution2D, Dense, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.models import Model, model_from_json
import json
import os.path
import matplotlib.pyplot as plt




#1 Import and preprocess data

#1.0 Hyperparameter:
image_rescale_size = (16,64)
#Selected green color channel as it displayed the most contrast between track and off-track
color_channel = "g"

#1.1 If pickle file with already converted data is available, load pickle file
try:
	X,y = img_preprocessing.load_data_from_pickle()

#1.2 If no pickle file available, preprocess data
except:
	X,y = img_preprocessing.load_preprocess_pickle_data_from_initial_file(image_rescale_size, color_channel)

#1.3 Split data into training and validation data

X, y = shuffle(X,y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train, test_size = 0.2)

print("The final training, validation and test set sizes are:")
print()
print("Train set size: {}".format(X_train.shape))
print("Validation set size: {}".format(X_val.shape))
print("Test set size: {}".format(X_test.shape))
print()


#2 Define model or load the model

#2.0 Hyperparameters (only for overall training, for model specific hyperparameters go to next "except:")
batch_size = 128
nb_epoch = 10

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
	print("Started model creation")
	print()

	#2.2.1 Model hyperparameters
	#2.2.1.1 Convolution - Number of output filters
	nb_filter1 = 32
	nb_filter2 = 64
	nb_filter3 = 96
	nb_filter4 = 96

	#2.2.1.2 Kernel size
	kernel_size_conv = (3,3)
	kernel_size_pool =(2,2)

	#2.2.1.3 Dropout
	drop_prob = 0.3

	#2.2.2 Define model layers

	# Define input tensor
	inputs = Input(shape=(image_rescale_size[0], image_rescale_size[1], 1))

	# 2d Convolution 32 layers, (3,3) Kernel size
	layer1 = Convolution2D(nb_filter1,kernel_size_conv[0],kernel_size_conv[1], border_mode = "same", activation = "elu")(inputs)
	layer2 = MaxPooling2D(kernel_size_pool,border_mode = "same")(layer1)
	layer3 = Dropout(drop_prob)(layer1)
	layer4 = Convolution2D(nb_filter2,kernel_size_pool[0],kernel_size_conv[1],border_mode = "same", activation = "elu")(layer3)
	layer5 = Convolution2D(nb_filter3,kernel_size_pool[0],kernel_size_conv[1],border_mode = "same", activation = "elu")(layer4)
	layer6 = Convolution2D(nb_filter4,kernel_size_pool[0],kernel_size_conv[1],border_mode = "same", activation = "elu")(layer5)
	layer7 = Flatten()(layer6)
	layer8 = Dense(64, activation = "elu")(layer7)
	layer9 = Dropout(drop_prob)(layer8)
	layer10 = Dense(16, activation = "elu")(layer9)
	layer11 = Dropout(drop_prob)(layer10)
	prediction = Dense(1)(layer11)

	#2.2.3 Initialize model
	model = Model(input = inputs, output = prediction)

	print("New model generated in this session according to specifications")
	print()

	# Set imported-markers to "True" for future reference
	model_imported = False
	weights_imported = False
	
#3 Compile model; if model.h5 available use these weigts to initialize, else random weigths
model.compile(optimizer = "Adam", loss = "binary_crossentropy", metrics = ["accuracy"])

#4 Train model
model.fit(X_train,y_train, batch_size = batch_size, nb_epoch = nb_epoch, validation_data = (X_val,y_val),shuffle = True)

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

#Print an overview of the model
print("MODEL SUMMARY")
print(model.summary())

