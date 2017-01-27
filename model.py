import tensorflow as tf
import numpy as np
import csv
import pickle
from scipy import misc
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers.core import Flatten
from keras.layers import Input, Convolution2D, Dense, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.models import Model, model_from_json
import json
import os.path

#1 Import data

# If required data was already converted earlier and is stored in a pickle file
try:
	
	with open("behavioral_cloning_X.p","rb") as f:
		X = pickle.load(f)

	with open("behavioral_cloning_y.p","rb") as f:
		y = pickle.load(f)
	print("Data was loaded from pickle files 'behavioral_cloning_X.p' and 'behavioral_cloning_y.p'")
	print()

# If required data was not converted yet, it is converted and eventually stored in a pickle file
except:
	# Import image names from csv-file
	with open("driving_log.csv","rt") as csvfile:
		file = csv.reader(csvfile, delimiter = ",")
		image_name = list(list(zip(*file))[0])

	# Import steering angles from csv-file
	with open("driving_log.csv","rt") as csvfile:
		file = csv.reader(csvfile, delimiter = ",")	
		steering_angle = list(list(zip(*file))[3])

	# Modify csv-input
	# if csv-file with header (Udacity standard input)
	if image_name[0] == "center":
		image_name = [name[name.find("IMG/"):] for name in image_name[1:]]
		y = np.array([float(angle) for angle in steering_angle[1:]])

	# if csv-file without header (Self-generated input with simulator)
	else:
		image_name = [name[name.find("IMG/"):] for name in image_name]
		y = np.array([float(angle) for angle in steering_angle])

	# Load first image file and add additional axis for images
	X = misc.imread(image_name[0])[np.newaxis,:]

	# Load all other image files
	for i in image_name[1:]:
		X = np.concatenate((X, misc.imread(i)[np.newaxis,:]), axis = 0)

	with open("behavioral_cloning_X.p","wb") as f:
		pickle.dump(X,f)
	with open("behavioral_cloning_y.p","wb") as f:
		pickle.dump(y,f)
	print("Data was converted and stored in the pickle files 'behavioral_cloning_X.p' and 'behavioral_cloning_y.p' for future use")
	print()


#2 Split data into training and validation data

X, y = shuffle(X,y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train, test_size = 0.2)

print("The final training, validation and test set sizes are:")
print()
print("Train set size: {}".format(X_train.shape))
print("Validation set size: {}".format(X_val.shape))
print("Test set size: {}".format(X_test.shape))
print()


#3 Define model or load the model

# Parameter overview
# Overall
batch_size = 5
nb_epoch = 2

try:
	# Import model
	with open("model.json","r") as f:
		model = model_from_json(json.load(f))

	# Import weights
	model.load_weights("model.h5",by_name = False)
	print("Model and weights were loaded from the files 'model.json' and 'model.h5'")
	print()
	model_imported = True
	weights_imported = True

except:	
	
	# Convolution
	# Number of output filters
	nb_filter1 = 32
	nb_filter2 = 64
	nb_filter3 = 96
	nb_filter4 = 96
	# Kernel size
	kernel_size_conv = (3,3)
	kernel_size_pool =(2,2)

	#Dropout
	drop_prob = 0.3

	# Return the input tensor
	inputs = Input(shape=(160, 320, 3))


	# Define layers
	# 2d Convolution 32 layers, (3,3) Kernel size
	layer1 = Convolution2D(nb_filter1,kernel_size_conv[0],kernel_size_conv[1], border_mode = "same")(inputs)
	layer2 = ELU()(layer1)
	layer3 = MaxPooling2D(kernel_size_pool,border_mode = "same")(layer2)
	layer4 = Dropout(drop_prob)(layer3)
	layer5 = Convolution2D(nb_filter2,kernel_size_pool[0],kernel_size_conv[1],border_mode = "same")(layer4)
	layer6 = ELU()(layer5)
	layer7 = Convolution2D(nb_filter3,kernel_size_pool[0],kernel_size_conv[1],border_mode = "same")(layer6)
	layer8 = ELU()(layer7)
	layer9 = Convolution2D(nb_filter4,kernel_size_pool[0],kernel_size_conv[1],border_mode = "same")(layer8)
	layer10 = Flatten()(layer9)
	layer11 = Dense(256)(layer10)
	layer12 = ELU()(layer11)
	layer13 = Dropout(drop_prob)(layer12)
	layer14 = Dense(16)(layer13)
	layer15 = ELU()(layer14)
	layer16 = Dropout(drop_prob)(layer15)
	prediction = Dense(1)(layer16)

	model = Model(input = inputs, output = prediction)

	print("New model generated in this session according to specifications")
	print()

	model_imported = False
	weights_imported = False
	

#4 Compile model; if model.h5 available use these weigts to initialize, else random weigths
model.compile(optimizer = "Adam", loss = "binary_crossentropy", metrics = ["accuracy"])

#5 Train model
model.fit(X_train,y_train, batch_size = batch_size, nb_epoch = nb_epoch, validation_data = (X_val,y_val),shuffle = True)

#6 Save model to model.json and weights to model.h5
# Save model to json
if model_imported == True:
	print("Model not saved as model was already imported from 'model.json'")
	print()
else:
	model_json = model.to_json()
	with open("model.json","w") as f:
		json.dump(model_json,f)
	print("Model was saved to 'model.json'")
	print()

# Save weights
if weights_imported == True:
	save_weights = input("Do you wand to save the trained weights [y/n]? If you don't save the weights the training effect of this run will be lost: ")
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
	print("As this was the first round of training weights were saved to 'model.h5'")
	print()

#Print an overview of the model
print("MODEL SUMMARY")
print(model.summary())