import tensorflow as tf
import numpy as np
import csv
import pickle
from scipy import misc
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

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



#2 Split data into training and validation data

X, y = shuffle(X,y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train, test_size = 0.2)

print("The final training, validation and test set sizes are:")
print()
print("Train set size: {}".format(X_train.shape))
print("Validation set size: {}".format(X_val.shape))
print("Test set size: {}".format(X_test.shape))



#3 Define model

#4 Compile model; if model.h5 available use these weigts to initialize, else random weigths

#5 Train model

#6 Save model to model.json and weights to model.h5
