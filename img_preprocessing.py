from scipy import misc
import pickle
import csv
import numpy as np
import pandas as pd

# Load and preprocess image (subfunction of load_preprocess_pickle_data_from_initial_file())
def load_and_preprocess_image(name,size):
	
	# Read image
	img = misc.imread(name)

	# Resize image
	img = misc.imresize(img,size)

	# Set image values to "float32"
	img = img.astype("float32")

	# Normalize image values between -1 and 1
	img = (img/127.5)-1

	return img


# If data is not already converted and stored in pickle file yet
# Load and convert data; store preprocessed data in pickle files
def load_preprocess_pickle_data_from_initial_file(image_rescale_size):
	
	# Import csv-file to be used as basic data reference
	csv_data = pd.read_csv("driving_log.csv")

	# Load and preprocess images
	X = []
	for img in csv_data["center"]:
		X.append(load_and_preprocess_image(img,image_rescale_size))
	X = np.array(X)

	# Load image labels
	y = np.array(csv_data["steering"])

	# Dump preprocessed X in pickle file for future reuse
	with open("behavioral_cloning_X.p","wb") as f:
		pickle.dump(X,f)

	# Dump preprocessed y in pickle file for future reuse
	with open("behavioral_cloning_y.p","wb") as f:
		pickle.dump(y,f)

	print("Data was converted and stored in the pickle files 'behavioral_cloning_X.p' and 'behavioral_cloning_y.p' for future use")
	print()
	return X,y


# Load data from pickle file (used if pickle files are available)
def load_data_from_pickle():
		
	# Load X from pickle file
	with open("behavioral_cloning_X.p","rb") as f:
		X = pickle.load(f)

	# Load y from pickle file
	with open("behavioral_cloning_y.p","rb") as f:
		y = pickle.load(f)

	print("Data loaded from pickle files 'behavioral_cloning_X.p' and 'behavioral_cloning_y.p'")
	print()
	return X,y