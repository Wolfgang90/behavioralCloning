from scipy import misc
import pickle
import csv
import numpy as np

# Preprocess image:
def preprocess_image(name,size):
	img = misc.imread(name)
	img = misc.imresize(img,size)
	img = img.astype("float32")
	img = (img/255)-0.5
	img = img[np.newaxis,:]
	return img


def load_data_from_pickle():
		
	with open("behavioral_cloning_X.p","rb") as f:
		X = pickle.load(f)

	with open("behavioral_cloning_y.p","rb") as f:
		y = pickle.load(f)
	print("Data was loaded from pickle files 'behavioral_cloning_X.p' and 'behavioral_cloning_y.p'")
	print()
	return X,y

	# If required data was not converted yet, it is converted and eventually stored in a pickle file
def load_preprocess_pickle_data_from_initial_file():
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
	X = preprocess_image(image_name[0],(32,32))

	# Load all other image files
	for i in image_name[1:]:
		X = np.concatenate((X, preprocess_image(i,(32,32))), axis = 0)

	with open("behavioral_cloning_X.p","wb") as f:
		pickle.dump(X,f)
	with open("behavioral_cloning_y.p","wb") as f:
		pickle.dump(y,f)
	print("Data was converted and stored in the pickle files 'behavioral_cloning_X.p' and 'behavioral_cloning_y.p' for future use")
	print()
	return X,y