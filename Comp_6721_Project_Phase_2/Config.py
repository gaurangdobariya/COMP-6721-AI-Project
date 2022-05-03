##please change the Config According to configuration
#SAVE_MODEL_NAME = "phase1Model"
SAVE_MODEL_NAME = "k-foldModel"
DIR_PATH= "Dataset"
PREDICTION_DIR_PATH= "TestDataset"
PREDICTION_IMAGE_PATH = "TestDataset\\Clothing_Mask1.jpg"

GENDER_PATH = "Bias\\gender"
AGE_PATH = "Bias\\age"

CLOTH_DIR= "{}/cloth".format(DIR_PATH)
N95_DIR= "{}/N95".format(DIR_PATH)
N95_VALVE_DIR= "{}/N95_valve".format(DIR_PATH)
SURGICAL_DIR= "{}/surgical".format(DIR_PATH)
NO_MASK_DIR= "{}/without".format(DIR_PATH)