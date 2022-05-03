import glob
import os
import torch
import tqdm as tqdm
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from Predict_image import predict_images
from Trainer import load_saved_model
from Config import  SAVE_MODEL_NAME, AGE_PATH, GENDER_PATH


def age_bias(model, data_path, device):
    for age in ['young', 'old']:
        predictedLabels = []
        actualLabels = []
        print("*"*50)
        print("Predictions for images {}".format(age))
        age_path = os.path.join(data_path, age)
        labels =['N95', 'N95_valve', 'cloth', 'surgical', 'without']
        for mask in labels:
            label_configs = {
                'cloth': 0,
                'N95':  0,
                'N95_valve': 0,
                'surgical':0,
                'without': 0
            }
            complete_image_path = os.path.join(age_path, mask, "*.jpg")
            files =glob.glob(complete_image_path)
            for image_path in tqdm.tqdm(files, total=len(files)):
                pred = predict_images(image_path, model, device=device)
                label_configs[pred] += 1
            for x in label_configs:
                predictedLabels.extend([x]*label_configs[x])
            actualLabels.extend([mask]*len(files))
        print(classification_report(actualLabels, predictedLabels))
        confusion_matrix_instance = confusion_matrix(actualLabels, predictedLabels)
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_instance, display_labels=labels)
        disp.plot()

def gender_bias(model, data_path, device):
    for age in ['Male', 'Female']:
        predictedLabels = []
        actualLabels = []
        print("*"*50)
        print("Predictions for images {}".format(age))
        age_path = os.path.join(data_path, age)
        labels =['N95', 'N95_valve', 'cloth', 'surgical', 'without']
        for mask in labels:
            label_configs = {
                'cloth': 0,
                'N95':  0,
                'N95_valve': 0,
                'surgical':0,
                'without': 0
            }
            complete_image_path = os.path.join(age_path, mask, "*.jpg")
            files =glob.glob(complete_image_path)
            for image_path in tqdm.tqdm(files, total=len(files)):
                pred = predict_images(image_path, model, device=device)
                label_configs[pred] += 1
            for x in label_configs:
                predictedLabels.extend([x]*label_configs[x])
            actualLabels.extend([mask]*len(files))
        print(classification_report(actualLabels, predictedLabels))
        confusion_matrix_instance = confusion_matrix(actualLabels, predictedLabels)
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_instance, display_labels=labels)
        disp.plot()

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = load_saved_model(SAVE_MODEL_NAME, device)
    print("Gender Bias")
    gender_bias(model,GENDER_PATH,device);
    print("*" * 50)
    print("Age Bias")
    age_bias(model,AGE_PATH,device)
