from os import listdir
from os.path import join, isfile
import torch
from PIL import Image
from torchvision import transforms
from Config import PREDICTION_DIR_PATH, SAVE_MODEL_NAME, DIR_PATH
from Trainer import load_saved_model


def predict_images(test_path, model, device):
    files = [file for file in listdir(test_path) if isfile(join(test_path, file))]
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((224, 224)),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    labels = ['cloth', 'N95', 'N95_valve', 'surgical', 'without']
    pred_dict = {}
    for f in files:
            path = join(test_path, f)
            image = Image.open(path).convert('RGB')
            image = transform(image)
            predict = model(image.to(device).unsqueeze(0))
            pred_dict[f] = labels[predict.argmax(1)]
    for key, value in pred_dict.items():
        print(key+" -> "+ value)

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = load_saved_model(SAVE_MODEL_NAME,device)
    predict_images(PREDICTION_DIR_PATH, model, device)