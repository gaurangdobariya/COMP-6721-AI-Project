
import torch
from PIL import Image
from torchvision import transforms
from Config import SAVE_MODEL_NAME, PREDICTION_IMAGE_PATH
from Trainer import load_saved_model


def predict_images(test_path, model, device):
    f=test_path
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((224, 224)),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    labels = ['cloth', 'N95', 'N95_valve', 'surgical', 'without']
    pred_dict = {}
    image = Image.open(f).convert('RGB')
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
    model = load_saved_model(SAVE_MODEL_NAME, device)
    predict_images(PREDICTION_IMAGE_PATH, model, device)