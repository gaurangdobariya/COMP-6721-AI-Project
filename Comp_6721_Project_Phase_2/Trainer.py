# Importing libraries
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from COMP_6721_CNN import COMP_6721_CNN
from Config import SAVE_MODEL_NAME, DIR_PATH
from sklearn.model_selection import KFold


# This function takes a device and build model
def create_model(device):
    print("---- Bulding Model ----")
    model = COMP_6721_CNN()
    model = model.to(device)
    entropyLoss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    return model, optimizer, entropyLoss


def test_model(model, testing_loader,device):
    print("=== Testing model ===")
    model.eval()
    predictions_list = []
    accurate_list = []
    with torch.no_grad():
        for data_chunk in testing_loader:
            images, labels = data_chunk
            images, labels = images.to(device), labels.to(device)
            _, pred_values = torch.max(model(images), dim=1)
            predictions_list.extend(pred_values.detach().cpu().numpy())
            accurate_list.extend(labels.detach().cpu().numpy())
   # test_accuracy = ((np.array(accurate_list)==np.array(predictions_list)).sum().item()/len(accurate_list))*100
    training_accuracy = torch.tensor(torch.sum(pred_values ==labels ).item() / len(pred_values))
           # training_accuracy = torch.tensor(torch.sum(prediction_values == labels).item() / len(prediction_values))

    print('Accuracy : {}'.format(training_accuracy))
    

def build_KFold_Iterator(dataset, model ,loss_criteria, optimizer,device):
    kfold = KFold(n_splits=10, shuffle=True, random_state=None)
    fold_val = 1
    for training_idx, testing_idx in kfold.split(dataset):
        print("Running Fold Num: ", fold_val)
        training_dataset = Subset(dataset, training_idx)
        testing_dataset = Subset(dataset, testing_idx)
        training_loader = DataLoader(training_dataset, batch_size=64, num_workers=0,shuffle=True)
        testing_loader = DataLoader(testing_dataset, batch_size=64, num_workers=0, shuffle=True)
        return_model = train_model(model, loss_criteria, optimizer, training_loader, device)
        reportTitle = "fold :"+str(fold_val)
        generate_classification_report(model,reportTitle,training_loader,device)
        test_model(model, testing_loader,device)
        fold_val+=1
    return return_model, training_loader



# This function takes a model, entropy loss, optimizer and training loader to
# train the model, after 10 epoches, model is trained and return model for evaluation.
def train_model(model, entropyloss, optimizer, train_loader, device):
    print("---- Training Model ----")
    for i in range(5):
        model.train()
        training_accuracy_list = []
        training_loss_list = []

        for data_chunk in train_loader:
            images, labels = data_chunk
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            training_loss = entropyloss(outputs, labels.long())
            _, prediction_values = torch.max(outputs, dim=1)
            training_accuracy = torch.tensor(torch.sum(prediction_values == labels).item() / len(prediction_values))
            training_accuracy_list.append(training_accuracy)
            training_loss_list.append(training_loss)
            training_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("Epoch: {}, training loss: {}, training accuracy: {}".format(
            i+1, torch.stack(training_loss_list).mean().item(), torch.stack(training_accuracy_list).mean().item()
        ))
    return model


# This function is used to save the generated model
def saveModel(model,modelName):
    print("---- Saving Model ----")
    torch.save(model.state_dict(), modelName)

# This function takes directory and model name to load the stored model
def load_saved_model(modelDirPath, device):
    print("---- Loading saved Model ----")
    loaded_model = torch.load(modelDirPath, map_location=device)
    model = COMP_6721_CNN()
    model.load_state_dict(loaded_model)
    model = model.to(device)
    return model


# This function is used to generate classification_report and confusion_matrix with given data
def generate_classification_report(model, reportTitle, data_loader, device):
    print("---- Generating Classification Report ----")
    model.eval()
    accuracies = []
    predictions = []
    with torch.no_grad():
        for data_chunk in data_loader:
            images, labels = data_chunk
            images, labels = images.to(device), labels.to(device)
            _, pred_values = torch.max(model(images), dim=1)
            predictions.extend(pred_values.detach().cpu().numpy())
            accuracies.extend(labels.detach().cpu().numpy())
    print("{} Classification Report".format(reportTitle))
    labels = ['N95', 'N95_valve', 'cloth', 'surgical', 'without']
    print(classification_report(accuracies, predictions, target_names=labels))
    print("---- Confusion Matrix ----")
    confusion_matrix_instance = confusion_matrix(accuracies, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_instance, display_labels = labels)
    disp.plot()
    
    


# Loads the data from the given path and splits it into training and validation sets
def get_data(images_path, test_split=0.25):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.RandomHorizontalFlip(0.2),
         transforms.Resize((224, 224)),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    dataset = ImageFolder(images_path, transform=transform)
    print(dataset.classes)
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_split)
    train = Subset(dataset, train_idx)
    test = Subset(dataset, test_idx)
    return train, test,dataset


# This function is treated as a main function to run the program. it  takes
# one boolean parameter is_model_saved and a directory path to store the model. If trianed model is available then it uses that model else trained new one from the data
def run_program(modelDirPath, is_model_saved=True):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')            
    train, test,dataset = get_data(modelDirPath)
    train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)
    if not is_model_saved:
        model, optimizer, entropyLoss = create_model(device)
        #model = train_model(model, entropyLoss, optimizer, train_loader, device)
        model, training_loader = build_KFold_Iterator(train, model, entropyLoss,optimizer,device)
        saveModel(model, SAVE_MODEL_NAME)
    model = load_saved_model(SAVE_MODEL_NAME, device)
    generate_classification_report(model, "Training", train_loader, device)
    generate_classification_report(model, "Testing", test_loader, device)
    


if __name__ == '__main__':
    torch.cuda.empty_cache()
    run_program(DIR_PATH, is_model_saved=True)
