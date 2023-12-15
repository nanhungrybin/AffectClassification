import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.init
from train import *
import test
from CustomDataset import *
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# 랜덤시드 고정
torch.manual_seed(777)

if device =="cuda":
    torch.cuda.manual_seed_all(777)



######### parameter setting #########

lr = 0.0001
num_epochs = 100
batch_size = 32



# 1. CK plus
CK_train_dataset = CustomDataset(csv_file='/workspace/Affectnet/mapping_CK_annotation.csv', transform=transform, train = True )
CK_test_dataset = CustomDataset(csv_file='/workspace/Affectnet/mapping_CK_annotation.csv', transform=transform, train = False )
    
################################### 데이터 로더 생성 ###############################

CK_dataloader_train = DataLoader(CK_train_dataset, batch_size = batch_size, shuffle=True)
CK_dataloader_test = DataLoader(CK_test_dataset, batch_size = batch_size, shuffle=False)

###################################################################################

total_samples = len(CK_dataloader_train.dataset)
total_batch = total_samples // batch_size

######## model load ############
# resnet18_pretrained = models.resnet18(pretrained = True).to(device)
resnet18_pretrained = models.resnet18(weights="DEFAULT")
####### optim, loss function #######
optimizer = torch.optim.Adam(resnet18_pretrained.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss().to(device)


params = {
    'num_epochs':num_epochs,
    "optimizer" : optimizer,
    "loss_function" :loss_function,
    "train_dataloader":CK_dataloader_train,
    "test_dataloader" : CK_dataloader_test,
    "device": device,
    "total_batch" : total_batch 
}


loss_function=params["loss_function"]
train_dataloader=params["train_dataloader"]
test_dataloader = params["test_dataloader"]


if __name__ == "__main__": # 객체를 불러 오는 것은 main함수에

    

    # output layer를 현재 data에 맞게 수정 
    num_classes = 8     # 원래 7 basic emotion이지만 neutral이 없음. 그런데 보통 다른 emotion data에는  
    num_features = resnet18_pretrained.fc.in_features
    resnet18_pretrained.fc = nn.Linear(num_features, num_classes)

    # model load
    model = resnet18_pretrained.to(device)


    for epoch in range(num_epochs) :
        model, loss, accuracy = train(model, train_dataloader, epoch, num_epochs, optimizer, loss_function, total_batch, device)
          
        # 이때가 되면 training된 model의 output을 test를 하는 것임
        # if epoch % 10 == 0:
        #     test.test(model, test_dataloader, device) 
    all_preds, all_labels = test.test(model, test_dataloader, device)  


    # confusion matrix 계산
    cm = confusion_matrix(all_labels, all_preds)

    # confusion matrix 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
 





