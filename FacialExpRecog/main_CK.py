import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.init
from train import *
import test
from CustomDataset import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
import numpy as np
import seaborn as sns
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"

# 랜덤시드 고정
torch.manual_seed(777)

if device =="cuda":
    torch.cuda.manual_seed_all(777)



######### parameter setting #########

# lr = 0.0001
# num_epochs = 100
# batch_size = 32


################# hyperparameter tunning with Wandb ###############
wandb.init(
    name = "Resnet18-100epochs-0.0001",
    project="affect_classification", 
    
    config = {
    "learning_rate": 0.0001,
    "num_epochs": 30,
    "batch_size": 32,
    "dropout": 0.2,
    "architecture": "Resnet18",
    "dataset": "CK plus"
})

config = wandb.config


# 1. CK plus
CK_train_dataset = CustomDataset(csv_file='/workspace/Affectnet/mapping_CK_annotation.csv', transform=transform, train = True )
CK_test_dataset = CustomDataset(csv_file='/workspace/Affectnet/mapping_CK_annotation.csv', transform=transform, train = False )
    
################################### 데이터 로더 생성 ###############################

CK_dataloader_train = DataLoader(CK_train_dataset, batch_size = config.batch_size, shuffle=True)
CK_dataloader_test = DataLoader(CK_test_dataset, batch_size = config.batch_size, shuffle=False)

###################################################################################

total_samples = len(CK_dataloader_train.dataset)
total_batch = total_samples // config.batch_size

######## model load ############
# resnet18_pretrained = models.resnet18(pretrained = True).to(device)
resnet18_pretrained = models.resnet18(weights="DEFAULT")
####### optim, loss function #######
optimizer = torch.optim.Adam(resnet18_pretrained.parameters(), lr = config.learning_rate)
loss_function = nn.CrossEntropyLoss().to(device)


params = {
    'num_epochs': config.num_epochs,
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

    # Weights file 저장경로
    save_path = f'/workspace/weights/{config.architecture}_{config.dataset}_lr{config.learning_rate}_epochs{config.num_epochs}_batch{config.batch_size}.pth'

    # output layer를 현재 data에 맞게 수정 
    num_classes = 7     # 원래 7 basic emotion이지만 neutral이 없음. 그런데 보통 다른 emotion data에는 neutral 포함 7 class
    num_features = resnet18_pretrained.fc.in_features
    resnet18_pretrained.fc = nn.Linear(num_features, num_classes)

    # model load
    model = resnet18_pretrained.to(device)

    ################## 학습과정 들어가기 전에 실시간으로 모델 tracking ##############
    wandb.watch(model)


    for epoch in range(config. num_epochs) :
        model, loss, accuracy = train(model, train_dataloader, epoch, config.num_epochs, optimizer, loss_function, total_batch, device)
          
        ################### wandb log 에 남길값 지정#################################
        # 각 epoch 별로 진행

        wandb.log({"acc":accuracy, "loss":loss})

        #########################################################
    all_preds, all_labels, accuracy = test.test(model, test_dataloader, device)  

    print(f"최종 TEST 성능 : {accuracy}")


    # 정밀도 계산
    precision = precision_score(all_labels, all_preds, average='macro')  # 여기에서 'macro'는 클래스별 정밀도의 평균을 의미합니다.
    
    # 리콜 계산
    recall = recall_score(all_labels, all_preds, average='macro')  # 여기에서 'macro'는 클래스별 리콜의 평균을 의미합니다.


    # confusion matrix 계산
    cm = confusion_matrix(all_labels, all_preds)


    # 각 클래스별로 맞춘 개수 출력
    for i in range(confusion_matrix.shape[0]):
        correct_samples = confusion_matrix[i, i]
        print(f"Class {i}: {correct_samples} samples correctly predicted.")

    # 각 클래스별 정확도 계산
    class_accuracy = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)

    # 각 클래스별 정확도 출력
    for i, acc in enumerate(class_accuracy):
        print(f"Class {i} Accuracy: {acc:.4f}")



    # confusion matrix 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
 

# Iteration 은 각 에포크 내에서 일어나는 단일 업데이트 단계
# 훈련 데이터가 큰 경우, 전체 데이터를 한 번에 처리하지 않고 작은 미니배치(mini-batch)로 나누어 학습을 수행. 이 미니배치에 대한 단일 업데이트 단계가 Iteration
# = > 1000개의 훈련 샘플이 있고 미니배치 크기가 100이라면, 에포크당 10개의 Iteration

# 에포크는 전체 데이터에 대한 한 번의 학습 주기이고, Iteration 은 각 학습 주기에서 모델 가중치를 업데이트하기 위한 단일 단계





