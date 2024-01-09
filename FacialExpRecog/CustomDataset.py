from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform = None, train = True):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.train = train

        self.count = 0
        
        # if self.train:
        #     self.data, _ = train_test_split(self.data, test_size=0.3, random_state=42)

            
    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx].path
        
        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, SyntaxError) as e:
            print(f"Error opening image {img_path}: {e}")
            self.count += 1
            print(self.count)
            # If an error occurs, move to the next image
            return self.__getitem__((idx + 1) % len(self))

        label = self.data.iloc[idx].label
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
# # ToTensor를 리스트로 감싸주기
# transform = transforms.Compose([transforms.ToTensor()])

# 이미지 크기를 조정하는 변환 추가
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])