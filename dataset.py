import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class WasteDataset(Dataset):
    def __init__(self, root_dir_1, root_dir_2, split, transform=None):
        self.transform = transform
        self.classes = ['Non-Recyclable', 'Recyclable']
        self.image_paths = []
        self.labels = []
    
        if os.path.exists(root_dir_1):
            raw_classes_1 = sorted(os.listdir(root_dir_1))
            recyclable_folders_1 = [
                'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 
                'cardboard_boxes', 'cardboard_packaging', 'glass_beverage_bottles', 
                'glass_cosmetic_containers', 'glass_food_jars', 'magazines', 
                'newspaper', 'office_paper', 'paper_cups', 'plastic_detergent_bottles', 
                'plastic_food_containers', 'plastic_water_bottles', 'steel_food_cans'
            ]
            
            for class_name in raw_classes_1:
                class_dir = os.path.join(root_dir_1, class_name)
                binary_label = 1 if class_name in recyclable_folders_1 else 0
                
                for subfolder in ['default', 'real_world']:
                    subfolder_dir = os.path.join(class_dir, subfolder)
                    if not os.path.exists(subfolder_dir):
                        continue
                        
                    image_names = os.listdir(subfolder_dir)
                    image_names.sort()
                    random.seed(42)
                    random.shuffle(image_names)
                    
                    num_images = len(image_names)
                    idx_70 = int(0.7 * num_images)
                    idx_85 = int(0.85 * num_images)
                    
                    if split == 'train':
                        selected_images = image_names[:idx_70]
                    elif split == 'val':
                        selected_images = image_names[idx_70:idx_85]
                    else:
                        selected_images = image_names[idx_85:]
                        
                    for img_name in selected_images:
                        self.image_paths.append(os.path.join(subfolder_dir, img_name))
                        self.labels.append(binary_label)

        if os.path.exists(root_dir_2):
            raw_classes_2 = sorted(os.listdir(root_dir_2))
            
            # Map specific V2 folders into standard Recyclable category rules
            recyclable_folders_2 = ['Metal', 'Glass', 'Paper', 'Cardboard', 'Plastic']
            
            for class_name in raw_classes_2:
                class_dir = os.path.join(root_dir_2, class_name)
                if not os.path.isdir(class_dir):
                    continue
                    
                binary_label = 1 if class_name in recyclable_folders_2 else 0
                
                image_names = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                image_names.sort()
                random.seed(42)
                random.shuffle(image_names)
                
                num_images = len(image_names)
                idx_70 = int(0.7 * num_images)
                idx_85 = int(0.85 * num_images)
                
                if split == 'train':
                    selected_images = image_names[:idx_70]
                elif split == 'val':
                    selected_images = image_names[idx_70:idx_85]
                else:
                    selected_images = image_names[idx_85:]
                    
                for img_name in selected_images:
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(binary_label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

if __name__ == '__main__':
    PATH_1 = "./data/recycling/images/images"
    PATH_2 = "./data/garbage_v2/original"  # Fixed target location
    basic_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    print("Testing dual dataset mapping...")
    train_dataset = WasteDataset(PATH_1, PATH_2, split='train', transform=basic_transform)
    print(f"Combined Training images loaded successfully: {len(train_dataset)}")
