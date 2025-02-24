import os
import torch.utils.data as data
from PIL import Image
import pandas as pd
import torch

class DDSM_dataset(data.Dataset):
    def __init__(self, root, view_laterality, split, transform):
        self.root = root
        self.view_laterality = view_laterality
        self.split = split
        self.transform = transform

        def process_line(line):
            mg_examination_id, lcc_img_id, rcc_img_id, lmlo_img_id, rmlo_img_id, breast_density = line.strip().split(',')
            return mg_examination_id, lcc_img_id, rcc_img_id, lmlo_img_id, rmlo_img_id, breast_density

        with open(os.path.join(self.root, 'data_frame/{}_density.txt'.format(self.split)), 'r') as f:
            self.data = list(map(process_line, f.readlines()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        series_id, lcc_img_id, rcc_img_id, lmlo_img_id, rmlo_img_id, density = self.data[idx]

        if self.view_laterality == 'lcc':
            image = Image.open(os.path.join(self.root, 'images', series_id, lcc_img_id + '.jpg')).convert('RGB')
        elif self.view_laterality == 'lmlo':
            image = Image.open(os.path.join(self.root, 'images', series_id, lmlo_img_id + '.jpg')).convert('RGB')
        elif self.view_laterality == 'rcc':
            image = Image.open(os.path.join(self.root, 'images', series_id, rcc_img_id + '.jpg')).convert('RGB')
        elif self.view_laterality == 'rmlo':
            image = Image.open(os.path.join(self.root, 'images', series_id, rmlo_img_id + '.jpg')).convert('RGB')
        elif self.view_laterality == 'left':
            cc_img = Image.open(os.path.join(self.root, 'images', series_id, lcc_img_id + '.jpg')).convert('RGB')
            mlo_img = Image.open(os.path.join(self.root, 'images', series_id, lmlo_img_id + '.jpg')).convert('RGB')
        elif self.view_laterality == 'right':
            cc_img = Image.open(os.path.join(self.root, 'images', series_id, rcc_img_id + '.jpg')).convert('RGB')
            mlo_img = Image.open(os.path.join(self.root, 'images', series_id, rmlo_img_id + '.jpg')).convert('RGB')
        else:
            lcc_img = Image.open(os.path.join(self.root, 'images', series_id, lcc_img_id + '.jpg')).convert('RGB')
            lmlo_img = Image.open(os.path.join(self.root, 'images', series_id, lmlo_img_id + '.jpg')).convert('RGB')
            rcc_img = Image.open(os.path.join(self.root, 'images', series_id, rcc_img_id + '.jpg')).convert('RGB')
            rmlo_img = Image.open(os.path.join(self.root, 'images', series_id, rmlo_img_id + '.jpg')).convert('RGB')

        if self.view_laterality == 'lcc' or self.view_laterality == 'lmlo' or self.view_laterality == 'rcc' or self.view_laterality == 'rmlo':
            image = self.transform(image)
            return image, density

        if self.view_laterality == 'left' or self.view_laterality == 'right':
            cc_img = self.transform(cc_img)
            mlo_img = self.transform(mlo_img)
            return cc_img, mlo_img, density

        else:
            lcc_img = self.transform(lcc_img)
            lmlo_img = self.transform(lmlo_img)
            rcc_img = self.transform(rcc_img)
            rmlo_img = self.transform(rmlo_img)
            return lcc_img, lmlo_img, rcc_img, rmlo_img, density


class VinDr_dataset(data.Dataset):
    def __init__(self, root, view_laterality, split, transform, file_extension='.jpg'):
        self.root = root
        self.view_laterality = view_laterality
        self.split = split
        self.transform = transform
        self.file_extension = file_extension
        
        df = pd.read_csv(os.path.join(self.root, f'mammogram_metadata_{self.split}.csv'))
        
        self.data = []
        for _, group in df.groupby('study_id'):
            study_dict = {
                'study_id': group['study_id'].iloc[0],
                'density': group['breast_density'].iloc[0]
            }
            
            has_all_views = True
            
            view_mapping = {
                ('L', 'CC'): 'lcc',
                ('L', 'MLO'): 'lmlo',
                ('R', 'CC'): 'rcc',
                ('R', 'MLO'): 'rmlo'
            }
            
            # Check and store each view
            for (lat, view), key in view_mapping.items():
                view_data = group[(group['laterality'] == lat) & (group['view_position'] == view)]
                if len(view_data) > 0:
                    study_dict[key] = view_data['image_id'].iloc[0]
                else:
                    has_all_views = False
                    break
            
            if has_all_views:
                self.data.append(study_dict)
        
        print(f"Loaded {len(self.data)} complete cases for {split} set")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        case = self.data[idx]
        study_id = case['study_id']
        density = case['density']
        
        try:
            if self.view_laterality == 'lcc':
                image = Image.open(os.path.join(self.root, 'processed_images', study_id, case['lcc'] + self.file_extension)).convert('RGB')
                image = self.transform(image)
                return image, density
                
            elif self.view_laterality == 'lmlo':
                image = Image.open(os.path.join(self.root, 'processed_images', study_id, case['lmlo'] + self.file_extension)).convert('RGB')
                image = self.transform(image)
                return image, density
                
            elif self.view_laterality == 'rcc':
                image = Image.open(os.path.join(self.root, 'processed_images', study_id, case['rcc'] + self.file_extension)).convert('RGB')
                image = self.transform(image)
                return image, density
                
            elif self.view_laterality == 'rmlo':
                image = Image.open(os.path.join(self.root, 'processed_images', study_id, case['rmlo'] + self.file_extension)).convert('RGB')
                image = self.transform(image)
                return image, density
                
            elif self.view_laterality == 'left':
                cc_img = Image.open(os.path.join(self.root, 'processed_images', study_id, case['lcc'] + self.file_extension)).convert('RGB')
                mlo_img = Image.open(os.path.join(self.root, 'processed_images', study_id, case['lmlo'] + self.file_extension)).convert('RGB')
                cc_img = self.transform(cc_img)
                mlo_img = self.transform(mlo_img)
                return cc_img, mlo_img, density
                
            elif self.view_laterality == 'right':
                cc_img = Image.open(os.path.join(self.root, 'processed_images', study_id, case['rcc'] + self.file_extension)).convert('RGB')
                mlo_img = Image.open(os.path.join(self.root, 'processed_images', study_id, case['rmlo'] + self.file_extension)).convert('RGB')
                cc_img = self.transform(cc_img)
                mlo_img = self.transform(mlo_img)
                return cc_img, mlo_img, density
                
            else:  
                lcc_img = Image.open(os.path.join(self.root, 'processed_images', study_id, case['lcc'] + self.file_extension)).convert('RGB')
                lmlo_img = Image.open(os.path.join(self.root, 'processed_images', study_id, case['lmlo'] + self.file_extension)).convert('RGB')
                rcc_img = Image.open(os.path.join(self.root, 'processed_images', study_id, case['rcc'] + self.file_extension)).convert('RGB')
                rmlo_img = Image.open(os.path.join(self.root, 'processed_images', study_id, case['rmlo'] + self.file_extension)).convert('RGB')
                
                lcc_img = self.transform(lcc_img)
                lmlo_img = self.transform(lmlo_img)
                rcc_img = self.transform(rcc_img)
                rmlo_img = self.transform(rmlo_img)
                
                return lcc_img, lmlo_img, rcc_img, rmlo_img, density
                
        except Exception as e:
            print(f"Error loading case {study_id}: {str(e)}")
            print(f"Available keys in case: {case.keys()}")
            raise e
        


class ThongNhat_dataset(data.Dataset):
    def __init__(self, root, view_laterality, split, transform=None):
        """
        Args:
            root (str): Root directory path
            view_laterality (str): 'lcc', 'lmlo', 'rcc', 'rmlo', 'left', 'right', or 'all'
            split (str): 'fold0_train', 'fold0_val', 'fold1_train', etc.
            transform: Optional transform to be applied on images
        """
        self.root = root
        self.view_laterality = view_laterality
        self.transform = transform
        self.split = split
        
        # Read the corresponding fold CSV file
        self.labels_df = pd.read_csv(os.path.join(self.root, f'ThongNhat_labels_{split}.csv'))
        
        self.density_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        
        # Create mapping from patient name to label
        self.patient_labels = {
            row['Patient']: self.density_map[row['Label']] 
            for _, row in self.labels_df.iterrows()
        }
        
        self.patients = []
        images_dir = os.path.join(root, 'processed_images')
        for patient_dir in os.listdir(images_dir):
            patient_path = os.path.join(images_dir, patient_dir)
            if os.path.isdir(patient_path):

                views = os.listdir(patient_path)
                if len(views) == 4: 
                    patient_name = patient_dir.split()[0]  
                    if patient_name in self.patient_labels:
                        self.patients.append(patient_dir)
        
        print(f"Found {len(self.patients)} complete cases")

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_dir = self.patients[idx]
        patient_path = os.path.join(self.root, 'processed_images', patient_dir)
        patient_name = patient_dir.split()[0]
        density = self.patient_labels[patient_name]
        
        try:
            if self.view_laterality == 'lcc':
                image = Image.open(os.path.join(patient_path, 'Left - CC.jpg')).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, density
                
            elif self.view_laterality == 'lmlo':
                image = Image.open(os.path.join(patient_path, 'Left - MLO.jpg')).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, density
                
            elif self.view_laterality == 'rcc':
                image = Image.open(os.path.join(patient_path, 'Right - CC.jpg')).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, density
                
            elif self.view_laterality == 'rmlo':
                image = Image.open(os.path.join(patient_path, 'Right - MLO.jpg')).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, density
                
            elif self.view_laterality == 'left':
                cc_img = Image.open(os.path.join(patient_path, 'Left - CC.jpg')).convert('RGB')
                mlo_img = Image.open(os.path.join(patient_path, 'Left - MLO.jpg')).convert('RGB')
                if self.transform:
                    cc_img = self.transform(cc_img)
                    mlo_img = self.transform(mlo_img)
                return cc_img, mlo_img, density
                
            elif self.view_laterality == 'right':
                cc_img = Image.open(os.path.join(patient_path, 'Right - CC.jpg')).convert('RGB')
                mlo_img = Image.open(os.path.join(patient_path, 'Right - MLO.jpg')).convert('RGB')
                if self.transform:
                    cc_img = self.transform(cc_img)
                    mlo_img = self.transform(mlo_img)
                return cc_img, mlo_img, density
                
            else: 
                lcc = Image.open(os.path.join(patient_path, 'Left - CC.jpg')).convert('RGB')
                lmlo = Image.open(os.path.join(patient_path, 'Left - MLO.jpg')).convert('RGB')
                rcc = Image.open(os.path.join(patient_path, 'Right - CC.jpg')).convert('RGB')
                rmlo = Image.open(os.path.join(patient_path, 'Right - MLO.jpg')).convert('RGB')
                
                if self.transform:
                    lcc = self.transform(lcc)
                    lmlo = self.transform(lmlo)
                    rcc = self.transform(rcc)
                    rmlo = self.transform(rmlo)
                    
                return lcc, lmlo, rcc, rmlo, density
                
        except Exception as e:
            print(f"Error loading patient {patient_dir}: {str(e)}")
            raise e