import os
import random
import numpy as np
import PIL
from PIL import Image
from einops import rearrange, repeat
import re
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .UniASET_constants import TASKS_GROUP_DICT, TASKS
from .augmentation import RandomHorizontalFlip, FILTERING_AUGMENTATIONS, RandomCompose, Mixup
from .utils import crop_arrays, SobelEdgeDetector

class UniASETBaseDataset(Dataset):
    def __init__(self, root_dir, tasks, base_size=(256, 256), img_size=(224, 224), seed=None, precision='fp32'):
        super().__init__()
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        self.data_root = root_dir 
        self.s = None
        self.n_joints = 20
        self.tasks = tasks
        self.subtasks = []
        for task in tasks:
            if task in TASKS_GROUP_DICT:
                self.subtasks += TASKS_GROUP_DICT[task]
            else:
                self.subtasks += [task]
                
        self.base_size = base_size
        self.img_size = img_size
        self.precision = precision
        

         # generate path list for pose train set
        self.label_paths_train_pose = []
        for label_dir in sorted(os.listdir(os.path.join(self.data_root, 'crop_ak_htm_train_split2'))):
            for label_path in sorted(os.listdir(os.path.join(self.data_root, 'crop_ak_htm_train_split2', label_dir))):
                self.label_paths_train_pose.append(os.path.join(label_dir, label_path))

        self.path_dict_train_pose = {}

        for j, label_path in enumerate(self.label_paths_train_pose):
            joint_id = label_path.split('/')[1].split('_')[3]
            animal_class = label_path.split('/')[0]
            if joint_id not in self.path_dict_train_pose:
                self.path_dict_train_pose[joint_id] = {}
            if animal_class not in self.path_dict_train_pose[joint_id]:
                self.path_dict_train_pose[joint_id][animal_class] = []
            self.path_dict_train_pose[joint_id][animal_class].append(j)

        # generate path list for pose test set
        self.label_paths_test_pose = []

        for label_dir in sorted(os.listdir(os.path.join(self.data_root, 'crop_ak_htm_test_split2'))):
            for label_path in sorted(os.listdir(os.path.join(self.data_root, 'crop_ak_htm_test_split2', label_dir))):
                self.label_paths_test_pose.append(os.path.join(label_dir, label_path))

        self.path_dict_test_pose = {}
        for j, label_path in enumerate(self.label_paths_test_pose):
            joint_id = label_path.split('/')[1].split('_')[3]
            animal_class = label_path.split('/')[0]
            if joint_id not in self.path_dict_test_pose:
                self.path_dict_test_pose[joint_id] = {}
            if animal_class not in self.path_dict_test_pose[joint_id]:
                self.path_dict_test_pose[joint_id][animal_class] = []
            self.path_dict_test_pose[joint_id][animal_class].append(j)
        
       
        # generate path list for mask train/test set
        self.label_paths_mask = os.listdir(os.path.join(self.data_root, 'oxford-iiit-pet','annotations256'))
        self.image_paths_mask = os.listdir(os.path.join(self.data_root, 'oxford-iiit-pet','images256'))

        self.path_dict_mask = {}
        for i, img_path in enumerate(self.image_paths_mask):
            sub_class = '_'.join(img_path.split('_')[:-1])
            if sub_class not in self.path_dict_mask:
                self.path_dict_mask[sub_class] = []
            self.path_dict_mask[sub_class].append(i)
        
        # Define the list of classes you want to move to the test set
        classes_to_move_to_test = ['Siamese', 'Bombay', 'english_setter', 'Ragdoll', 'havanese']

        # Initialize empty dictionaries for train and test sets
        self.path_dict_test_mask = {}
        self.path_dict_train_mask = {}

        # Iterate through the original dictionary and split into train and test sets
        for class_name, paths in self.path_dict_mask.items():
            if class_name in classes_to_move_to_test:
                self.path_dict_test_mask[class_name] = paths
            else:
                self.path_dict_train_mask[class_name] = paths
        
        self.image_paths_test_mask = []
        for subclass in self.path_dict_test_mask:
            for idx in self.path_dict_test_mask[subclass]:
                self.image_paths_test_mask.append(idx)

    def load_img(self, label_path, task='keypoints2d', test = False):
       
        if task == 'keypoints2d' or task == 'cls':
            animal_class = label_path.split('/')[0]
            label_name = label_path.split('/')[1]
            img_idx =label_name.split('_')[1]
            img_name = "imgid_{}_rgb.png".format(img_idx)
            if test == True:
               
                img_path = os.path.join(self.data_root, 'crop_ak_rgb_test_split2', animal_class, img_name)
            else:
                img_path = os.path.join(self.data_root, 'crop_ak_rgb_train_split2', animal_class, img_name)
            try:
                # open image file
              
                img = Image.open(img_path)
                img = img.resize((224, 224))
                img = np.asarray(img)
                
                # type conversion
                img = img.astype('float32') / 255
                
                # shape conversion
                img = np.transpose(img, (2, 0, 1))
                success = True
                
            except PIL.UnidentifiedImageError:
                print(f'PIL Error on {img_path}')
                img = -np.ones(3, 80, 80).astype('float32')
                success = False
            
            return img, success
        
        elif task == 'mask':
            img_path = os.path.join(self.data_root, 'oxford-iiit-pet','images256', label_path)
            try:
                # open image file
                img = Image.open(img_path)
                img = img.resize((224, 224))
                img = np.asarray(img.convert("RGB"))
                
                # type conversion
                img = img.astype('float32') / 255

                # shape conversion
                img = np.transpose(img, (2, 0, 1))
                success = True
                
            except:
                print(f'PIL Error on {img_path}')
                # img = -np.ones(3, 80, 80).astype('float32')
                success = False

        return img, success


    def load_label(self, task, label_path, flag = False ):
        if task == 'keypoints2d' or task == 'cls' :

            if flag == True:
                # task_root = os.path.join(self.data_root, 'crop_ak_htm_train_split2')
                task_root = os.path.join(self.data_root, 'crop_ak_htm_test_split2')
            else:
                task_root = os.path.join(self.data_root, 'crop_ak_htm_train_split2')
            
            label_path = os.path.join(task_root, label_path)
  
            # open label file
            label = Image.open(label_path)
            label = label.resize((224, 224))
            label = np.asarray(label) 
            
            # type conversion
            label = label.astype('float32')

            # shape conversion
            if label.ndim == 2:
                label = label[np.newaxis, ...]

            elif label.ndim == 3:
                label = np.transpose(label, (2, 0, 1))
                
            
            return label/255

        elif task == 'mask':
            task_root = os.path.join(self.data_root, 'oxford-iiit-pet' ,'annotations256')
    
            label_path = os.path.join(task_root, label_path)

            # open label file
            label = Image.open(label_path)
    
            label = label.resize((224, 224))

            label = np.asarray(label)
            
            # type conversion
            label = label.astype('float32')

            # shape conversion
            if label.ndim == 2:
                label = label[np.newaxis, ...]

            elif label.ndim == 3:
                label = np.transpose(label, (2, 0, 1))
                
            return label
        
        
    def load_task(self, task, label_path, test = False):
            
        label = self.load_label(task, label_path, test)

        mask = np.ones_like(label)
            
        return label, mask
    
    def choose_support_pose(self, path_idxs, path_list, random_class):
           
            remaining_elements_needed = max(0, self.shot - len(path_idxs))

            # Determine the number of elements to take from path_list[random_class] in order
            elements_to_take = min(remaining_elements_needed, len(path_list[random_class]))

            # Take elements from path_list[random_class] in order
            selected_elements = path_list[random_class][:elements_to_take]

            # If more elements are needed, repeat the data
            while len(selected_elements) < remaining_elements_needed:
                num_repeats = min(remaining_elements_needed - len(selected_elements), len(path_list[random_class]))
                selected_elements = np.concatenate((selected_elements, path_list[random_class][:num_repeats]))

            # Concatenate the selected elements to path_idxs
            path_idxs = np.concatenate((path_idxs, selected_elements))

            return path_idxs

    def choose_support_mask(self, path_idxs, path_list):
        
        remaining_elements_needed = max(0, self.shot - len(path_idxs))

        # Determine the number of elements to take from path_list[random_class] in order
        elements_to_take = min(remaining_elements_needed, len(path_list))

        # Take elements from path_list[random_class] in order
        selected_elements = path_list[:elements_to_take]

        # If more elements are needed, repeat the data
        while len(selected_elements) < remaining_elements_needed:
            num_repeats = min(remaining_elements_needed - len(selected_elements), len(path_list))
            selected_elements = np.concatenate((selected_elements, path_list[:num_repeats]))

        # Concatenate the selected elements to path_idxs
        path_idxs = np.concatenate((path_idxs, selected_elements))

        return path_idxs

    
    def preprocess_default(self, labels, masks, channels):
        labels = torch.from_numpy(labels).float()

        if masks is not None:
            masks = torch.from_numpy(masks).float().expand_as(labels)
        else:
            masks = torch.ones_like(labels)
            
        labels = labels[:, channels]
        masks = masks[:, channels]
            
        return labels, masks

    def preprocess_batch(self, task, imgs, labels, masks, channels=None, drop_background=True):
        imgs = torch.from_numpy(imgs).float()

        # process all channels if not given
        if channels is None:
            if task in TASKS_GROUP_DICT:
                channels = range(len(TASKS_GROUP_DICT[task]))
            else:
                raise ValueError(task)
        labels, masks = self.preprocess_default(labels, masks, channels)

        # ensure label values to be in [0, 1]
        labels = labels.clip(0, 1)
        
        # precision conversion
        if self.precision == 'fp16':
            imgs = imgs.half()
            labels = labels.half()
            masks = masks.half()
        elif self.precision == 'bf16':
            imgs = imgs.bfloat16()
            labels = labels.bfloat16()
            masks = masks.bfloat16()

        return imgs, labels, masks


class UniASETHybridDataset(UniASETBaseDataset):
    def __init__(self, root_dir, tasks, shot, tasks_per_batch, domains_per_batch,
                 image_augmentation, unary_augmentation, binary_augmentation, mixed_augmentation, dset_size=-1, **kwargs):
        super().__init__(root_dir, tasks, **kwargs)
        
        assert shot > 0
        self.shot = shot
        self.tasks_per_batch = tasks_per_batch

        self.domains_per_batch = domains_per_batch
        self.dset_size = dset_size
        
        if image_augmentation:
            self.image_augmentation = RandomHorizontalFlip()
        else:
            self.image_augmentation = None
        
        if unary_augmentation:
            self.unary_augmentation = RandomCompose(
                [augmentation(**kwargs) for augmentation, kwargs in FILTERING_AUGMENTATIONS.values()],
                p=0.8,
            )
        else:
            self.unary_augmentation = None

        if binary_augmentation is not None:
            self.binary_augmentation = Mixup()
        else:
            self.binary_augmentation = None

        self.mixed_augmentation = mixed_augmentation


    def __len__(self):
        return 9999
    

    def sample_batch(self, task, channel):
        # sample data paths
        if task == 'keypoints2d':
            joints = np.random.choice(range(self.n_joints), 1 , replace=False)
            # sample image path indices in each 
            path_idxs = np.array([], dtype=np.int64)
            for joint in joints:      
                path_list = self.path_dict_train_pose[str(joint)]
                random_classes = random.sample(list(path_list.keys()), 1)                
                for random_class in random_classes:
                    path_idxs = self.choose_support_pose(path_idxs, path_list, random_class)
                    path_idxs= np.concatenate((path_idxs,np.random.choice(path_list[random_class], 
                                                            self.shot, replace=True)))
            # load images and labels
            imgs = []
            labels = []
            masks = []
            label_path_list = []

            for path_idx in path_idxs:
                # index image path
                label_path = self.label_paths_train_pose[int(path_idx)]
                label_path_list.append(label_path)
                # load image, label, and mask
                img, success = self.load_img(label_path, task)
                label, mask = self.load_task(task, label_path)
                if not success:
                    mask = np.zeros_like(label)
                    
                imgs.append(img)
                labels.append(label)
                masks.append(mask)
                
            # form a batch
            imgs = np.stack(imgs)
            labels = np.stack(labels) if labels[0] is not None else None
            masks = np.stack(masks) if masks[0] is not None else None
            
            # preprocess and make numpy arrays to torch tensors
            imgs, labels, masks = self.preprocess_batch(task, imgs, labels, masks, [channel])

            return imgs, labels, masks, label_path_list
        
        if task == 'mask':
            sub_class = random.sample(self.path_dict_train_mask.keys(), 1)
            
            # sample image path indices in each 
            path_idxs = np.array([], dtype=np.int64)
            path_list = self.path_dict_train_mask[sub_class[0]]            
            path_idxs = self.choose_support_mask(path_idxs, path_list)
            path_idxs= np.concatenate((path_idxs,np.random.choice(path_list, 
                                                    self.shot, replace=True)))
            # load images and labels
            imgs = []
            labels = []
            masks = []
            label_path_list = []

            for path_idx in path_idxs:
                # index image path
                label_path = self.label_paths_mask[int(path_idx)]
                label_path_list.append(label_path)
                # load image, label, and mask
                
                img, success = self.load_img(label_path, task)
                label, mask = self.load_task(task, label_path)
                if not success:
                    mask = np.zeros_like(label)
                    
                imgs.append(img)
                labels.append(label)
                masks.append(mask)


            # form a batch
            imgs = np.stack(imgs)
            labels = np.stack(labels) if labels[0] is not None else None
            masks = np.stack(masks) if masks[0] is not None else None
            
            # preprocess and make numpy arrays to torch tensors
            imgs, labels, masks = self.preprocess_batch(task, imgs, labels, masks, [channel])

            return imgs, labels, masks, label_path_list
        
        if task == 'cls':
            joints = np.random.choice(range(self.n_joints), 1 , replace=False)

            # sample image path indices in each 
            path_idxs = np.array([], dtype=np.int64)
            for joint in joints:      
                path_list = self.path_dict_train_pose[str(joint)]
                flag = np.random.random()
                random_classes_support = random.sample(list(path_list.keys()), 1)
                if(flag < 0.5):
                    random_classes_query =  random.sample(list(path_list.keys()), 1)
                    cls_gt = 0      
                else:
                    random_classes_query = random_classes_support
                    cls_gt = 1
            
            for support_class, query_class in zip(random_classes_support, random_classes_query):
               
                path_idxs = self.choose_support_pose(path_idxs, path_list, support_class)
                path_idxs= np.concatenate((path_idxs,np.random.choice(path_list[query_class], 
                                                        self.shot, replace=True)))
            # load images and labels
            imgs = []
            labels = []
            masks = []
            label_path_list = []

            for path_idx in path_idxs:
                # index image path
                label_path = self.label_paths_train_pose[int(path_idx)]
                label_path_list.append(label_path)
              
                # load image, label, and mask
                img, success = self.load_img(label_path, task)
                label, mask = self.load_task(task, label_path)

                if not success:
                    mask = np.zeros_like(label)
                    
                imgs.append(img)
                labels.append(label)
                masks.append(mask)
                
            # form a batch
            imgs = np.stack(imgs)
            labels = np.stack(labels) if labels[0] is not None else None
            masks = np.stack(masks) if masks[0] is not None else None
          
            N, C, H, W = labels.shape
            cls_gts = np.full((N, 1, H, W), cls_gt)
            labels = cls_gts
            # preprocess and make numpy arrays to torch tensors
            imgs, labels, masks = self.preprocess_batch(task, imgs, labels, masks, [channel])
       
            return imgs, labels, masks, label_path_list          
           
    def sample_tasks(self):
        # sample subtasks
        replace = len(self.subtasks) < self.tasks_per_batch
        subtasks = np.random.choice(self.subtasks, self.tasks_per_batch, replace=replace)
        
        # parse task and channel from the subtasks
        tasks = []
        channels = []
        for subtask in subtasks:
       
            task = '_'.join(subtask.split('_')[:-1])
            channel = int(subtask.split('_')[-1])
            tasks.append(task)
            channels.append(channel)
        
        return tasks, channels
    
    def __getitem__(self, idx):
        # sample tasks
        tasks, channels = self.sample_tasks()
        if self.binary_augmentation is not None:
            tasks_aux, channels_aux = self.sample_tasks()
            
        X = []
        Y = []
        M = []
        t_idx = []
  
        for i in range(self.tasks_per_batch):
            # sample a batch of images, labels, and masks for each task
   
            X_, Y_, M_, label_path_list = self.sample_batch(tasks[i], channels[i])
           
            # apply image augmentation
            if self.image_augmentation is not None:
                (X_, Y_, M_), image_aug = self.image_augmentation(X_, Y_, M_, get_augs=True)
            else:
                image_aug = lambda x: x
            
            X.append(X_)
            Y.append(Y_)
            M.append(M_)
            
            t_idx.append(TASKS.index(f'{tasks[i]}_{channels[i]}'))

        # form a global batch
        X = torch.stack(X)
        Y = torch.stack(Y)
        M = torch.stack(M)
       
        # task and task-group index
        t_idx = torch.tensor(t_idx)

        return X, Y, M, t_idx, label_path_list
    
class UniASETContinuousDataset(UniASETBaseDataset):
    def __init__(self, root_dir, task, channel_idx=-1, dset_size=-1, image_augmentation=False, shot = 5, **kwargs):
        super().__init__(root_dir, [task], **kwargs) 
        self.task = task
        self.channel_idx = channel_idx
        self.dset_size = dset_size
        self.n_channels = len(TASKS_GROUP_DICT[task])
        self.shot = shot 
        if image_augmentation:
            self.image_augmentation = RandomHorizontalFlip()
        else:
            self.image_augmentation = None
        
    def __len__(self):
        if self.dset_size > 0:
            return self.dset_size
        else:
            if self.task == 'mask':
                return len(self.image_paths_test_mask)
            else:       
                return len(self.label_paths_test_pose)
    
    def sample_batch(self, task, channel, class_name, joint_id = 0, path_idxs=None, cls_gt = None):

         # load images and labels
        if task == 'keypoints2d':
            imgs = []
            labels = []
            masks = []
            label_path_list = []
            path_idxs = np.array([], dtype=np.int64)
            path_list = self.path_dict_test_pose[str(joint_id)] 
            if class_name not in path_list:
                return  imgs, labels, masks, label_path_list, False
            path_idxs = self.choose_support_pose(path_idxs, path_list, class_name)
        
            for path_idx in path_idxs:
                # index image path
                label_path = self.label_paths_test_pose[int(path_idx)]
                label_path_list.append(label_path)
                # load image, label, and mask

                img, success = self.load_img(label_path, task, test=True)
                label, mask = self.load_task(task, label_path, test=True)
                if not success:
                    mask = np.zeros_like(label)
                imgs.append(img)
                labels.append(label)
                masks.append(mask)    
            # form a batch
            imgs = np.stack(imgs)
            labels = np.stack(labels) if labels[0] is not None else None
            masks = np.stack(masks) if masks[0] is not None else None
            
            # preprocess and make numpy arrays to torch tensors
            imgs, labels, masks = self.preprocess_batch(task, imgs, labels, masks, [channel])
        
            return imgs, labels, masks, label_path_list, True
        
        if task == 'mask':
            imgs = []
            labels = []
            masks = []
            label_path_list = []
            path_idxs = np.array([], dtype=np.int64)
            path_list = self.path_dict_test_mask[class_name] 

            path_idxs = self.choose_support_mask(path_idxs, path_list)

            for path_idx in path_idxs:
                # index image path
                img_path = self.image_paths_mask[int(path_idx)]
                label_path = self.label_paths_mask[int(path_idx)]
                label_path_list.append(label_path)
                # load image, label, and mask

                img, success = self.load_img(img_path, task, test=True)
                label, mask = self.load_task(task, label_path, test=True)
                if not success:
                    mask = np.zeros_like(label)
                imgs.append(img)
                labels.append(label)
                masks.append(mask)    
            # form a batch
            imgs = np.stack(imgs)
            labels = np.stack(labels) if labels[0] is not None else None
            masks = np.stack(masks) if masks[0] is not None else None
            
            # preprocess and make numpy arrays to torch tensors
            imgs, labels, masks = self.preprocess_batch(task, imgs, labels, masks, [channel]) 

            return imgs, labels, masks, label_path_list, True

        if task == 'cls':
            imgs = []
            labels = []
            masks = []
            label_path_list = []

            path_idxs = np.array([], dtype=np.int64)
            path_list = self.path_dict_test_pose[str(joint_id)] 

            if class_name not in path_list:
                return  imgs, labels, masks, label_path_list, False
            if cls_gt == 1:
                path_idxs = self.choose_support_pose(path_idxs, path_list, class_name)
            else:
                filtered_keys = [key for key in path_list.keys() if key not in [class_name]]
                random_classes_query =  random.sample(filtered_keys, 1)
                path_idxs = self.choose_support_pose(path_idxs, path_list, random_classes_query[0])
        
            for path_idx in path_idxs:
                # index image path
                label_path = self.label_paths_test_pose[int(path_idx)]
                label_path_list.append(label_path)
                # load image, label, and mask

                img, success = self.load_img(label_path, task, test=True)
                label, mask = self.load_task(task, label_path, test= True)
                if not success:
                    mask = np.zeros_like(label)
                imgs.append(img)
                labels.append(label)
                masks.append(mask)    

            # form a batch
            imgs = np.stack(imgs)
            labels = np.stack(labels) if labels[0] is not None else None
            masks = np.stack(masks) if masks[0] is not None else None
            N, C, H, W = labels.shape
            cls_gts = np.full((N, 1, H, W), cls_gt)
           
            labels = cls_gts

            # preprocess and make numpy arrays to torch tensors
            imgs, labels, masks = self.preprocess_batch(task, imgs, labels, masks, [channel])

            return imgs, labels, masks, label_path_list, True
  
    def __getitem__(self, idx):
        
        if self.task == 'keypoints2d' :
            # label_path = random.choice(self.label_paths)
            label_path = self.label_paths_test_pose[idx % len(self.label_paths_test_pose)]
            
            # load image, label, and mask
            class_name = label_path.split('/')[0]
            joint_id = label_path.split('/')[1].split('_')[3]
            
            X = []
            Y = []
            M = []


            img, success = self.load_img(label_path, task=self.task, test=True)
            label, mask = self.load_task(self.task, label_path, test=True)
        
            if not success:
                mask = np.zeros_like(label)

            X_S, Y_S, M_S, label_path_list, success= self.sample_batch(self.task, self.channel_idx, class_name, joint_id)

            meta_info = [class_name, joint_id]
            if not success: 
                return self.__getitem__(idx+ 1)
      
        if self.task == 'mask':

            img_path = self.image_paths_mask[self.image_paths_test_mask[idx % len(self.image_paths_test_mask)]]
            label_path = self.image_paths_mask[self.image_paths_test_mask[idx % len(self.image_paths_test_mask)]]
 
            img, success = self.load_img(img_path, self.task, test=True)
            
            sub_class = '_'.join(img_path.split('_')[:-1])
            
            meta_info = [sub_class]
            label, mask = self.load_task(self.task, label_path, test=True)
            if not success:
                mask = np.zeros_like(label)
            
            X_S, Y_S, M_S, label_path_list, success = self.sample_batch(self.task, self.channel_idx, sub_class)

        if self.task == 'cls':
            # label_path = random.choice(self.label_paths)
            label_path = self.label_paths_test_pose[idx % len(self.label_paths_test_pose)]            
            flag = np.random.random()
            if flag < 0.5:
                cls_gt = 1
            else:
                cls_gt = 0
            # load image, label, and mask
            class_name = label_path.split('/')[0]
            joint_id = label_path.split('/')[1].split('_')[3]
            
            X = []
            Y = []
            M = []

            img, success = self.load_img(label_path, task=self.task, test=True)
            label, mask = self.load_task(self.task, label_path, test=True)
         
            if not success:
                mask = np.zeros_like(label)

            X_S, Y_S, M_S, label_path_list, success= self.sample_batch(self.task, self.channel_idx, class_name, joint_id, cls_gt=cls_gt)

            meta_info = [class_name, joint_id]
            if not success: 
                return self.__getitem__(idx+ 1)
        
        # preprocess labels
        imgs, labels, masks = self.preprocess_batch(self.task,
                                                    img[None],
                                                    None if label is None else label[None],
                                                    None if mask is None else mask[None],
                                                    channels=([self.channel_idx] if self.channel_idx >= 0 else None),
                                                    drop_background=False)
        
        if self.task == 'cls':
            N, C, H, W = labels.shape
            cls_gts = np.full((N, 1, H, W), cls_gt) 
            labels = cls_gts
        X, Y, M = imgs[0], labels[0], masks[0]
        if self.image_augmentation is not None:
            X, Y, M = self.image_augmentation(X, Y, M)
        
        t_idx = torch.tensor([TASKS.index(f'{self.task}_{c}') for c in range(len(TASKS_GROUP_DICT[self.task]))])
      
        return X, Y, M, t_idx, X_S, Y_S, M_S, meta_info
         


class UniASETCrossExperimentDataset(UniASETBaseDataset):
    def __init__(self, root_dir, s, task, channel_idx=-1, dset_size=-1, image_augmentation=False, shot = 5, **kwargs):
        super().__init__(root_dir, s, [task], **kwargs)
        super().__init__(root_dir, s, [task], **kwargs)
        
        super().__init__(root_dir, s, [task], **kwargs)    
        
        self.task = task
        self.channel_idx = channel_idx
        self.dset_size = dset_size
        self.n_channels = len(TASKS_GROUP_DICT[task])
        self.shot = shot 
        if image_augmentation:
            self.image_augmentation = RandomHorizontalFlip()
        else:
            self.image_augmentation = None
        

    def __len__(self):
        if self.dset_size > 0:
            return self.dset_size
        else:
            if self.task == 'mask':
                # print("self.testsize_mask: ",self.testsize_mask)
                return self.testsize_mask
            else:       
                # print("len(self.label_paths_test_pose): ", len(self.label_paths_test_pose))
                return len(self.label_paths_test_pose)
    
    def choose_support_pose(self, path_list, support_classes):
            path_idxs_list = []
            print("support_classes: ", len(support_classes))
            for support_class in support_classes:
                path_idxs = np.array([], dtype=np.int64)
                remaining_elements_needed = max(0, self.shot - len(path_idxs))

                # Determine the number of elements to take from path_list[random_class] in order
                elements_to_take = min(remaining_elements_needed, len(path_list[support_class]))
    
                # Take elements from path_list[random_class] in order
                selected_elements = path_list[support_class][:elements_to_take]

                # If more elements are needed, repeat the data
                while len(selected_elements) < remaining_elements_needed:
                    num_repeats = min(remaining_elements_needed - len(selected_elements), len(path_list[support_class]))
                    selected_elements = np.concatenate((selected_elements, path_list[support_class][:num_repeats]))

                # Concatenate the selected elements to path_idxs
                path_idxs = np.concatenate((path_idxs, selected_elements))
                path_idxs_list.append(path_idxs)

            return path_idxs_list
    
    def sample_batches(self, task, channel, class_name, joint_id = 0, path_idxs=None):
        # sample data paths
        # sample s for support and query

         # load images and labels
        if task == 'keypoints2d':
            path_list = self.path_dict_train_pose[str(joint_id)] 
            support_classes = [cls_name for cls_name in path_list if not cls_name == class_name ]
            
            print("query_class: ", class_name, "support_classes: ", support_classes)

            batch_list = []
            label_path_list = []       
            path_idxs_list = self.choose_support_pose(path_list, support_classes[:35])
            
            for path_idxs in path_idxs_list:
                imgs = []
                labels = []
                masks = []
                for path_idx in path_idxs:
                    # index image path
                    label_path = self.label_paths_train_pose[int(path_idx)]
                    label_path_list.append(label_path)
                    # load image, label, and mask
                    # print("Support batch label_path of kp: ",label_path)
                    img, success = self.load_img(label_path, task)
                    label, mask = self.load_task(task, label_path)
                    if not success:
                        mask = np.zeros_like(label)
                    imgs.append(img)
                    labels.append(label)
                    masks.append(mask)    
                # form a batch
                imgs = np.stack(imgs)
                labels = np.stack(labels) if labels[0] is not None else None
                masks = np.stack(masks) if masks[0] is not None else None
                # preprocess and make numpy arrays to torch tensors
                imgs, labels, masks = self.preprocess_batch(task, imgs, labels, masks, [channel])
                batch_cls = imgs, labels, masks
                batch_list.append(batch_cls)
            
            return batch_list
        
        if task == 'mask':
            imgs = []
            labels = []
            masks = []
            label_path_list = []
            path_idxs = np.array([], dtype=np.int64)
            path_list = self.path_dict_train_mask[class_name] 
        
            path_idxs = self.choose_support_mask(path_idxs, path_list)

            for path_idx in path_idxs:
                # index image path
                img_path = self.image_paths_mask[int(path_idx)]
                label_path = self.label_paths_mask[int(path_idx)]
                label_path_list.append(label_path)
                # load image, label, and mask
                # print("Support batch label_path of kp: ",img_path)
                img, success = self.load_img(img_path, task)
                label, mask = self.load_task(task, label_path)
                if not success:
                    mask = np.zeros_like(label)
                imgs.append(img)
                labels.append(label)
                masks.append(mask)    
            # form a batch
            imgs = np.stack(imgs)
            labels = np.stack(labels) if labels[0] is not None else None
            masks = np.stack(masks) if masks[0] is not None else None
            
            # preprocess and make numpy arrays to torch tensors
            imgs, labels, masks = self.preprocess_batch(task, imgs, labels, masks, [channel]) 

            return imgs, labels, masks, label_path_list, True

        if task == 'cls':
            imgs = []
            labels = []
            masks = []
            label_path_list = []

            path_idxs = np.array([], dtype=np.int64)
            path_list = self.path_dict_train_pose[str(joint_id)] 
            flag = np.random.random()
            if class_name not in path_list:
                return  imgs, labels, masks, label_path_list, False
            if flag < 0.5:
                random_classes_query = class_name
                cls_gt = 1
                path_idxs = self.choose_support_pose(path_idxs, path_list, random_classes_query)
            else:
                random_classes_query =  random.sample(list(path_list.keys()), 1)
                cls_gt = 0
                path_idxs = self.choose_support_pose(path_idxs, path_list, random_classes_query[0])
        
            for path_idx in path_idxs:
                # index image path
                label_path = self.label_paths_train_pose[int(path_idx)]
                label_path_list.append(label_path)
                # load image, label, and mask
                # print("Support batch label_path of kp: ",label_path)
                img, success = self.load_img(label_path, task)
                label, mask = self.load_task(task, label_path)
                if not success:
                    mask = np.zeros_like(label)
                imgs.append(img)
                labels.append(label)
                masks.append(mask)    
            # form a batch
            imgs = np.stack(imgs)
            labels = np.stack(labels) if labels[0] is not None else None
            masks = np.stack(masks) if masks[0] is not None else None
            
            # preprocess and make numpy arrays to torch tensors
            imgs, labels, masks = self.preprocess_batch(task, imgs, labels, masks, [channel])

            return imgs, labels, masks, label_path_list, True


    def __getitem__(self, idx):
        
        if self.task == 'keypoints2d' :
            # label_path = random.choice(self.label_paths)
            label_path = self.label_paths_test_pose[idx % len(self.label_paths_test_pose)]
            
            # load image, label, and mask
            class_name = label_path.split('/')[0]
            joint_id = label_path.split('/')[1].split('_')[3]
            
            X = []
            Y = []
            M = []

            # print("Prompt label_path of kp : ", label_path)
            img, success = self.load_img(label_path, task=self.task, test=True)
            label, mask = self.load_task(self.task, label_path, test=True)
        
            if not success:
                mask = np.zeros_like(label)

            support_batches = self.sample_batches(self.task, self.channel_idx, class_name, joint_id)
            meta_info = [class_name, joint_id]
            if not success: 
                return self.__getitem__(idx+ 1)
      

        if self.task == 'mask':

            img_path = self.image_paths_mask[self.image_paths_test_mask[idx % len(self.image_paths_test_mask)]]
            label_path = self.image_paths_mask[self.image_paths_test_mask[idx % len(self.image_paths_test_mask)]]
            # print("Prompt label_path of mask : ", img_path)
            img, success = self.load_img(img_path, self.task, test=True)
            
            sub_class = '_'.join(img_path.split('_')[:-1])
            
            meta_info = [sub_class]
            label, mask = self.load_task(self.task, label_path, test=True)
            if not success:
                mask = np.zeros_like(label)
            
            X_S, Y_S, M_S, label_path_list, success = self.sample_batch(self.task, self.channel_idx, sub_class)

        if self.task == 'cls':
            # label_path = random.choice(self.label_paths)
                label_path = self.label_paths_test_pose[idx % len(self.label_paths_test_pose)]
                
                # load image, label, and mask
                class_name = label_path.split('/')[0]
                joint_id = label_path.split('/')[1].split('_')[3]
                
                X = []
                Y = []
                M = []

                # print("Prompt label_path of kp : ", label_path)
                img, success = self.load_img(label_path, task=self.task, test=True)
                label, mask = self.load_task(self.task, label_path, test=True)
            
                if not success:
                    mask = np.zeros_like(label)

                X_S, Y_S, M_S, label_path_list, success= self.sample_batch(self.task, self.channel_idx, class_name, joint_id)
                meta_info = [class_name, joint_id]
                if not success: 
                    return self.__getitem__(idx+ 1)
        
        # preprocess labels
        imgs, labels, masks = self.preprocess_batch(self.task,
                                                    img[None],
                                                    None if label is None else label[None],
                                                    None if mask is None else mask[None],
                                                    channels=([self.channel_idx] if self.channel_idx >= 0 else None),
                                                    drop_background=False)  
        X, Y, M = imgs[0], labels[0], masks[0]
        if self.image_augmentation is not None:
            X, Y, M = self.image_augmentation(X, Y, M)
        
        t_idx = torch.tensor([TASKS.index(f'{self.task}_{c}') for c in range(len(TASKS_GROUP_DICT[self.task]))])
        # t_idx = TASKS.index(f'{self.task}_{channels[i]}')
        return X, Y, M, t_idx, support_batches, meta_info
         
