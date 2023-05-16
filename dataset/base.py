import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import random
class BaseNumpyDataset(Dataset):
    """Custom dataset class for classification"""

    def __init__(
        self,
        data_dict: dict,
        image_key: str = "images",
        label_key: str = "labels",
        transforms=None,
        is_ul_unknown=False,
        num_classes=10,
        soft_domain=False,
        labeled_data_num=0,
        domain_labels=None,
        dual_sample=False,
        dual_sample_type=None,
    ):
        
        self.dataset = data_dict
        self.image_key = image_key
        self.label_key = label_key
        self.transforms = transforms
        self.is_ul_unknown = is_ul_unknown
        self.ood_num=0
        self.num_classes=num_classes
        self.soft_domain=soft_domain 
        self.total_num=len(self.dataset['images'])
        if not is_ul_unknown:
            self.num_per_cls_list = self._load_num_samples_per_class()
        else:
            self.num_per_cls_list = None
        if self.soft_domain:
            self.repeat_num=10
            self.labeled_data_num=labeled_data_num
            self.soft_labels = domain_labels
            self.prediction =domain_labels.repeat(self.repeat_num,axis=0)
            self.prediction=self.prediction.reshape(len(self.prediction)//self.repeat_num,self.repeat_num)
            self.count = 0
        self.dual_sample=dual_sample
        if self.dual_sample:
            self.dual_sample_type=dual_sample_type 
            self.class_weight, self.sum_weight = self.get_weight()
            self.class_dict = self._get_class_dict()
            
            

    def __getitem__(self, idx):
        img = self.dataset[self.image_key][idx]
        label = -1 if self.is_ul_unknown else self.dataset[self.label_key][idx]
        if self.transforms is not None: 
            img = self.transforms(img)                 
        if self.dual_sample:
            if self.dual_sample_type == "ClassReversedSampler":
                sample_class = self.sample_class_index_by_weight()
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.dual_sample_type== "ClassBalancedSampler":
                sample_class = random.randint(0, self.cls_num-1)
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.dual_sample_type== "uniform":
                sample_index = random.randint(0, self.__len__() - 1)

            sample_img, sample_label = self.dataset[self.image_key][sample_index], self.dataset[self.label_key][sample_index]
            if self.transforms is not None: 
                sample_img = self.transforms(sample_img)
            meta=dict()
            meta['dual_image'] = sample_img
            meta['dual_label'] = sample_label 
            return img, label, meta
            
        if self.soft_domain: 
            return img,label,self.soft_labels[idx],idx           
        return img, label, idx

    def __len__(self):
        return len(self.dataset[self.image_key])

    # label-to-class quantity
    def _load_num_samples_per_class(self):
        labels = self.dataset[self.label_key]
        classes = range(-1,self.num_classes)
        classwise_num_samples = [0]*(len(classes)-1)
        for i in classes:
            if i==-1:
                self.ood_num=len(np.where(labels == i)[0])
                continue
            classwise_num_samples[i] = len(np.where(labels == i)[0])
 
        return np.array(classwise_num_samples)
    
    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.num_classes):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i
    
    def soft_label_update(self,results):
        self.count += 1

        # While updating the noisy label y_i by the probability s, we used the average output probability of the network of the past 10 epochs as s.
        idx = (self.count - 1) % 10
        self.prediction[self.labeled_data_num:, idx] = results[self.labeled_data_num:]

        if self.count >= 10:
            self.soft_labels = self.prediction.mean(axis=1)
        return 
    
    def _get_class_dict(self):
        class_dict = dict()
        for i, cat_id in enumerate(self.dataset[self.label_key]): 
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_weight(self):
        num_list = [0] * self.num_classes
        cat_list = []
        for category_id in self.dataset[self.label_key]:
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight


    def select_dataset(self, indices=None, labels=None, return_transforms=False):
        if indices is None:
            indices = np.array(list([i for i in range(len(self.dataset[self.image_key]))]))
        imgs = self.dataset[self.image_key][indices]

        if not self.is_ul_unknown:
            _labels = self.dataset[self.label_key][indices]
        else:
            _labels = np.array([-1 for _ in range(len(indices))])

        if labels is not None:
            # override specified labels (i.e., pseudo-labels)
            _labels = np.array(labels)

        assert len(_labels) == len(imgs)
        dataset = {self.image_key: imgs, self.label_key: _labels}

        if return_transforms:
            return dataset, self.transforms    
        return dataset
