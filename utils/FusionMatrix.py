
from tokenize import group 
import numpy as np 
from matplotlib import pyplot as plt

class FusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes 
        self.reset() 

    def reset(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)

    def update(self, output, label):
        length = output.shape[0]
        for i in range(length):  
            self.matrix[int(output[i]), int(label[i])] += 1

    def get_rec_per_class(self):
        rec = np.array(
            [
                self.matrix[i, i] / self.matrix[:, i].sum()
                for i in range(self.num_classes)
            ]
        )
        rec[np.isnan(rec)] = 0
        return rec

    def get_pre_per_class(self):
        pre = np.array(
            [
                self.matrix[i, i] / self.matrix[i, :].sum()
                for i in range(self.num_classes)
            ]
        )
        pre[np.isnan(pre)] = 0
        return pre
    
    def get_acc_per_class(self): 
        acc = np.array(
            [self.matrix[i, i]/self.matrix[:,i].sum() for i in range(self.num_classes)]
        )
        acc[np.isnan(acc)] = 0 
        acc=np.around(acc,4) 
        return acc
     
    def get_TNR(self): 
        tnr=self.matrix[0,0]/(self.matrix[0,0]+self.matrix[1,0]) 
        return tnr if not np.isnan(tnr) else 0
    
    def get_TPR(self): 
        tpr=self.matrix[1,1]/(self.matrix[1,1]+self.matrix[0,1])        
        return tpr if not np.isnan(tpr) else 0

    def get_accuracy(self):
        acc = (
            np.sum([self.matrix[i, i] for i in range(self.num_classes)])
            / self.matrix.sum()
        ) 
        return acc if not np.isnan(acc) else 0

    def plot_per_acc(self,save_path):
        acc=self.get_acc_per_class()
        self.plot_acc_bar(acc, [i for i in range(self.num_classes)], save_path) 
        return acc
    
    def get_group_splits(self,splits=[3,3,4]):
        group_splits=[]
        if splits!=[]:
            assert sum(splits)==self.num_classes
            num_ids=[i for i in range(self.num_classes)]
            c=0
            for i in splits:
                group_splits.append(num_ids[c:c+i])
                c+=i
        return group_splits
    
    def plot_acc_bar(self,acc,tick_label,save_path):
        assert len(acc)==len(tick_label)
        fig, ax = plt.subplots(figsize=(10,10))
        bars1 =plt.bar(range(len(acc)), acc,tick_label=tick_label)
        for b in bars1:  
            height = b.get_height()
            ax.annotate('{}'.format(height), 
                    xy=(b.get_x() + b.get_width() / 2, height), 
                    xytext=(0,3),  
                    textcoords="offset points",  
                    va = 'bottom', ha = 'center',  
                    weight='heavy',
                    ) 
        plt.show()
        plt.savefig(save_path)
        plt.close()
            
    def get_group_acc(self,split=[3,3,4]):
        acc=self.get_acc_per_class()
        group_splits=self.get_group_splits(split)
        group_acc=[np.mean(acc[idxs]) for idxs in group_splits]
        return group_acc
    
    def plot_group_acc(self,save_path,split=[3,3,4]):
        group_acc=self.get_group_acc(split=split)
        self.plot_acc_bar(group_acc, ['Many','Medium','Few'], save_path)
        return group_acc
        
    def plot_confusion_matrix(self, normalize = False, cmap=plt.cm.Blues):

        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = self.matrix.T

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=np.arange(self.num_classes), yticklabels=np.arange(self.num_classes),
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return fig
 
 
        