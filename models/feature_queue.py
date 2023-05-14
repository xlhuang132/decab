from collections import defaultdict
import torch 
import torch.distributed as dist
import diffdist.functional as distops 


class FeatureQueue:

    def __init__(self, cfg, classwise_max_size=None, bal_queue=False ):
        self.num_classes = cfg.DATASET.NUM_CLASSES
        self.feat_dim = cfg.MODEL.QUEUE.FEAT_DIM
        self.max_size = cfg.MODEL.QUEUE.MAX_SIZE
 
        self.bank = defaultdict(lambda: torch.empty(0, self.feat_dim).cuda())
        self.prototypes = torch.zeros(self.num_classes, self.feat_dim).cuda()
        
        self.classwise_max_size = classwise_max_size
        self.bal_queue = bal_queue
        self.only_momentum=False
        
    def enqueue(self, features, labels):
        for idx in range(self.num_classes):
            # per class max size
            max_size = (
                self.classwise_max_size[idx] * 5  # 5x samples
            ) if self.classwise_max_size is not None else self.max_size
            if self.bal_queue:
                max_size = self.max_size
            # select features by label
            cls_inds = torch.where(labels == idx)[0]
            if len(cls_inds):
                with torch.no_grad():
                    # push to the memory bank
                    feats_selected = features[cls_inds]
                    self.bank[idx] = torch.cat([self.bank[idx], feats_selected], 0)

                    # fixed size
                    current_size = len(self.bank[idx])
                    if current_size > max_size:
                        self.bank[idx] = self.bank[idx][current_size - max_size:]

                    # update prototypes
                    self.prototypes[idx, :] = self.bank[idx].mean(0)
