import torch.nn as nn
import numpy as np
import torch
EPSILON_FP16 = 1e-5
from utils.funcs import vison_utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class BCELoss(nn.Module):
    def __init__(self):

        super().__init__()
        self.func = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, actual):
        # bs, s, o = pred.shape
        # pred = pred.reshape(bs*s, o)
        #pred = self.sigmoid(pred)
        pred = torch.clamp(pred, min=EPSILON_FP16, max=1.0-EPSILON_FP16)

        return self.func(pred, actual)

class LocalizatioLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bounding_loss = self.bounding_box_l2
        self.classification_loss =self.l2_loss# BCELoss()#
        self.bouding_conv_func= lambda x :x
    def l2_loss(self,pred, actual):
        #actual=torch.where(actual==1,1 ,0)
        loss =(torch.pow(torch.mean(torch.pow(1-pred[[i for i in range(len(pred))],[actual.tolist()]],2),dim=1),1/2))
        return loss[0]

    def bounding_box_l2(self,pred, actual):
        """
        :param pred:list of tuple in n dimension
        :param actual: list of tuple in n dimension
        :return: sum of l2 distance for each point
        """
        num_points = len(pred)
        loss = 0#torch.zeros(pred[0].shape)
        for i in range(num_points):
            loss +=torch.sum(torch.pow(torch.sum(torch.pow(pred[i]-actual[i],2),dim=1),1/2))

        return loss/(pred[0].shape[0]*2)

    def loss_calc(self, pred_classif, actual_classif,pred_bounding,actual_bounding):
        loss=self.bouding_conv_func(self.bounding_loss(pred_bounding,actual_bounding))+10*self.classification_loss(pred_classif, actual_classif)
        return loss
    def convert(self,x):
        x_len=x.shape[1]-4

        return x[:,0:x_len],[x[:,x_len:x_len+2],x[:,x_len+2:x_len+4]]
    def forward(self,pred,actual):
        pred_classif, pred_bounding=self.convert(pred)
        actual_classif, actual_bounding=self.convert(actual)
        return self.loss_calc( pred_classif, torch.tensor(actual_classif,dtype=torch.long)[:,0],pred_bounding,actual_bounding)
    def get_individual_loss(self,actual,pred):
        pred_classif, pred_bounding = self.convert(pred)
        actual_classif, actual_bounding = self.convert(actual)
        boundig_loss = self.bouding_conv_func(self.bounding_loss(pred_bounding,actual_bounding))
        classification_loss=self.classification_loss(pred_classif, torch.tensor(actual_classif,dtype=torch.long)[:,0])
        return boundig_loss , classification_loss



class MultiBoxLoss(LocalizatioLoss):
    """
    The MultiBox loss, a loss function for object detection.
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy=None, threshold=0.6, neg_pos_ratio=3, alpha=1.,pixel_shape=(28,28),n_class=None):
        super(MultiBoxLoss, self).__init__()
        if priors_cxcy is None:
            self.priors_cxcy = self.create_prior_boxes()
        else:
            self.priors_cxcy = priors_cxcy
        self.pixel_shape=pixel_shape
        self.n_class=n_class
        self.priors_xy = torch.clamp(vison_utils.cxcy_to_xy(self.priors_cxcy),min=0,max=1)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)


    def convert(self,x):
        if x.shape[1]!=5:
            preds=x[:,:self.priors_cxcy.shape[0]*(self.n_class+1)].reshape((x.shape[0],self.priors_cxcy.shape[0],self.n_class+1))
            locs = x[:, self.priors_cxcy.shape[0] * (self.n_class + 1):].reshape((x.shape[0],self.priors_cxcy.shape[0],4))
            x=torch.cat([preds,locs],dim=2)
            #x=x.reshape(x.shape[0],self.priors_cxcy.shape[0],self.n_class+1+4)#[:,0:x_len],torch.cat([x[:,x_len:x_len+2],x[:,x_len+2:x_len+4]])
            x_len = x.shape[2] - 4
        else :
            x = x.reshape(x.shape[0], 1, 5)
            x_len=1
        return x[:,:,:x_len],x[:,:,x_len:]

    @staticmethod
    def create_prior_boxes():
        """
        Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.
        :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
        """
        fmap_dims = {'conv4_3': 3,
                     'conv7': 9,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

        obj_scales = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375,
                      'conv9_2': 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}
        #fmap_dims = {'conv4_3': 10}
        #obj_scales = {'conv4_3': [0.324693658031057,0.13982694076936,0.254101116776089,0.121527277611988,0.379986683048495,0.162851193909395,0.171834183680792]}
        #aspect_ratios = {'conv4_3': [1.54719497498501, 1.49927167625187, 6.85407259539126,3.90017628205128,1.08222360161625,1.01875795512884,2.71517599378192]}
        fmap_dims = {'conv4_3': 10}
        obj_scales = {'conv4_3': [0.38, 0.14, 0.28, 0.11, 0.33,0.08,0.16,0.12,0.10,0.23,0.36]}
        aspect_ratios = {'conv4_3': [0.99, 1.33, 1.96, 2.13, 1.45,4.00,1.004,1.71,2.80,2.95,1.21]}
        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]
                    for loop in range(len(aspect_ratios[fmap])):
                        ratio=aspect_ratios[fmap][loop]
                        scale=obj_scales[fmap][loop]
                        prior_boxes.append(
                            [cx, cy, scale * np.sqrt(ratio), scale/ np.sqrt(ratio)])

                        # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                        # scale of the current feature map and the scale of the next feature map
                        # if ratio == 1.:
                        #     try:
                        #         additional_scale = np.sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                        #     # For the last feature map, there is no "next" feature map
                        #     except IndexError:
                        #         additional_scale = 1.
                        #     prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
        prior_boxes.clamp_(0, 1)  # (8732, 4)

        return prior_boxes
    def quick_fix(self):pass

    def forward(self, pred,actual,split=False):
        """
        Forward propagation.
        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        predicted_scores, predicted_locs = self.convert(pred)
        labels, boxes = self.convert(actual)
        boxes=boxes/self.pixel_shape[0]
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        true_classes = torch.ones((batch_size, n_priors), dtype=torch.long).to(device)*10# (N, 8732)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = vison_utils.find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, 8732)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][0][object_for_each_prior]  # (8732)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 10  # (8732)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = vison_utils.cxcy_to_gcxgcy(vison_utils.xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 10  # (N, 8732)

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive


        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS
        if split : return [loc_loss,conf_loss ]
        return conf_loss + self.alpha * loc_loss
    def get_individual_loss(self,actual,pred):
        loss=self.forward(pred,actual,split=True)

        return loss[0] , loss[1]
    def visualize_image(self,pred,actual):

        predicted_scores, predicted_locs = self.convert(pred)


        # targets = torch.cat([targets[:, 0].reshape((targets.shape[0], 1)), torch.ones((targets.shape[0], 1)),
        #                      targets[:, 1:].reshape((targets.shape[0], 4))], dim=1)
        # targets = targets.reshape((targets.shape[0], 1, 6))
        labels, boxes = self.convert(actual)
        boxes = boxes/self.pixel_shape[0]
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 8732)

        # For each image
        pred_locs=[]
        prioris=[]
        overlaps=[]
        predicted_s=[]

        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = vison_utils.find_jaccard_overlap(boxes[i],
                                                       self.priors_xy)  # (n_objects, 8732)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.
            pred_sc = torch.tensor([
                int(actual[i, 0]), predicted_scores[i, prior_for_each_object[0], int(actual[i, 0])]])
            prioris.append([torch.cat([pred_sc,self.priors_xy[prior_for_each_object[0]]],dim=0)])

            pred_=vison_utils.cxcy_to_xy(vison_utils.gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))
            pred_ = torch.clamp(pred_, 0, 1)
            pred_locs.append(pred_[prior_for_each_object[0]].tolist())
            overlaps.append(overlap[:,prior_for_each_object[0]])

            prioris,overlaps=list(map(list,prioris)),list(map(list,overlaps))

            predicted_s.append(predicted_scores[i,prior_for_each_object[0],int(actual[i,0])].detach())
        return prioris,torch.tensor(overlaps),torch.tensor( predicted_s),torch.floor(torch.tensor(pred_locs)*28*4)


