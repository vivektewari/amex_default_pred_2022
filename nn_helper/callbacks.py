import pandas as pd
import torch
from output_diagnostics.metrics import amex_metric
from utils.visualizer import Visualizer
import os

from catalyst.dl  import  Callback, CallbackOrder,Runner

class MetricsCallback(Callback):

    def __init__(self,
                 directory=None,
                 model_name='',
                 check_interval=1,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 prefix: str = "amex_metric",

                 ):
        super().__init__(CallbackOrder.Metric)

        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.directory = directory
        self.model_name = model_name
        self.check_interval = check_interval

        self.visualizer = Visualizer()

    # def on_batch_end(self,state: State):# #
    #     targ = state.input[self.input_key].detach().cpu().numpy()
    #     out = state.output[self.output_key]
    #
    #     clipwise_output = out[self.model_output_key].detach().cpu().numpy()
    #
    #     self.prediction.append(clipwise_output)
    #     self.target.append(targ)
    #
    #     y_pred = clipwise_output.argmax(axis=1)
    #     y_true = targ.argmax(axis=1)

    # score = f1_score(y_true, y_pred, average="macro")
    # state.batch_metrics[self.prefix] = score

    def on_epoch_end(self, state: Runner):
        """Event handler for epoch end.

        Args:
            runner ("IRunner"): IRunner instance.
        """
        #print(torch.sum(state.model.fc1.weight),state.model.fc1.weight[5][300])
        #print(torch.sum(state.model.conv_blocks[0].conv1.weight))
        if self.directory is not None: torch.save(state.model.state_dict(), str(self.directory) + '/' +
                                                  self.model_name + "_" + str(
            state.stage_epoch_step) + ".pth")

        if (state.stage_epoch_step + 1) % self.check_interval == 0:
            preds = state.batch['logits']
            metric=amex_metric(state.batch['targets'], preds)
            print("{} is {}".format(self.prefix,metric ))
            self.visualizer.display_current_results(state.stage_epoch_step, state.epoch_metrics['train']['loss'],
                                                    name='train_loss')
            self.visualizer.display_current_results(state.stage_epoch_step, state.epoch_metrics['valid']['loss'],
                                                    name='valid_loss')
            self.visualizer.display_current_results(state.stage_epoch_step, metric[0],
                                                    name='amex_metric')
            self.visualizer.display_current_results(state.stage_epoch_step, metric[1],
                                                    name='bad_capture')
            self.visualizer.display_current_results(state.stage_epoch_step, metric[2],
                                                    name='auc')
class MetricsCallback_loc(Callback):

    def __init__(self,
                 directory=None,
                 model_name='',
                 check_interval=1,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 prefix: str = "bound_loss,classification_loss,acc_pre_rec_f1",
                 func = amex_metric,
                 pixel =None):
        super().__init__(CallbackOrder.Metric)

        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.directory = directory
        self.model_name = model_name
        self.check_interval = check_interval
        self.func = func
        self.vision_utils = vison_utils
        self.visualizer = Visualizer()
        self.pixel = pixel

    # def on_batch_end(self,state: State):# #
    #     targ = state.input[self.input_key].detach().cpu().numpy()
    #     out = state.output[self.output_key]
    #
    #     clipwise_output = out[self.model_output_key].detach().cpu().numpy()
    #
    #     self.prediction.append(clipwise_output)
    #     self.target.append(targ)
    #
    #     y_pred = clipwise_output.argmax(axis=1)
    #     y_true = targ.argmax(axis=1)

    # score = f1_score(y_true, y_pred, average="macro")
    # state.batch_metrics[self.prefix] = score

    def on_epoch_end(self, state: Runner):
        """Event handler for epoch end.

        Args:
            runner ("IRunner"): IRunner instance.
        """
        #print(torch.sum(state.model.fc1.weight),state.model.fc1.weight[5][300])
        #print(torch.sum(state.model.conv_blocks[0].conv1.weight))
        if self.directory is not None: torch.save(state.model.state_dict(), str(self.directory) + '/' +
                                                  self.model_name + "_" + str(
            state.stage_epoch_step) + ".pth")

        if (state.stage_epoch_step + 1) % self.check_interval == 0:

            preds = state.batch['logits']
            box_count = int(preds.shape[1] / 15)
            #pred_class= torch.argmax(state.batch['logits'][:,:10], dim=1)
            temp = preds.reshape((preds.shape[0],box_count,15))[:,:,:11]
            pred_class=torch.argmax(temp.reshape((preds.shape[0],box_count*11)), dim=1)%11
            #accuracy_metrics=getMetrics(state.batch['targets'][:, 0], pred_class)
            loss=self.func(state.batch['targets'], preds)
            print("{} is {}{}".format(self.prefix, loss,accuracy_metrics))
            self.visualizer.display_current_results(state.stage_epoch_step, state.epoch_metrics['train']['loss'],
                                                    name='train_loss')
            self.visualizer.display_current_results(state.stage_epoch_step, state.epoch_metrics['valid']['loss'],
                                                    name='valid_loss')
            # self.visualizer.display_current_results(state.stage_epoch_step, accuracy_metrics[0],
            #                                         name='accuracy')
            self.visualizer.display_current_results(state.stage_epoch_step, loss[0],
                                                    name='bounding_loss')
            self.visualizer.display_current_results(state.stage_epoch_step, loss[1],
                                                    name='classification_loss')

    def on_batch_end(self,state):
        # if state.global_batch_step == 1:
        #     self.rub_pred()
        #torch.nn.utils.clip_grad_value_(state.model.parameters(), clip_value=1.0)
        target = state.batch['targets'].clone().detach()
        target[:, 1:] = target[:, 1:] / self.pixel
        if state.loader_batch_step == 1:
            if state.is_train_loader:
                model_output=state.get_model(1).model_outputs(state.batch['logits'])

                prec_rec = self.get_prec_recall(target,model_output)
                self.visualizer.display_current_results(state.stage_epoch_step, prec_rec[0],
                                                        name='precision')
                self.visualizer.display_current_results(state.stage_epoch_step, prec_rec[1],
                                                        name='recall')



        if state.loader_batch_step==1 and (state.global_epoch_step)%1 ==0 and state.is_train_loader:
            preds = state.batch['logits']
            pred_class = torch.argmax(state.batch['logits'][:, :10], dim=1)
            #max_prob=torch.max(state.batch['logits'][:, :10], dim=1)[0]
            prioris,overlaps,prob,pred_locs=state.criterion.visualize_image(preds,state.batch['targets'])
            pred_locs=self.pred_boxes(model_output,[i for i in range(11)])
            #self.rub_pred()
            draw_list=[[prioris[i],pred_locs[i]] for i in range(len(prioris))]
            self.draw_image(draw_list, color_intensity=[(0,0,200),(200,200,0)],msg = (overlaps,prob))
            #self.draw_image(prioris, msg=(overlaps, prob))
            self.get_grads(state)

            #print("max_gradient is "+ torch.max(state.model.state_dict().values()[0]))

    def get_prec_recall(self,targets,model_pred_output):
        """
        Algo:
        1.conver targets in batch,boxes,6
        2.conver preds in batch,boxes,6
        :param targets:
        :return:
        """
        batch_size=targets.shape[0]
        #1
        targets=torch.cat([targets[:,0].reshape((targets.shape[0],1)),torch.ones((targets.shape[0],1)),targets[:,1:].reshape((targets.shape[0],4))],dim=1)
        targets=targets.reshape((targets.shape[0],1,6))
        #2
        n_class = 11
        classes=[i for i in range(n_class)]

        boxes=int(model_pred_output.shape[1]/(n_class+4))

        pred_boxes=self.pred_boxes(model_pred_output,classes)
        for i in range(batch_size):
            true_boxes_i = targets[i]
            pred_boxes_i=pred_boxes[i]#self.vision_utils.non_max_suppression( pred_boxes[i], classes, background_class=11, pred_thresh=0.9, overlap_thresh=0.2)
            if i==0:
                tp_fp_np=self.vision_utils.tp_fp_fn(pred_boxes_i, true_boxes_i, classes[:-1], iou_threshold=0)
            else:
                tp_fp_np += self.vision_utils.tp_fp_fn(pred_boxes_i, true_boxes_i, classes[:-1], iou_threshold=0)
        tp_fp_np_all=torch.cat([torch.sum(tp_fp_np,dim=0).reshape((1,3)),tp_fp_np],dim=0)
        precision=tp_fp_np_all[:,0]/(tp_fp_np_all[:,0]+tp_fp_np_all[:,1])
        recall=tp_fp_np_all[:,0]/(tp_fp_np_all[:,0]+tp_fp_np_all[:,2])
        precision=torch.nan_to_num(precision,nan=0)
        recall= torch.nan_to_num(recall, nan=0)

        max_prec,max_rec,min_prec,min_rec=max(precision[1:]),max(recall[1:]),min(precision[1:]),min(recall[1:])
        return  [precision[0],recall[0],max_prec,max_rec,min_prec,min_rec]








