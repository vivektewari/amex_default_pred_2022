import pandas as pd
import torch
import torchmetrics
from output_diagnostics.metrics import amex_metric
from utils.visualizer import Visualizer

from catalyst.dl  import  Callback, CallbackOrder,Runner

class MetricsCallback(Callback):

    def __init__(self,
                 directory=None,
                 model_name='',
                 check_interval=1,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 prefix: str = "amex_metric",
                 visdom_env:str='default'

                 ):
        super().__init__(CallbackOrder.Metric)

        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.directory = directory
        self.model_name = model_name
        self.check_interval = check_interval

        self.visualizer = Visualizer(env=visdom_env)
        self.my_actual=[]
        self.my_preds = []

    def on_batch_end(self, state: Runner):
        if state.is_valid_loader:
            self.my_preds.extend(state.batch['logits'].detach())
            self.my_actual.extend(state.batch['targets'].detach())



    def on_epoch_end(self, state: Runner):
        """Event handler for epoch end.

        Args:
            runner ("IRunner"): IRunner instance.
        """
        #print(torch.sum(state.model.fc1.weight),state.model.fc1.weight[5][300])
        #print(torch.sum(state.model.conv_blocks[0].conv1.weight))
        if self.directory is not None: torch.save(state.model.state_dict(), str(self.directory) + '/' +
                                                  self.model_name + "_" + str(
            state.epoch_step) + ".pth")

        if (state.epoch_step + 1) % self.check_interval == 0:
            preds = state.batch['logits']
            #preds[state.batch['targets']]
            metric=amex_metric(torch.tensor(self.my_actual).flatten(), torch.tensor(self.my_preds).flatten())
            self.my_actual=[]
            self.my_preds=[]
            print("{} is {}".format(self.prefix,metric ))
            reg_loss=0
            for param in state.model.parameters(): reg_loss += torch.sum(param.detach()** 2)
            self.visualizer.display_current_results(state.epoch_step, reg_loss,
                                                    name='regularization_loss')
            self.visualizer.display_current_results(state.epoch_step, state.epoch_metrics['train']['loss'],
                                                    name='train_loss')
            self.visualizer.display_current_results(state.epoch_step, state.epoch_metrics['valid']['loss'],
                                                    name='valid_loss')
            self.visualizer.display_current_results(state.epoch_step, metric[0],
                                                    name='amex_metric_on_whole_v')
            self.visualizer.display_current_results(state.epoch_step, metric[1],
                                                    name='auc_on_whole_v')
            self.visualizer.display_current_results(state.epoch_step, metric[2],
                                                    name='bad_capture_on_whole_v')
            sum1=torch.where(state.batch['targets'] == 1, preds, torch.tensor(0.0, dtype=torch.float)).sum()
            sum0 = torch.where(state.batch['targets'] == 0, preds, torch.tensor(0.0, dtype=torch.float)).sum()
            q10=torch.where(state.batch['targets'] == 1, preds, torch.tensor(0.0, dtype=torch.float))

            count1=state.batch['targets'].sum()
            count0=len(state.batch['targets'])-count1
            self.visualizer.display_current_results(state.epoch_step, sum1/count1,
                                                    name='1mean')
            self.visualizer.display_current_results(state.epoch_step,torch.quantile(q10[q10>0],0.9),
                                                    name='1_0.90_quantile')
            self.visualizer.display_current_results(state.epoch_step, sum0/count0,
                                                    name='0mean')
            acc=torchmetrics.Accuracy()
            self.visualizer.display_current_results(state.epoch_step, acc(preds,state.batch['targets'].long()),
                                                    name='accuracy')




class IteratorCallback(Callback):
    def __init__(self,
                 info_dict={},
                 last_epoch=50,
                 metric_loc="",
                 ):
        super().__init__(CallbackOrder.Metric)
        self.info_dict=info_dict
        self.capturing_metrics_list = {'valid_loss': [], 'amex_metric': [],
                                       'bad_capture': [], 'auc': []}
        self.metric_loc = metric_loc
        self.last_epoch = last_epoch
    def set_info_dict(self,info_dict):
        self.info_dict=info_dict
    def set_blank_csv(self):

        pd.DataFrame(columns=list(self.info_dict.keys()) + list(self.capturing_metrics_list.keys())).to_csv(self.metric_loc,
                                                                                                            index=False)

    def on_epoch_end(self, state: Runner):
        """Event handler for epoch end.

        Args:
            runner ("IRunner"): IRunner instance.
        """
        if state.epoch_step ==1 :
            self.capturing_metrics_list = {'valid_loss': [], 'amex_metric': [],
                                      'bad_capture': [], 'auc': []}
        preds = state.batch['logits']
        metric = amex_metric(state.batch['targets'], preds)
        capturing_metrics={'valid_loss':-state.epoch_metrics['valid']['loss'],'amex_metric':metric[0],
                           'auc':metric[1],'bad_capture':metric[2]}

        for key in capturing_metrics.keys():# trying to keep best 4
            self.capturing_metrics_list[key].append(capturing_metrics[key])
            if state.epoch_step<=4:continue
            else:
                self.capturing_metrics_list[key].sort()
                self.capturing_metrics_list[key].pop(0)
        if state.epoch_step ==self.last_epoch :

            for key in capturing_metrics.keys():  # trying to keep best 4
                if key == 'valid_loss': multiplier = -1
                else:multiplier=1
                self.capturing_metrics_list[key]=[multiplier*sum(self.capturing_metrics_list[key])/4]

            self.info_dict.update(self.capturing_metrics_list)
            pd.DataFrame(self.info_dict).to_csv(self.metric_loc,mode='a',header=False,index=False)










