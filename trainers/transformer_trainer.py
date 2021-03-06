
from catalyst.dl import SupervisedRunner
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.funcs import count_parameters

def train(model,data_loader,data_loader_v,loss_func,callbacks=None,pretrained=None,lr=0.001,epoch=100):


    criterion = loss_func()

    count_parameters(model)

    if pretrained is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(pretrained, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
    optimizer = optim.SGD(model.parameters(), lr=lr)




    #print("train: {}, val: {}".format( len(data_loader), len(data_loader_v)))
    loaders = {
        "train": DataLoader(data_loader,batch_size=128,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False),
        "valid":DataLoader(data_loader_v,batch_size=4096,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False)}

    runner = SupervisedRunner(

        output_key="logits",
        input_key="image_pixels",
        target_key="targets")
    # scheduler=scheduler,

    runner.train(
        model=model,
        criterion=criterion,
        loaders=loaders,
        optimizer=optimizer,

        num_epochs=epoch,
        verbose=True,
        logdir=f"fold0",
        callbacks=callbacks,
    )

    # main_metric = "epoch_f1",
    # minimize_metric = False

if __name__ == "__main__":
    from nn_helper.callbacks import *
    from input_diagnostics.common import *
    import nn_helper.dataLoaders as dl
    import nn_helper.losses as loss
    import models.models as mdl
    #load dataloader
    data_loader_ = dl.__dict__[config.data_loader]
    pickle_jar = [config.data_loc + 'data_created/' + 'dict1.pkl', config.data_loc + 'data_created/' + 'dict2.pkl']
    data_loader=data_loader_(group=pickle_jar, n_skill=4, max_seq=13, dev=True)
    pickle_jar = [config.data_loc + 'data_created/' + 'v_dict1.pkl', config.data_loc + 'data_created/' + 'v_dict2.pkl']
    data_loader_v = data_loader_(group=pickle_jar, n_skill=4, max_seq=13, dev=True)
    #load model and cost function
    model=mdl.__dict__[config.model]
    model=model(input_size=186,output_size=1,num_blocks=4)
    loss_func=loss.__dict__[config.loss_func]
    callbacks = [MetricsCallback(input_key="targets", output_key="logits",
                         directory=config.weight_loc, model_name='transformer_v1',check_interval=1)]

    train(model=model,data_loader=data_loader, data_loader_v= data_loader_v,loss_func=loss_func,callbacks=callbacks,pretrained=config.weight_loc+'transformer_v1_3.pth')
