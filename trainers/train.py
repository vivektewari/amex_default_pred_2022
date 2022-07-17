from catalyst.dl import SupervisedRunner
from config import *
from utils.funcs import get_dict_from_class,count_parameters
import torch.optim as optim
from torch.utils.data import DataLoader
#from callbacks import MetricsCallback
from sklearn.model_selection import StratifiedKFold


def train(model_param,model_,data_loader_param,data_loader,loss_func,callbacks=None,pretrained=None):
    randSeed=23
    data_load = data_loader(**get_dict_from_class(data_loader_param))
    criterion = loss_func
    model = model_(**get_dict_from_class(model_param))
    count_parameters(model)
    # model = FCLayered(**get_dict_from_class(model_param,model))
    if pretrained is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(pretrained, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    skf = StratifiedKFold(n_splits=15, shuffle=True, random_state=randSeed)
    train = data_load.data
    train["fold"] = -1

    # train.set_index('index',inplace=True)
    for fold_id, (train_index, val_index) in enumerate(skf.split(train, train["fold"])):
        train.iloc[val_index, -1] = fold_id

    # # check the proportion
    fold_proportion = pd.pivot_table(train, columns="fold", values="label", aggfunc=len)

    use_fold = 0
    train[0:100]['fold']=1# keeping initial few file in training for seeing bounding box quality
    train_file = train.query("fold != @use_fold")
    val_file = train.query("fold == @use_fold")

    print("[fold {}] train: {}, val: {}".format(use_fold, len(train_file), len(val_file)))

    loaders = {
        "train": DataLoader(data_loader(data_frame=train_file, **get_dict_from_class(data_loader_param)),
                            batch_size=16,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False),
        "valid": DataLoader(data_loader(data_frame=val_file, **get_dict_from_class(data_loader_param)),
                            batch_size=256,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False)
    }

    callbacks = callbacks
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
    a=loss_func
    callbacks = [MetricsCallback_loc(input_key="targets", output_key="logits",
                        directory=saveDirectory, model_name='featureExtr_4',func=a.get_individual_loss,pixel = model_param.input_image_dim[0])]

    train(model_param,model,data_loader_param,data_loader,loss_func,callbacks,pretrained=pre_trained_model)#'/home/pooja/PycharmProjects/digitRecognizer/rough/localization/fold0/checkpoints/train.138.pth')