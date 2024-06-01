from ml.models.HessNet import HessNet
from ml.models.unet3d import U_Net, Unet3d
from ml.models.unet2d import U_Net2d
from ml.models.JoB_VS import Network
from ml.models.VesselConvs import JustConv, TwoConv
from ml.transformers_models.UNETR import UNETR
from ml.models.GenUnet import GenUnet


def get_model(model_name):
    if model_name == 'Unet3d_16ch':
        #return(Unet3d(channels=16))
        return(U_Net(channels=16))
        
    elif model_name == 'Unet2d_16ch':
        return(U_Net2d(channels=16))

    elif model_name == "LittleUnet":
        return(GenUnet(dim=3, in_channels=1, out_channels=1,
               channels=8, depth=1))

    elif model_name == 'HessNet':
        return(HessNet(in_channels=1, out_channels=1, n_hess_blocks=4))

    elif model_name == 'UNETR':
        return(UNETR(in_channels=1, out_channels=1, img_size=(64, 64, 64),
              feature_size=16, hidden_size=512,
              mlp_dim=512, num_heads=4,
              norm_name='batch'))

    elif model_name == 'JoB-VS':
        return(Network(modalities=1, num_classes=1))

    elif model_name == 'JustConv':
        return(JustConv(1, 1))

    elif model_name == 'TwoConv':
        return(TwoConv(1, 1, 5))

    else:
        raise RuntimeError(f"ml.get_model::Error: unknown model name <{model_name}>")