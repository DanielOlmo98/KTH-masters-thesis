import torch
import utils
import dl
from dl.dataloader import KFoldValLoaders, CamusDatasetPNG
import os
from unet_training import load_unet


def check_predictions(unet, val_loader):
    unet.eval()
    with torch.no_grad():
        for data in iter(val_loader):
            img, seg, _ = data
            seg = seg[0]
            # img = img[0]
            # img = img.unsqueeze(dim=0)
            prediction = unet(img)

            prediction = torch.softmax(prediction, dim=1)
            prediction = (prediction > 0.5).float().squeeze(dim=0)
            dl.metrics.print_metrics(prediction, seg)
            prediction = prediction.cpu().detach().numpy()
            img = img.cpu().detach().squeeze(dim=0).squeeze(dim=0).numpy()
            seg = seg.cpu().detach().squeeze(dim=0).numpy().astype('float32')
            # utils.plot_onehot_seg(img, seg, title='Ground Truth')
            # utils.plot_onehot_seg(img, prediction, title='Prediction')
            utils.plot_onehot_seg(img, prediction, outline=seg)
            '''q
            green: overlap
            orange: missed
            red: segmented background
            '''
            # utils.plot_image_g(np.abs(seg - prediction[0]), title='Difference')


def get_checkpoints_paths(path):
    checkpoint_paths = [path for path in os.listdir(path) if '.pt' in path]
    with open(f'{path}settings.txt', 'r') as file:
        settings = file.read()
        # TODO read settings, change save settings format first
    return checkpoint_paths, settings


def val_folds(net_name):
    path = f"{utils.get_project_root()}/dl/train_results/{net_name}/"
    val_metrics = {'ED': {'p': [], 'r': [], 'f1': []}, 'ES': {'p': [], 'r': [], 'f1': []}}
    checkpoint_path_list, settings = get_checkpoints_paths(path)
    val_loaders = KFoldValLoaders(CamusDatasetPNG(), split=len(checkpoint_path_list))
    for i, checkpoint_path in enumerate(checkpoint_path_list):
        val_loader = val_loaders[i]
        unet = load_unet(checkpoint_path, out_channels=4, levels=levels, top_ch=top_ch)
        check_predictions(unet, val_loader)
        dl.unet_training.evaluate_unet(unet, val_loader, val_metrics)
        dl.unet_training.save_metrics(f'{path}test_', val_metrics)


if __name__ == '__main__':
    path = 'train_results/unet_4levels_augment_False_64top/ fold_0.pt'
    unet = load_unet(path, out_channels=4, levels=4, top_ch=64)
    val_loaders = KFoldValLoaders(CamusDatasetPNG(), split=8)
    check_predictions(unet, val_loaders[0])
    print()
