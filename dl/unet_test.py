import torch
import utils
import dl
import json
from dl.dataloader import KFoldValLoaders, CamusDatasetPNG
import os
from unet_training import load_unet, evaluate_unet, save_metrics


def check_predictions(unet, val_loader, n_images):
    unet.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            img, seg, _ = data
            seg = seg[0]
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
            ''' Legend:
            green: overlap
            orange: missed
            red: segmented background
            '''
            if i + 1 == n_images:
                return
            # utils.plot_image_g(np.abs(seg - prediction[0]), title='Difference')


def get_checkpoints_paths(path):
    checkpoint_paths = [path for path in os.listdir(path) if '.pt' in path]
    with open(f'{path}settings.json', 'r') as file:
        settings = json.load(file)
    return checkpoint_paths, settings


def val_folds(net_name):
    path = f"train_results/{net_name}/"
    val_metrics = {'ED': {'p': [], 'r': [], 'f1': []}, 'ES': {'p': [], 'r': [], 'f1': []}}
    checkpoint_path_list, settings = get_checkpoints_paths(path)
    val_loaders = KFoldValLoaders(CamusDatasetPNG(), split=len(checkpoint_path_list))
    for i, checkpoint_path in enumerate(checkpoint_path_list):
        print(i)
        val_loader = val_loaders[i]
        unet = load_unet(path + checkpoint_path, **settings['unet_settings'])
        # check_predictions(unet, val_loader)
        evaluate_unet(unet, val_loader, val_metrics)
    eval_results = save_metrics(f'{path}test_', val_metrics)
    print('ED')
    print(eval_results.xs('avg').xs('ED', axis=1))
    print('\nES')
    print(eval_results.xs('avg').xs('ES', axis=1))


def eval_test_set(unet, net_name):
    test_set = CamusDatasetPNG(dataset='camus_png_test')
    test_metrics = {'ED': {'p': [], 'r': [], 'f1': []}, 'ES': {'p': [], 'r': [], 'f1': []}}
    subset = dl.dataloader.MySubset(test_set, indices=list(range(len(test_set))), transformer=None)
    test_loader = dl.dataloader.DataLoader(subset, batch_size=1, shuffle=True)
    evaluate_unet(unet, test_loader, test_metrics)
    save_metrics(f'train_results/{net_name}/test_', test_metrics)


if __name__ == '__main__':
    path = 'train_results/unet_4levels_augment_False_16top/fold_0.pt'
    unet = load_unet(path, out_channels=4, levels=4, top_ch=64)
    val_loaders = KFoldValLoaders(CamusDatasetPNG(), split=8)
    check_predictions(unet, val_loaders[0], n_images=1)
    # val_folds('unet_5levels_augment_False_64top')
    print()
    # eval_test_set(unet)
