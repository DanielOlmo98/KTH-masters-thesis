import torch
import utils
import dl
import dl.metrics
import json
from dl.dataloader import KFoldValLoaders, CamusDatasetPNG
import os
from unet_model import Unet
import pandas as pd
import numpy as np
from scipy import stats


def load_unet(filename, output_ch, levels, top_feature_ch):
    saved_unet = Unet(output_ch=output_ch, levels=levels, top_feature_ch=top_feature_ch)
    saved_unet.load_state_dict(torch.load(filename))
    return saved_unet.cuda()


def evaluate_unet(unet, val_loader):
    unet.eval()
    val_loader = iter(val_loader)
    next_batch = next(val_loader)
    n_classes = next_batch[1].size()[1]
    metric_lists_ES = [[] for _ in range(n_classes)]  # one list per class
    metric_lists_ED = [[] for _ in range(n_classes)]  # one list per class
    try:
        while True:
            with torch.no_grad():
                img, seg, ED_or_ES = next_batch
                prediction = torch.softmax(unet(img), dim=1)

            for i in range(img.shape[0]):
                for n in range(n_classes):
                    # append 1x3 tensor per class containing precision recall and f1 for the class
                    if ED_or_ES[i] == 1:
                        metric_lists_ED[n].append(dl.metrics.get_f1_metrics(prediction[:, n, :, :], seg[:, n, :, :]))
                    elif ED_or_ES[i] == 2:
                        metric_lists_ES[n].append(dl.metrics.get_f1_metrics(prediction[:, n, :, :], seg[:, n, :, :]))

            next_batch = next(val_loader)

    except StopIteration:
        return metric_lists_ES, metric_lists_ED


def average_metrics(metric_lists_ES, metric_lists_ED, val_metrics):
    metric_lists_ED = torch.FloatTensor(metric_lists_ED)
    metric_lists_ES = torch.FloatTensor(metric_lists_ES)
    for metric_list, val_m_dict_key in zip([metric_lists_ED, metric_lists_ES], ['ED', 'ES']):
        p_list = []
        r_list = []
        f1_list = []
        for n in range(metric_lists_ED.size()[0]):
            p_list.append(torch.mean(metric_list[n, :, 0]))
            r_list.append(torch.mean(metric_list[n, :, 1]))
            f1_list.append(torch.mean(metric_list[n, :, 2]))
        val_metrics[val_m_dict_key]['p'].append(p_list)
        val_metrics[val_m_dict_key]['r'].append(r_list)
        val_metrics[val_m_dict_key]['f1'].append(f1_list)
    pass


def save_metrics(savename, val_metrics_ES_and_ED):
    m_frames = []
    for val_metrics in val_metrics_ES_and_ED.values():
        fold_arrays = []
        p_tensor = torch.tensor(val_metrics['p'])
        r_tensor = torch.tensor(val_metrics['r'])
        f1_tensor = torch.tensor(val_metrics['f1'])
        folds, n_classes = p_tensor.shape
        for fold in range(folds):
            fold_arrays.append(np.stack((p_tensor[fold], r_tensor[fold], f1_tensor[fold])))

        row_idxs = pd.MultiIndex.from_product(
            [range(folds), list(val_metrics.keys())],
            names=['fold', 'metric']
        )
        col_sub_idxs = pd.MultiIndex.from_product([range(n_classes)], names=['class'])
        metrics_frame = pd.DataFrame(np.vstack(fold_arrays), index=row_idxs, columns=col_sub_idxs)
        avgs = calc_metric_avgs(metrics_frame, list(val_metrics.keys()))
        m_frames.append(pd.concat([metrics_frame, avgs]))

    metrics_frame_full = pd.concat(m_frames, keys=list(val_metrics_ES_and_ED.keys()), axis=1)
    metrics_frame_full.to_csv(f'{savename}metrics.csv')
    return metrics_frame_full


def calc_metric_avgs(metrics_frame, metrics_name):
    avgs = []
    f1_std = []
    for metric in metrics_name:
        avgs.append(metrics_frame.xs(metric, level=1).mean())
        if metric == 'f1':
            f1_std.append(metrics_frame.xs(metric, level=1).std())

    metrics_name.append('f1_std_dev')
    row_idxs = pd.MultiIndex.from_product(
        [['avg'], metrics_name],
        names=['fold', 'metric']
    )
    return pd.DataFrame([*avgs, *f1_std], index=row_idxs)


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
        # print(f'Evaluating fold {i}')
        val_loader = val_loaders[i]
        unet = load_unet(path + checkpoint_path, **settings['unet_settings'])
        # check_predictions(unet, val_loader)
        ES_list, ED_list = evaluate_unet(unet, val_loader)
        average_metrics(ES_list, ED_list, val_metrics)
    eval_results = save_metrics(f'{path}val_', val_metrics)
    return eval_results


def eval_test_set(unet, net_name):
    test_set = CamusDatasetPNG(dataset='camus_png_test')
    test_metrics = {'ED': {'p': [], 'r': [], 'f1': []}, 'ES': {'p': [], 'r': [], 'f1': []}}
    subset = dl.dataloader.MySubset(test_set, indices=list(range(len(test_set))), transformer=None)
    test_loader = dl.dataloader.DataLoader(subset, batch_size=1, shuffle=True)
    ES_list, ED_list = evaluate_unet(unet, test_loader)
    average_metrics(ES_list, ED_list, test_metrics)
    save_metrics(f'train_results/camus_png/{net_name}/test_', test_metrics)


def wilcox_test(net_name1, net_name2):
    """
    TEST
    :param net_name1:
    :param net_name2:
    :return:
    """

    def get_ED_ES_lists(net_name):
        path = f"train_results/{net_name}/"

        with open(f'{path}settings.json', 'r') as file:
            settings = json.load(file)

        test_set = CamusDatasetPNG(dataset=settings['dataloader_settings']['dataset']['path'].split("/")[2] + "_test")

        subset1 = dl.dataloader.MySubset(test_set, indices=list(range(len(test_set))), transformer=None)
        test_loader1 = dl.dataloader.DataLoader(subset1, batch_size=1, shuffle=True)

        unet = load_unet()

        return evaluate_unet(unet, test_loader1)

    ED1, ES1 = get_ED_ES_lists(net_name1)
    ED2, ES2 = get_ED_ES_lists(net_name2)

    stats.wilcoxon(ED1, ED2)
    stats.wilcoxon(ES1, ES2)


if __name__ == '__main__':
    # path = 'train_results/camus_png/unet_5levels_augment_False_16top/fold_0.pt'
    # unet = load_unet(path, out_channels=4, levels=5, top_ch=64)
    # val_loaders = KFoldValLoaders(CamusDatasetPNG(), split=8)
    # check_predictions(unet, val_loaders[0], n_images=1)
    for net_folder in os.listdir('train_results/camus_png'):
        print(f'\n\n\n{net_folder}')
        eval_results = val_folds(f'camus_png/{net_folder}')
        with pd.option_context('precision', 3):
            print('ED')
            print(eval_results.xs('avg').xs('ED', axis=1))
            print('\nES')
            print(eval_results.xs('avg').xs('ES', axis=1))
    # eval_test_set(unet)
