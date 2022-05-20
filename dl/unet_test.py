import matplotlib.pyplot as plt
import tensorflow_estimator.python.estimator.early_stopping
import torch
import utils
import dl
import dl.metrics
import json
from dl.dataloader import KFoldValLoaders, CamusDatasetPNG
import os
from unet_model import Unet
from wavelet_unet_model import WaveletUnet
import pandas as pd
import numpy as np
from scipy import stats
import random
import itertools
import cv2
import torch
from torch.nn.functional import one_hot
import matplotlib.patches as mpatches


def load_unet(filename, output_ch, levels, top_feature_ch, wavelet=False, trainable_params=None):
    if wavelet:
        saved_unet = WaveletUnet(output_ch=output_ch, levels=levels, top_feature_ch=top_feature_ch)
        saved_unet.load_state_dict(torch.load(filename))
    else:
        saved_unet = Unet(output_ch=output_ch, levels=levels, top_feature_ch=top_feature_ch)
        saved_unet.load_state_dict(torch.load(filename))

    return saved_unet.cuda()


def evaluate_unet(unet, val_loader, only_f1=False):
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
                        metric_lists_ED[n].append(
                            dl.metrics.get_f1_metrics(prediction[:, n, :, :], seg[:, n, :, :], only_f1))
                    elif ED_or_ES[i] == 2:
                        metric_lists_ES[n].append(
                            dl.metrics.get_f1_metrics(prediction[:, n, :, :], seg[:, n, :, :], only_f1))

            next_batch = next(val_loader)

    except StopIteration:
        return metric_lists_ES, metric_lists_ED


def average_indvidual_metrics(metric_lists_ES, metric_lists_ED, val_metrics):
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
    return


def save_metrics(savename, val_metrics_ES_and_ED):
    """
    Saves metrics toa csv file.
    """
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
        avgs = calc_metric_folds_avgs(metrics_frame, list(val_metrics.keys()))
        m_frames.append(pd.concat([metrics_frame, avgs]))

    metrics_frame_full = pd.concat(m_frames, keys=list(val_metrics_ES_and_ED.keys()), axis=1)
    metrics_frame_full.to_csv(f'{savename}metrics.csv')
    return metrics_frame_full


def calc_metric_folds_avgs(metrics_frame, metrics_name):
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


def check_predictions(net_name, dataset_name, n_images=1):
    """
    Plots the predicted segmentation as an overlay to the image and the ground truth as an outline.
    """
    path = f"train_results/{dataset_name}/{net_name}/"
    checkpoint_path_list, settings = get_checkpoints_paths(path)
    val_loaders = KFoldValLoaders(CamusDatasetPNG(dataset_name), split=8)
    val_loader = val_loaders[0]
    unet = load_unet(path + checkpoint_path_list[0], **settings['unet_settings'])
    unet.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            img, seg, _ = data
            seg = seg[0]
            prediction = unet(img)

            prediction = torch.softmax(prediction, dim=1)
            prediction = (prediction > 0.5).float().squeeze(dim=0)
            dl.metrics.print_metrics(prediction, seg)
            print()
            prediction = prediction.cpu().detach().numpy()
            img = img.cpu().detach().squeeze(dim=0).squeeze(dim=0).numpy()
            seg = seg.cpu().detach().squeeze(dim=0).numpy().astype('float32')
            utils.plot_onehot_seg(img, prediction, outline=seg, title=f'{dataset_name}, {net_name}')
            ''' Legend:
            green: overlap
            orange: missed
            red: segmented background
            '''
            if i + 1 == n_images:
                return
            # utils.plot_image_g(np.abs(seg - prediction[0]), title='Difference')


def plot_image_prediction(net_name, dataset_name, img):
    """
    Plots predicted segmentation as an overlay on the image.
    """
    img = cv2.resize(img, (256, 256), cv2.INTER_LINEAR)
    img = utils.normalize_0_1(img.astype(np.float32))
    img = torch.tensor(img, device='cuda', dtype=torch.float)
    img = torch.unsqueeze(img, dim=0)
    img = torch.unsqueeze(img, dim=0)
    path = f"train_results/{dataset_name}/{net_name}/"
    checkpoint_path_list, settings = get_checkpoints_paths(path)
    unet = load_unet(path + checkpoint_path_list[0], **settings['unet_settings'])
    unet.eval()
    with torch.no_grad():
        prediction = unet(img)
        prediction = torch.softmax(prediction, dim=1)
        prediction = (prediction > 0.5).float().squeeze(dim=0)
        prediction = prediction.cpu().detach().numpy()
        utils.plot_onehot_seg(img, prediction, title=f'{dataset_name}, {net_name}')


def get_checkpoints_paths(path):
    checkpoint_paths = [path for path in os.listdir(path) if '.pt' in path]
    with open(f'{path}settings.json', 'r') as file:
        settings = json.load(file)
        return checkpoint_paths, settings


def val_folds(net_name, dataset_name):
    path = f"train_results/{dataset_name}/{net_name}/"
    val_metrics = {'ED': {'p': [], 'r': [], 'f1': []}, 'ES': {'p': [], 'r': [], 'f1': []}}
    checkpoint_path_list, settings = get_checkpoints_paths(path)
    val_loaders = KFoldValLoaders(CamusDatasetPNG(dataset_name), split=len(checkpoint_path_list))
    for i, checkpoint_path in enumerate(checkpoint_path_list):
        # print(f'Evaluating fold {i}')
        val_loader = val_loaders[i]
        unet = load_unet(path + checkpoint_path, **settings['unet_settings'])
        # check_predictions(unet, val_loader)
        ES_list, ED_list = evaluate_unet(unet, val_loader)
        average_indvidual_metrics(ES_list, ED_list, val_metrics)
    eval_results = save_metrics(f'{path}val_', val_metrics)
    return eval_results


def eval_test_set(net_name, dataset_name):
    path = f"train_results/{dataset_name}/{net_name}/"
    test_set = CamusDatasetPNG(dataset=f'{dataset_name}_test')
    test_metrics = {'ED': {'p': [], 'r': [], 'f1': []}, 'ES': {'p': [], 'r': [], 'f1': []}}
    subset = dl.dataloader.MySubset(test_set, indices=list(range(len(test_set))), transformer=None)
    checkpoint_path_list, settings = get_checkpoints_paths(path)
    test_loader = dl.dataloader.DataLoader(subset, batch_size=1, shuffle=False)
    for checkpoint_path in checkpoint_path_list:
        unet = load_unet(path + checkpoint_path, **settings['unet_settings'])
        ES_list, ED_list = evaluate_unet(unet, test_loader)
        average_indvidual_metrics(ES_list, ED_list, test_metrics)
    return save_metrics(f'{path}test_', test_metrics)


def get_ED_ES_list_test(net_name, dataset_name):
    ED, ES = [[], [], []], [[], [], []]
    path = f"train_results/{dataset_name}/{net_name}/"
    test_set = CamusDatasetPNG(dataset=f'{dataset_name}_test')
    subset = dl.dataloader.MySubset(test_set, indices=list(range(len(test_set))), transformer=None)
    checkpoint_path_list, settings = get_checkpoints_paths(path)
    test_loader = dl.dataloader.DataLoader(subset, batch_size=1, shuffle=False)
    for checkpoint_path in checkpoint_path_list:
        unet = load_unet(path + checkpoint_path, **settings['unet_settings'])
        ES_list, ED_list = evaluate_unet(unet, test_loader, only_f1=True)
        for i in range(1, 4):  # skip the background class
            ES[i - 1].append(ES_list[i])
            ED[i - 1].append(ED_list[i])
    return ED, ES


def average_folds(f1_scores):
    f1_scores = torch.tensor(f1_scores)
    averaged_scores = [[] for _ in range(f1_scores.shape[0])]  # one list per class
    for i in range(f1_scores.shape[-1]):
        for j in range(f1_scores.shape[0]):
            averaged_scores[j].append(torch.mean(f1_scores[j, :, i]).item())

    return averaged_scores


def wilcox_test(net_name1, net_name2, dataset_name1, dataset_name2):
    ED1, ES1 = get_ED_ES_list_test(net_name1, dataset_name1)
    ED2, ES2 = get_ED_ES_list_test(net_name2, dataset_name2)

    ED1 = average_folds(ED1)
    ED2 = average_folds(ED2)
    ES1 = average_folds(ES1)
    ES2 = average_folds(ES2)

    for nclass in range(len(ED1)):
        EDwil = stats.wilcoxon(ED1[nclass], ED2[nclass])
        ESwil = stats.wilcoxon(ES1[nclass], ES2[nclass])
        if EDwil.pvalue > 0.01:
            print(f'ED class {nclass + 1} pvalue: {EDwil.pvalue:.4f}  statistic: {EDwil.statistic}')
        if ESwil.pvalue > 0.01:
            print(f'ES class {nclass + 1} pvalue: {ESwil.pvalue:.4f}  statistic: {ESwil.statistic}')


def wilx_compare_all():
    net_name1 = 'unet_5levels_augment_False_64top'
    datasets = os.listdir('train_results')

    for combination in itertools.combinations(datasets, 2):
        dataset1 = combination[0]
        dataset2 = combination[1]
        print(f'\n\n\n{dataset1} and {dataset2}')
        wilcox_test(net_name1, net_name1, dataset1, dataset2)


def scrap_volume(net_name, dataset_name):
    """
    Displays the image with segmentations that contain more than 1 volume per class, and calculates how large they
    are relative to the target area.
    """
    path = f"train_results/{dataset_name}/{net_name}/"
    test_set = CamusDatasetPNG(dataset=f'{dataset_name}_test')
    subset = dl.dataloader.MySubset(test_set, indices=list(range(len(test_set))), transformer=None)
    checkpoint_path_list, settings = get_checkpoints_paths(path)
    test_loader = dl.dataloader.DataLoader(subset, batch_size=1, shuffle=False)
    for checkpoint_path in checkpoint_path_list:
        unet = load_unet(path + checkpoint_path, **settings['unet_settings'])
        unet.eval()
        val_loader = iter(test_loader)
        next_batch = next(val_loader)
        n_classes = next_batch[1].size()[1]
        img_number = 0
        try:
            while True:
                with torch.no_grad():
                    img_number = + 1
                    img, seg, ED_or_ES = next_batch
                    prediction = torch.softmax(unet(img), dim=1)

                for i in range(img.shape[0]):
                    for n in range(3, n_classes):
                        pred = prediction[0, n, :, :].detach().cpu().numpy()
                        class_seg = seg[0, n, :, :].detach().cpu().numpy()
                        pred = utils.binarize(pred)
                        pred = (pred * 255).astype(np.uint8)
                        n_vols, labeled_img, stats, centroids = cv2.connectedComponentsWithStats(pred,
                                                                                                 stats=cv2.CC_STAT_AREA)
                        areas = stats[:, -1][1:]
                        n_vols -= 1
                        if n_vols > 1:
                            labeled_img = one_hot(torch.tensor(labeled_img).type(torch.int64),
                                                  num_classes=n_vols + 1).permute(2, 0, 1).numpy()[1:]

                            flat_seg = class_seg.flatten()
                            seg_vol = np.sum(flat_seg)
                            total_scrap_area = 0
                            del_indexes = []
                            for n in range(n_vols):
                                vol_area = labeled_img[n].flatten()
                                union = np.sum(vol_area * flat_seg)
                                if (union / seg_vol) < 0.4:
                                    total_scrap_area += np.sum(vol_area)
                                else:
                                    del_indexes.append(n)
                                    labeled_img = np.insert(labeled_img, 0, labeled_img[n], axis=0)
                                    labeled_img = np.delete(labeled_img, n + 1, axis=0)
                                    # labeled_img = np.moveaxis(labeled_img, n, 0)

                            percent_scrap_vol = (total_scrap_area / np.sum(areas)) * 100
                            utils.plot_onehot_seg(img, labeled_img, outline=seg[0, :, :, :],
                                                  colors=['green', *['red'] * 5],
                                                  title=f'Img: {img_number}, Class {n}, Scrap volumes: {n_vols - 1}\n'
                                                        f' Scrap area: {percent_scrap_vol:.2f}%')

                    next_batch = next(val_loader)

        except StopIteration:
            return


def disp_bad_segs(net_name, dataset_name, score_threshold=0.7, worse_than_thresh=True, seg_class=None):
    """
    Displays segmentations with classes that have f1 scores under threshold.
    """
    path = f"train_results/{dataset_name}/{net_name}/"
    test_set = CamusDatasetPNG(dataset=f'{dataset_name}_test')
    subset = dl.dataloader.MySubset(test_set, indices=list(range(len(test_set))), transformer=None)
    checkpoint_path_list, settings = get_checkpoints_paths(path)
    test_loader = dl.dataloader.DataLoader(subset, batch_size=1, shuffle=False)
    fold = 0
    for checkpoint_path in checkpoint_path_list:
        unet = load_unet(path + checkpoint_path, **settings['unet_settings'])
        unet.eval()
        val_loader = iter(test_loader)
        next_batch = next(val_loader)
        n_classes = next_batch[1].size()[1]
        try:
            while True:
                with torch.no_grad():
                    img, seg, ED_or_ES = next_batch
                    prediction = torch.softmax(unet(img), dim=1)

                for i in range(img.shape[0]):
                    for n in range(1, n_classes):

                        if seg_class:
                            n = seg_class

                        pred = prediction[:, n, :, :]
                        class_seg = seg[:, n, :, :]
                        # pred = utils.binarize(pred)
                        # pred = (pred * 255).astype(np.uint8)

                        f1 = dl.metrics.get_f1_metrics(pred, class_seg, only_f1=True)

                        display = f1 < score_threshold
                        if not worse_than_thresh:
                            display = not display

                        if display:
                            utils.plot_onehot_seg(img[0], prediction[0], outline=seg[0], alpha_overlay=0.1,
                                                  title=f'Fold: {fold}, Class {n}, F1: {f1:.3f}')

                        if seg_class:
                            break

                next_batch = next(val_loader)

        except StopIteration:
            fold += 1
            pass


def score_boxplots(net_names, dataset_names, colors=None, legend_names=None):
    """
    Creates grouped box plots of the scores of each class of test set predictions of networks given by lists of net_names and dataset_names.
    :param net_names: List of network folder names
    :param dataset_names: List of dataset folder names
    """
    ED = []
    ES = []
    for net_name, dataset_name in zip(net_names, dataset_names):
        ED_temp, ES_temp = get_ED_ES_list_test(net_name, dataset_name)
        ED.append(average_folds(ED_temp))
        ES.append(average_folds(ES_temp))
    print([x for x in range(1, len(ED) * (len(ED[0])), len(ED))])
    if colors is None:
        colors = ['red', 'blue', 'green', 'orange', 'black']

    fig, (ax_ED, ax_ES) = plt.subplots(1, 2, figsize=(10, 5), dpi=220)
    ax_ED.set_ylabel('F1 Score')
    ax_ED = utils.boxplots(ax_ED, ED, colors, title=f'End Dyastole (ED)')
    ax_ES = utils.boxplots(ax_ES, ES, colors, title=f'End Systole (ES)')

    if legend_names is not None:
        handles = []
        for i, name in enumerate(legend_names):
            handles.append(mpatches.Patch(color=colors[i], label=name))

        ax_ED.legend(handles=handles, loc='lower left')

    ax_ED.set_xticks([x + 1 for x in range(1, len(ED) * (len(ED[0]) + 1), len(ED) + 1)])
    ax_ED.set_xticklabels(['LV endo', 'LV epi', 'LA'])
    ax_ES.set_xticks([x + 1 for x in range(1, len(ES) * (len(ES[0]) + 1), len(ES) + 1)])
    ax_ES.set_xticklabels(['LV endo', 'LV epi', 'LA'])

    plt.show()


if __name__ == '__main__':
    net_name1 = 'testwavelet_newunet_5level_augment_False_32top'
    net_5_64 = 'unet_5levels_augment_False_64top'
    net_noise_aug = 'unet_5levels_augment_noise_64top'
    net_newwav = 'wavelet_newunet_5level_augment_False_32top'
    dataset_png = 'camus_png'
    dataset_hmf = 'camus_hmf'
    dataset_bayes = 'camus_wavelet_sigma0.15_bayes'
    dataset_combined = 'camus_combined_50-0.1_w0.7_eps0.001'

    # legend_names = ['Standard', 'HMF', 'BayesShrink']
    # score_boxplots([net_5_64, net_noise_aug], [dataset_combined, dataset_combined],
    #                legend_names=legend_names)

    # wilx_compare_all()
    # datasets = os.listdir('train_results')
    # # for dataset_name in datasets:
    #
    # disp_bad_segs(net_name1, dataset_name, score_threshold=0.9, worse_than_thresh=False, seg_class=3)
    #
    # imgs = utils.load_images()
    # img = imgs[-1]
    # predict_image(net_name1, dataset_name, img)

    # scrap_volume(net_name1, dataset_name)

    # check_predictions(net_name1, dataset_name, n_images=3)
    # eval_results = val_folds(net_name1, dataset_name)
    #
    # # print(f'\n\n{dataset_name}')

    # eval_results = eval_test_set(net_newwav, dataset_combined)
    # eval_results = val_folds(net_newwav, dataset_combined)
    # with pd.option_context('precision', 3):
    #     print('ED')
    #     print(eval_results.xs('avg').xs('ED', axis=1))
    #     print('\nES')
    #     print(eval_results.xs('avg').xs('ES', axis=1))
    #
    #
    # net_name2 = 'unet_5levels_augment_False_64top'
    # dataset_name2 = 'camus_png'
    wilcox_test(net_5_64, net_newwav, dataset_png, dataset_combined)
