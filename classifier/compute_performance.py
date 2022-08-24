import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

from classifier.tools.resnet import resnet50
from classifier.tools.data_loader import MIDIDataset


def opts_parser():
    desc = 'Script to compute performance of given model.'
    parser = ArgumentParser(description=desc)
    parser.add_argument('modelfile', metavar='MODELFILE', type=str,
                        help='File to load the learned weights from (.pt format)')
    parser.add_argument('--omit_onset', action='store_true',
                        help='Whether to omit onset or not.')
    parser.add_argument('--split-root', type=str, help='Path pointing to data split files.')
    parser.add_argument('--cm', action='store_true', help='Switches to confusion-matrix computation.')
    return parser


def get_model_valid_data(modelfile, device, omit_onset, train_split=None, valid_split=None):
    """ Loads (pre-trained) model. """
    checkpoint = torch.load(modelfile, map_location=device)
    state_dict = {k.replace('module.', ''): checkpoint['model.state_dict'][k] for k in
                  checkpoint['model.state_dict'].keys()}
    if modelfile.name.startswith('resnet50'):
        model = resnet50(in_channels=(1 if omit_onset else 2), num_classes=13).to(device)
    else:
        raise NotImplementedError('Only supports resnet 50 for now')
    print(model)
    model.load_state_dict(state_dict)
    model.eval()

    # make training and validation data loader
    train = MIDIDataset(train=False, txt_file=train_split, classes=13, omit=None, seg_num=90, age=False, transform=None)
    train_loader = DataLoader(train, batch_size=90, shuffle=False)
    valid = MIDIDataset(train=False, txt_file=valid_split, classes=13, omit=None, seg_num=90, age=False, transform=None)
    valid_loader = DataLoader(valid, batch_size=90, shuffle=False)

    return model, valid_loader, train_loader


def compute_confusion_matrix(data, model, device, omit_onsets, data_name):
    """ Computes confusion matrix for given data. """
    confusion_matrix = np.zeros((13, 13))
    with torch.no_grad():
        for j, valset in tqdm(enumerate(data)):
            # val_in, val_out = valset
            val_in = valset["X"].to(device)
            val_out = valset["Y"].to(device)

            cur_true_label = int(val_out[0])
            if cur_true_label != int(val_out[-1]):
                print("Error!! => Diff label in same batch.")
                return
            if omit_onsets:
                val_in = val_in[:, 1:, :, :]  # note channel

            val_pred = model(val_in)  # probability
            _, val_label_pred = torch.max(val_pred.data, 1)
            val_label_pred = val_label_pred.tolist()
            cur_pred_label = max(val_label_pred, key=val_label_pred.count)

            # update confusion matrix
            confusion_matrix[cur_true_label][cur_pred_label] += 1

    print(confusion_matrix)
    np.save('confusion_matrix_{}.npy'.format(data_name), confusion_matrix)


def compute_performance(data, model, criterion, device, omit_onsets, data_type):
    """ Computes various performance measures according to original code of Kim et al. """
    with torch.no_grad():
        # average the acc of each batch
        val_loss, val_acc = 0.0, 0.0

        val_preds = []
        val_ground_truths = []
        val_correct = 0
        cur_midi_truths = []
        cur_pred_label = -1  # majority label

        for j, valset in tqdm(enumerate(data)):

            # val_in, val_out = valset
            val_in = valset["X"].to(device)
            val_out = valset["Y"].to(device)

            cur_true_label = int(val_out[0])
            cur_midi_truths.append(cur_true_label)
            if cur_true_label != int(val_out[-1]):
                print("Error!! => Diff label in same batch.")
                return

            if omit_onsets:
                val_in = val_in[:, 1:, :, :]  # note channel

            ################################################################
            val_pred = model(val_in)  # probability
            val_softmax = torch.softmax(val_pred, dim=1)
            batch_confidence = torch.sum(val_softmax, dim=0)  # =1
            batch_confidence = torch.div(batch_confidence, 90)  # avg value
            v_loss = criterion(val_pred, val_out)
            val_loss += v_loss
            # accuracy
            _, val_label_pred = torch.max(val_pred.data, 1)

            # changed accuracy metric
            # acc for each batch (=> one batch = one midi)
            val_label_pred = val_label_pred.tolist()

            occ = [val_label_pred.count(x) for x in range(13)]
            max_vote = max(occ)
            occ = np.array(occ)
            dup_list = np.where(max_vote == occ)[0]
            # returns indices of same max occ
            if len(dup_list) > 1:
                max_confidence = -1.0
                for dup in dup_list:
                    if batch_confidence[dup] > max_confidence:
                        cur_pred_label = dup
            else:
                cur_pred_label = max(val_label_pred, key=val_label_pred.count)
            if cur_true_label == cur_pred_label:
                val_correct += 1

            # f1 score
            val_preds.append(cur_pred_label)
            val_ground_truths.append(cur_true_label)

            # reset for next midi
            cur_midi_truths = []
            cur_pred_label = -1  # majority label

    avg_valloss = val_loss / len(data)

    # score
    # 1. accuracy
    print("============================================")

    val_acc = val_correct / len(data)

    # 2. weighted f1-score
    w_f1score = f1_score(val_ground_truths, val_preds, average="weighted")

    precision, recall, f1, supports = precision_recall_fscore_support(val_ground_truths, val_preds, average=None,
                                                                      labels=list(range(13)), warn_for=tuple())

    # print learning process
    print("\n######## {} #########".format(data_type))
    print("Accuracy: {:.4f} | Loss: {:.4f}" "".format(val_acc, avg_valloss))
    print("F1-score: %.4f" % (w_f1score))
    print("{:<30}{:<}".format("Precision", "Recall"))
    for p, r in zip(precision, recall):
        print("{:<30}{:<}".format(p, r))
    print()


def run(modelfile, omit_onset, split_root, cm):
    """ Prepare everything for performance check. """
    # prepare model, data
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    valid_split = split_root / 'valid.txt'
    train_split = split_root / 'train.txt'
    model, valid_loader, train_loader = get_model_valid_data(modelfile, device, omit_onset, train_split, valid_split)
    criterion = nn.CrossEntropyLoss()

    # run performance checks
    if cm:
        compute_confusion_matrix(train_loader, model, device, omit_onset, 'train')
        compute_confusion_matrix(valid_loader, model, device, omit_onset, 'valid')
    else:
        compute_performance(valid_loader, model, criterion, device, omit_onset, 'valid')


def main():
    # parse arguments
    parser = opts_parser()
    options = parser.parse_args()

    modelfile = Path(options.modelfile)
    if not modelfile.exists():
        raise FileNotFoundError('Please define valid model-file!')
    split_root = Path(options.split_root)
    if not split_root.exists():
        raise NotADirectoryError('Please define a valid path to split files!')

    run(modelfile, options.omit_onset, split_root, options.cm)


if __name__ == '__main__':
    main()
