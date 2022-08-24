""" Taken from original repo (https://github.com/KimSSung/Deep-Composer-Classification/blob/master/trainer.py) """

# Trainer class (config.mode = 'basetrain' -> base training

# from config import get_config
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

# from tqdm import tqdm

# to import from sibling folders
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# from models.resnet_ver2 import resnet18, resnet34, resnet50, resnet101, resnet152

from classifier.tools.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

# dataloader
from classifier.tools.data_loader import MIDIDataset

# score metric
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

# from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from config import splits_root


class Trainer:
    def __init__(self, args, save_dir):
        self.config = args

        # 0 : acc / 1: loss / 2: f1 / 3: precision / 4: recall
        self.best_valid = [-1.0, 30000.0, -1.0, [], []]

        if self.config.onset is True:
            self.input_shape = (2, 400, 88)
        elif self.config.onset is False:
            self.input_shape = (1, 400, 88)

        self.valid_seg = self.config.val_seg
        self.train_seg = self.config.trn_seg

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.omitlist = []
        if self.config.omit:
            self.omitlist = self.config.omit.split(",")  # ['2', '5']. str list.

        self.label_num = self.config.composers - len(self.omitlist)
        print("\n==> Total label # :", self.label_num)
        # if age == True ==> label: 0, 1, 2
        if self.config.age:
            self.label_num = 3

        # save dir
        self.save_dir = save_dir
        print("==> SAVE at {}\n".format(self.save_dir))
        os.makedirs(os.path.join(self.save_dir, "model"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "dataset/train"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "dataset/valid"), exist_ok=True)

        print("mode", self.config.mode)
        self.data_load(self.config.mode)
        self.num_batches = len(self.train_loader)

        # Define model
        self.model = self.model_selection()
        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(self.device)

        # Define optimizer
        self.optimizer = self.optim_selection()
        print()
        print("==> Optim: ", self.optimizer)

        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.epochs
        )
        # self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',factor=0.5,patience=10,verbose=True) #0.5 best for midi370

        # tensorboard
        # self.writer = SummaryWriter("trainlog/")
        self.valid_times = 0  # increased when validation called


    def model_selection(self):
        if self.config.model_name == "resnet18":
            return resnet18(
                in_channels=int(self.input_shape[0]), num_classes=self.label_num
            )
        elif self.config.model_name == "resnet34":
            return resnet34(
                in_channels=int(self.input_shape[0]), num_classes=self.label_num
            )
        elif self.config.model_name == "resnet50":
            return resnet50(
                in_channels=int(self.input_shape[0]), num_classes=self.label_num
            )
        elif self.config.model_name == "resnet101":
            return resnet101(
                in_channels=int(self.input_shape[0]), num_classes=self.label_num
            )
        elif self.config.model_name == "resnet152":
            return resnet152(
                in_channels=int(self.input_shape[0]), num_classes=self.label_num
            )
        elif self.config.model_name == "convnet":
            return convnet(
                in_channels=int(self.input_shape[0]), num_classes=self.label_num
            )

        elif self.config.model_name == "resnet50":
            return wide_resnet50_2(
                in_channels=int(self.input_shape[0]), num_classes=self.label_num
            )
        elif self.config.model_name == "resnet101":
            return wide_resnet101_2(
                in_channels=int(self.input_shape[0]), num_classes=self.label_num
            )

    def optim_selection(self):
        if self.config.optim == "Nesterov":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.lr,
                momentum=0.9,
                nesterov=True,
                weight_decay=0.0001,
            )
        elif self.config.optim == "SGD":  # weight_decay = l2 regularization
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.lr,
                momentum=0.9,
                nesterov=False,
                weight_decay=0.0001,
            )
        elif self.config.optim == "Adadelta":  # default lr = 1.0
            return optim.Adadelta(
                self.model.parameters(),
                lr=self.config.lr,
                rho=0.9,
                eps=1e-06,
                weight_decay=1e-6,
            )
        elif self.config.optim == "Adagrad":  # default lr = 0.01
            return optim.Adagrad(
                self.model.parameters(),
                lr=self.config.lr,
                lr_decay=0,
                weight_decay=1e-6,
                initial_accumulator_value=0,
                eps=1e-10,
            )
        elif self.config.optim == "Adam":  # default lr=0.001
            return optim.Adam(
                self.model.parameters(), lr=self.config.lr, weight_decay=1e-6
            )
        elif self.config.optim == "AdamW":  # default lr=0.001
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.01,
                amsgrad=False,
            )
        elif self.config.optim == "SparseAdam":  # default lr = 0.001
            return optim.SparseAdam(
                self.model.parameters(),
                lr=self.config.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
            )
        elif self.config.optim == "Adamax":  # default lr=0.002
            return optim.Adamax(
                self.model.parameters(),
                lr=self.config.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=1e-6,
            )
        elif self.config.optim == "ASGD":
            return optim.ASGD(
                self.model.parameters(),
                lr=self.config.lr,
                lambd=0.0001,
                alpha=0.75,
                t0=1000000.0,
                weight_decay=1e-6,
            )
        elif self.config.optim == "RMSprop":  # default lr=0.01
            return optim.RMSprop(
                self.model.parameters(),
                lr=self.config.lr,
                alpha=0.99,
                eps=1e-08,
                weight_decay=0,
                momentum=0,
                centered=False,
            )
        elif self.config.optim == "Rprop":  # default lr=0.01
            return optim.Rprop(
                self.model.parameters(),
                lr=self.config.lr,
                etas=(0.5, 1.2),
                step_sizes=(1e-06, 50),
            )

    def data_load(self, mode):
        transpose_rng = None
        if mode == "basetrain":
            print(">>>>>> Base Training <<<<<<\n")

            # Loader for base training
            if self.config.transform is not None:
                print("+++ Add {}".format(self.config.transform))
                if "Transpose" in self.config.transform:
                    transpose_rng = int(self.config.transform.replace("Transpose", ""))
                elif "Tempo" in self.config.transform:
                    pass
                else:
                    print("Wrong Augmentation Command!")

            print("==> train seg:", self.train_seg)
            print("==> valid seg: ", self.valid_seg)
            print("==> train batch:", self.config.train_batch)
            print("==> valid batch:", self.valid_seg)
            print()
            
            t = MIDIDataset(
                train=True,  # newly added
                txt_file=os.path.join(splits_root, "train.txt"),
                classes=self.label_num,
                omit=self.config.omit,  # str
                seg_num=self.train_seg,
                age=self.config.age,
                transform=self.config.transform,
                transpose_rng=transpose_rng,
            )
            v = MIDIDataset(
                train=False,  # newly added
                txt_file=os.path.join(splits_root, "valid.txt"),
                classes=self.label_num,
                omit=self.config.omit,
                seg_num=self.valid_seg,
                age=self.config.age,
                transform=None,
            )

            # create batch
            self.train_loader = DataLoader(
                t, batch_size=self.config.train_batch, shuffle=True
            )
            self.valid_loader = DataLoader(v, batch_size=self.valid_seg, shuffle=False)

            ###################### Loader for base training #############################

            # save train_loader & valid_loader
            if self.config.save_trn:
                torch.save(
                    self.train_loader, os.path.join(self.save_dir, "dataset", "train", "train_loader.pt"),
                )
                print("train_loader saved!")
                torch.save(
                    self.valid_loader, os.path.join(self.save_dir, "dataset", "valid", "valid_loader.pt"),
                )
                print("valid_loader saved!")

                # load train_loader & valid_loader (to check whether well saved)
                self.train_loader = torch.load(
                    os.path.join(self.save_dir, "dataset", "train", "train_loader.pt"),
                )
                print("train_loader loaded!")
                self.valid_loader = torch.load(
                    os.path.join(self.save_dir, "dataset", "valid", "valid_loader.pt"),
                )
                print("valid_loader loaded!")

    def set_mode(self, mode="train"):
        if mode == "train":
            self.model.train()
        elif mode == "eval":
            self.model.eval()
        else:
            raise ("Mode error. It should be either train or eval")

    def train(self, mode):

        self.set_mode("train")  # model.train()

        # print input shape
        print("\nInput shape:", self.input_shape)
        print()

        # train
        loss_list = {}
        for epoch in range(self.config.epochs + 1):

            trn_running_loss, trn_acc = 0.0, 0.0
            train_preds = []
            ground_truths = []
            trn_correct = 0
            trn_total = 0
            for i, trainset in enumerate(self.train_loader):
                # train_mode
                print('sample {}/{}\r'.format(i, len(self.train_loader)), flush=True, end='')
                # unpack
                # train_in, train_out = trainset
                train_in = trainset["X"]
                train_out = trainset["Y"]

                ##### Optional: Remove onset channel = [0]
                ##### Run here when --input_shape 1,400,128
                if int(self.input_shape[0]) == 1:
                    # if torch.sum(train_in[:,1:,:,:]) < torch.sum(train_in[:,:1,:,:]): print("1 is onset")
                    train_in = train_in[:, 1:, :, :]  # note channel

                ################################################################

                # use GPU
                train_in = train_in.to(self.device)
                train_out = train_out.to(self.device)
                # grad init
                self.optimizer.zero_grad()

                # forward pass
                # print(train_in.shape)
                train_pred = self.model(train_in)
                # calculate acc
                _, label_pred = torch.max(train_pred.data, 1)

                # accuracy
                trn_total += train_out.size(0)
                trn_correct += (label_pred == train_out).sum().item()

                # print('-------------------------')
                # print("pred:",label_pred)
                # print("true:",train_out)
                # print()

                # f1 accuracy
                train_preds.extend(label_pred.tolist())
                ground_truths.extend(train_out.tolist())

                # calculate loss
                t_loss = self.criterion(train_pred, train_out)
                # back prop
                t_loss.backward()
                # weight update
                self.optimizer.step()

                trn_running_loss += t_loss.item()

            ###### After each epoch..... ######
            # score
            # 1. accuracy
            trn_acc = trn_correct / trn_total

            # 2. weighted f1-score
            w_f1score = f1_score(ground_truths, train_preds, average="weighted")

            precision, recall, f1, supports = precision_recall_fscore_support(
                ground_truths,
                train_preds,
                average=None,
                labels=list(range(self.label_num)),
                warn_for=tuple(),
            )
            # print learning process
            print(
                "Epoch:  %d | Train Loss: %.4f | f1-score: %.4f | accuracy: %.4f"
                % (epoch, trn_running_loss / self.num_batches, w_f1score, trn_acc)
            )
            # print("Train accuracy: %.2f" % (trn_acc))
            # print("Precision:", precision)
            # print("Recall:", recall)

            # TensorBoard
            # record running loss
            # self.writer.add_scalar(
            #     "training loss", trn_running_loss / self.num_batches, epoch
            # )
            # self.writer.add_scalar("training acc", w_f1score, epoch)

            ################## VALID ####################
            val_term = 10
            min_valloss = 10000.0

            if epoch % val_term == 0:

                if epoch == 0:

                    if mode == "basetrain":
                        avg_valloss, avg_valacc = self.valid(
                            self.valid_loader, self.model
                        )

                    elif mode == "advtrain":
                        # 1. Test + Attack Test -> adv_valid_loader_1
                        avg_valloss_1, avg_valacc_1 = self.valid(
                            self.valid_loader_1, self.model
                        )

                        # 2. Only Test
                        avg_valloss_2, avg_valacc_2 = self.valid(
                            self.valid_loader_2, self.model
                        )

                else:

                    if mode == "basetrain":
                        avg_valloss, avg_valacc = self.valid(
                            self.valid_loader, self.model
                        )

                    elif mode == "advtrain":
                        avg_valloss_1, avg_valacc_1 = self.valid(
                            self.valid_loader_1, self.model
                        )
                        avg_valloss_2, avg_valacc_2 = self.valid(
                            self.valid_loader_2, self.model
                        )

                lr = self.optimizer.param_groups[0]["lr"]

                if mode == "basetrain":
                    print(
                        """epoch: {}/{} | lr: {:.6f} |
		trn f1 score: {:.4f} | trn acc: {:.4f} | trn loss: {:.4f} |
		val loss: {:.4f} | val acc: {:.4f}""".format(
                            epoch + 1,
                            self.config.epochs,
                            lr,
                            w_f1score,
                            trn_acc,
                            trn_running_loss / self.num_batches,
                            avg_valloss,
                            avg_valacc,
                        )
                    )

                    # loss list
                    loss_list[epoch] = avg_valloss

                    # save model
                    if avg_valloss < min_valloss:
                        min_valloss = avg_valloss
                        if self.config.save_trn:
                            torch.save(
                                {
                                    "epoch": epoch,
                                    "model.state_dict": self.model.state_dict(),
                                    "loss": avg_valloss,
                                    "acc": avg_valacc,
                                },
                                os.path.join(self.save_dir, "model",
                                self.config.model_name
                                + "_valloss_"
                                + str(float(avg_valloss))
                                + "_acc_"
                                + str(float(avg_valacc))
                                + ".pt"),
                            )
                            print("model saved!")

        # print best valid f1 score
        print()
        print("######## Best F1-score #########")
        print(
            "Accuracy: {:.4f} | Loss: {:.4f}"
            "".format(self.best_valid[0], self.best_valid[1])
        )
        print("F1-score: %.4f" % (self.best_valid[2]))
        print("{:<30}{:<}".format("Precision", "Recall"))
        for p, r in zip(self.best_valid[3], self.best_valid[4]):
            print("{:<30}{:<}".format(p, r))
        print()


        # loss
        sorted_loss = sorted(loss_list.items(), key=lambda x: x[1])
        print("######## Sorted Loss List #########")
        for loss_item in sorted_loss:
            print("{}th : {}".format(loss_item[0], loss_item[1].item())) #epoch-loss

    def valid(self, valid_loader, model):
        #############################
        ######## valid function ######
        #############################

        self.valid_times += 1
        with torch.no_grad():  # important!!! for validation
            # validate mode
            self.set_mode("eval")  # model.eval()

            # average the acc of each batch
            val_loss, val_acc = 0.0, 0.0

            val_preds = []
            val_ground_truths = []

            # val_total = 0 # = len(valid_loader)
            val_correct = 0

            cur_midi_preds = []
            cur_midi_truths = []
            pred_labels = [-1] * self.label_num
            cur_true_label = -1
            cur_pred_label = -1  # majority label
            for j, valset in enumerate(valid_loader):

                # val_in, val_out = valset
                val_in = valset["X"]
                val_out = valset["Y"]

                cur_true_label = int(val_out[0])
                cur_midi_truths.append(cur_true_label)
                if cur_true_label != int(val_out[-1]):
                    print("Error!! => Diff label in same batch.")
                    return

                ##### Optional: Remove onset channel = [0]
                ##### Run here when --input_shape 1,400,128
                if int(self.input_shape[0]) == 1:
                    # if torch.sum(train_in[:,1:,:,:]) < torch.sum(train_in[:,:1,:,:]): print("1 is onset")
                    val_in = val_in[:, 1:, :, :]  # note channel
                    # print(val_in.shape)
                    # print(train_out.shape)

                ################################################################

                # to GPU
                val_in = val_in.to(self.device)
                val_out = val_out.to(self.device)

                # forward
                val_pred = self.model(val_in)  # probability
                val_softmax = torch.softmax(val_pred, dim=1)
                batch_confidence = torch.sum(val_softmax, dim=0)  # =1
                batch_confidence = torch.div(
                    batch_confidence, self.valid_seg
                )  # avg value
                # print("confidence: ")
                # print(batch_confidence)
                v_loss = self.criterion(val_pred, val_out)
                val_loss += v_loss

                # scheduler.step(v_loss)  # for reduceonplateau
                # self.scheduler.step()  # for cos

                # accuracy
                _, val_label_pred = torch.max(val_pred.data, 1)

                # val_total += val_out.size(0)
                # val_correct += (val_label_pred == val_out).sum().item()

                # changed accuracy metric
                # acc for each batch (=> one batch = one midi)
                val_label_pred = val_label_pred.tolist()

                occ = [val_label_pred.count(x) for x in range(self.label_num)]
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
                # print(dup_list)
                # print(cur_pred_label)
                # print("cur preds:", val_label_pred)
                # print("cur outs:", val_out)
                # print("cur pred label:",cur_pred_label)
                # print("cur true label:", cur_true_label)
                # print("===========================================")
                if cur_true_label == cur_pred_label:
                    val_correct += 1

                # f1 score
                val_preds.append(cur_pred_label)
                val_ground_truths.append(cur_true_label)

                # reset for next midi
                cur_midi_preds = []
                cur_midi_truths = []
                pred_labels = [-1] * self.label_num
                cur_true_label = -1
                cur_pred_label = -1  # majority label

            avg_valloss = val_loss / len(valid_loader)

            # score
            # 1. accuracy
            # print("len valid_loader:", len(valid_loader))
            # print("len val_preds:", len(val_preds))
            # print("len val_ground_truths:", len(val_ground_truths))
            print("============================================")

            val_acc = val_correct / len(valid_loader)

            # 2. weighted f1-score
            w_f1score = f1_score(val_ground_truths, val_preds, average="weighted")

            precision, recall, f1, supports = precision_recall_fscore_support(
                val_ground_truths,
                val_preds,
                average=None,
                labels=list(range(self.label_num)),
                warn_for=tuple(),
            )

            # print learning process
            print("\n######## Valid #########")
            print("Accuracy: {:.4f} | Loss: {:.4f}" "".format(val_acc, avg_valloss))
            print("F1-score: %.4f" % (w_f1score))
            print("{:<30}{:<}".format("Precision", "Recall"))
            for p, r in zip(precision, recall):
                print("{:<30}{:<}".format(p, r))
            print()

            # Valid TensorBoard
            # record running loss
            # self.writer.add_scalar("valid loss", avg_valloss, self.valid_times)
            # self.writer.add_scalar("valid acc", w_f1score, self.valid_times)

            if self.best_valid[2] < w_f1score:
                self.best_valid = [val_acc, avg_valloss, w_f1score, precision, recall]

        self.set_mode("train")  # model.train()

        return avg_valloss, w_f1score
