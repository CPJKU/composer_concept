""" Taken and readapted to MIDI data from original repo (https://github.com/zhangrh93/InvertibleCE) """

import re
import sys
from config import results_root, asap_root

sys.path.append(".")

from unsupervised.uns_utils import *
import unsupervised.ModelWrapper as ModelWrapper
import unsupervised.ChannelReducer as ChannelReducer

import os
from pathlib import Path
import pickle

import pydotplus

import numpy as np
import matplotlib.pyplot as plt
import time
import json


FONT_SIZE = 30
CALC_LIMIT = 1e9
TRAIN_LIMIT = 50
USE_TRAINED_REDUCER = False
ESTIMATE_NUM = -1


class Explainer:
    def __init__(
        self,
        title="",
        layer_name="",
        class_names=None,
        utils=None,
        keep_feature_images=True,
        useMean=True,
        reducer_type="NMF",
        n_components=10,
        featuretopk=20,
        featureimgtopk=5,
        epsilon=1e-4,
        nmf_initialization="nndsvd",
        dimension=None,
        iter_max=1000,
    ):
        self.title = title
        self.layer_name = layer_name
        self.class_names = class_names
        self.class_nos = len(class_names) if class_names is not None else 0
        self.X_features = None
        self.loaders_sizes = None

        self.keep_feature_images = keep_feature_images
        self.useMean = useMean
        self.reducer_type = reducer_type
        self.featuretopk = featuretopk
        self.featureimgtopk = featureimgtopk  # number of images for a feature
        self.n_components = n_components
        self.epsilon = epsilon
        self.nmf_initialization = nmf_initialization
        self.dimension = dimension
        self.iter_max = iter_max

        self.utils = utils

        self.reducer = None
        self.feature_distribution = None

        self.feature_base = []
        self.features = {}
        self.features_contrast = {}

        self.exp_location = Path(results_root)

        self.font = FONT_SIZE

    def load(self):
        title = self.title
        with open(self.exp_location / title / (title + ".pickle"), "rb") as f:
            tdict = pickle.load(f)
            self.__dict__.update(tdict)

    def save(self):
        if not os.path.exists(self.exp_location):
            os.mkdir(self.exp_location)
        title = self.title
        if not os.path.exists(self.exp_location / title):
            os.mkdir(self.exp_location / title)
        with open(self.exp_location / title / (title + ".pickle"), "wb") as f:
            pickle.dump(self.__dict__, f)

    def train_model(self, model, loaders):
        self._train_reducer(model, loaders)
        self._estimate_weight(model, loaders)
        #save the dimension of the available dataset
        self.loaders_sizes = [len(dl.dataset) for dl in loaders]

    def _train_reducer(self, model, loaders):

        print("Training reducer:")

        if self.reducer is None:
            if not self.reducer_type in ChannelReducer.ALGORITHM_NAMES:
                print("reducer not exist")
                return

            if ChannelReducer.ALGORITHM_NAMES[self.reducer_type] == "decomposition":
                self.reducer = ChannelReducer.ChannelDecompositionReducer(
                    n_components=self.n_components,
                    reduction_alg=self.reducer_type,
                    max_iter=self.iter_max,
                    # nndsvda better when sparsity is not desired
                    # nndsvd better fpr sparsity
                    init=self.nmf_initialization,
                )
            elif (
                ChannelReducer.ALGORITHM_NAMES[self.reducer_type] == "3d_decomposition"
            ):
                self.reducer = ChannelReducer.ChannelTensorDecompositionReducer(
                    dimension=self.dimension,
                    rank=self.n_components,
                    iter_max=self.iter_max,
                )
            else:
                self.reducer = ChannelReducer.ChannelClusterReducer(
                    n_components=self.n_components, reduction_alg=self.reducer_type,
                )

        X_features = []
        for loader in loaders:
            # save all activations of the output of the target layer for the input composer dataset.
            X_features.append(model.get_feature(loader, self.layer_name))
        # save X_features for later usage, to avoid having to recompute them
        self.X_features = X_features
        print("1/5 Feature maps gathered.")

        if not self.reducer._is_fit:
            # concatenate the activations corresponding to the different composers
            nX_feature = np.concatenate(X_features)
            total = np.product(nX_feature.shape)
            l = nX_feature.shape[0]
            if total > CALC_LIMIT:
                p = CALC_LIMIT / total
                print("dataset too big, train with {:.2f} instances".format(p))
                idx = np.random.choice(l, int(l * p), replace=False)
                nX_feature = nX_feature[idx]

            print("loading complete, with size of {}".format(nX_feature.shape))
            start_time = time.time()
            # the NMF is produced from the matrix of two different composers at the same time. Then we'll know if some concepts are in common or not
            self.reducer.fit_transform(nX_feature)
            # nX contains W in the NMF result (W,H)
            # import matplotlib.pyplot as plt

            print("2/5 Reducer trained, spent {} s.".format(time.time() - start_time))

        # get the NMF CAVs (i.e. the "frequent" activation patterns produced by NMF, i.e. the H in NMF result (W,H))
        if isinstance(self.reducer, ChannelReducer.ChannelTensorDecompositionReducer):
            self.cavs = self.reducer.precomputed_tensors.factors[0].T
        else:
            self.cavs = self.reducer._reducer.components_
        # compute some statistics (to remove)
        # nX = nX.mean(axis=(1, 2))
        # self.feature_distribution = {
        #     "overall": [
        #         (nX[:, i].mean(), nX[:, i].std(), nX[:, i].min(), nX[:, i].max())
        #         for i in range(self.n_components)
        #     ]
        # }

        reX = []
        # self.feature_distribution["classes"] = []
        for X_feature in X_features:
            # transform the pieces for each composer, according to the factorization produced before
            # the result is in shape: n x h x w x c'
            if self.reducer_type == "NMF":
                t_feature = self.reducer.transform(X_feature)
                # inverse NMF transform the vector n x h x w x c' to the original A matrix n x h x w x c
                reX.append(self.reducer.inverse_transform(t_feature))
            else:
                t_feature, indices = self.reducer.transform(X_feature)
                reX.append(self.reducer.inverse_transform(t_feature, indices))

        err = []
        prec = []
        for i in range(len(self.class_names)):
            # find the predicted class for the original activation layer
            res_true = model.feature_predict(
                X_features[i], layer_name=self.layer_name
            ).argmax(axis=1)
            # find the predicted class for the reconstructed activation layer
            res_recon = model.feature_predict(
                reX[i], layer_name=self.layer_name
            ).argmax(axis=1)
            # check when it is true and normalize for the number of pieces considered
            err.append(np.count_nonzero(res_true == res_recon) / len(X_features[i]))
            # append also how good is the original classifier
            prec.append(np.count_nonzero(res_true == i) / len(X_features[i]))

        self.reducer_err = np.array(err)
        self.original_precision = np.array(prec)
        if type(self.reducer_err) is not np.ndarray:
            self.reducer_err = np.array([self.reducer_err])

        print("3/5 Error estimated, fidelity: {}.".format(self.reducer_err))
        print(f"Original precision: {self.original_precision}")

        return (self.reducer_err,)

    def _estimate_weight(self, model, loaders):
        if self.reducer is None:
            return

        nX_feature = np.concatenate(self.X_features)

        print("4/5 Weight estimator initialized.")

        self.test_weight = []
        for cav in self.cavs:
            # computing the CAV score for 2 target class at the same time
            # let's extract the logits for the 2 classes. We perturb the input with a very small variation in the direction of the CAV
            res1 = model.feature_predict(
                nX_feature - self.epsilon * cav, layer_name=self.layer_name
            )
            res2 = model.feature_predict(
                nX_feature + self.epsilon * cav, layer_name=self.layer_name
            )

            res_dif = res2 - res1
            # average across input pieces (remember, they considered only 10 pieces per target)
            dif = res_dif.mean(axis=0) / (
                2 * self.epsilon
            )  # this could have been done earlier like in the equation, but I guess it's the same if we switch order
            if type(dif) is not np.ndarray:
                dif = np.array([dif])
            self.test_weight.append(dif)
            # at this point, test_weights contains a matrix of shape (n_components,n_target_classes)

        self.test_weight = np.array(self.test_weight)
        print("5/5 Weight estimated.")
        # print the weights
        for i, weight in enumerate(self.test_weight):
            print(f"Weights for CAV{i} (for target classes) : {weight} ")
        # find contrastive CAVs and print them
        find_contrastive_cavs(self.test_weight)

    def generate_features(self, model, loaders):
        self._visualize_features(model, loaders)
        self._save_features()
        self._save_features(contrast=True)
        if self.keep_feature_images == False:
            self.features = {}
        return

    def _feature_filter(self, featureMaps, threshold=None):
        """This function has 2 objectives: 
        if self.useMean == True it average the feature map of shape n x h x w x c and produce n x c representation
        if threshold != None ... to complete
        """
        if self.useMean:
            res = featureMaps.mean(axis=(1, 2))
        else:
            res = featureMaps.max(axis=(1, 2))
        if threshold is not None:
            res = -abs(res - threshold)
        return res

    def _update_feature_dict(self, x, h, fm, nx, nh, nfm, threshold=None, keep="highest"):
        """Concatenate x with nx and h with nh, order them 
        wrt the average for each piece and take the 5 with highest (or lowest) value.
        Or just return nx, nh if x (and h) is None"""

        if type(x) == type(None):
            return nx, nh, nfm
        else:
            x = np.concatenate([x, nx])
            h = np.concatenate([h, nh])
            fm = np.concatenate([fm, nfm])

            if keep == "highest":
                nidx = fm.argsort()[::-1][
                    :self.featureimgtopk
                ]
            elif keep == "lowest":
                nidx = fm.argsort()[
                    : self.featureimgtopk
                ]
            else:
                raise ValueError("keep can be only 'highest' or 'lowest'")
            x = x[nidx, ...]
            h = h[nidx, ...]
            fm = fm[nidx]
            return x, h, fm

    def _visualize_features(self, model, loaders, featureIdx=None, inter_dict=None):
        # this seems to just clip featuretopk at 20, if ever the number of components (i.e. number of features, i.e., number of cavs) is higher
        featuretopk = min(self.featuretopk, len(self.cavs))

        imgTopk = (
            self.featureimgtopk
        )  # the number of images that maximally activate each concept to use as example
        if featureIdx is None:
            featureIdx = []
            tidx = []
            w = self.test_weight
            for i, _ in enumerate(self.class_names):
                # take the weights for all CAVs that refer to only a specific target class
                tw = w[:, i]
                # concatenate in a single list, the index of the weights in decreasing order.
                # for 3 features and 2 target classes you will have first the 3 indices of the 1st target class, then the other 3
                tidx += tw.argsort()[::-1][:featuretopk].tolist()
            # why bothering argsorting and reversing if we are loosing the order with the "set" function?
            # I guess this initial part is doing something only if you have more components than 20, so you are saving only the indices of the bigger 20s
            featureIdx += list(set(tidx))

        # this next part stop the computation for the already computed features
        # the first time, nowIdx will be empty and featureIdx will stay the same
        nowIdx = set(self.features.keys())
        featureIdx = list(set(featureIdx) - nowIdx)
        featureIdx.sort()
        if len(featureIdx) == 0:
            print("All feature gathered")
            return

        print("visualizing features :")
        print(featureIdx)
        # initialize the features
        features = {}
        features_contrast = {}
        for No in featureIdx:
            features[No] = [None, None, None]
            features_contrast[No] = [None, None, None]

        print("loading training data")
        X_features = self.X_features
        composers_pieces = []
        for loader in loaders:
            # save all activations of the output of the target layer for the input composer dataset.
            feat, data = model.get_feature(loader, self.layer_name, return_data=True)
            # X_features.append(feat)
            composers_pieces.append(np.stack(data))

        assert len(X_features) == len(self.class_names)
        # iterate over dataloaders (one dataloader for each target class)
        for i, (X_feature, comp_pieces) in enumerate(zip(X_features, composers_pieces)):
            # produce the W from the (W,H) NMF factorization already trained
            if self.reducer_type == "NMF":
                featureMaps = self.reducer.transform(X_feature)
            else:
                featureMaps, _ = self.reducer.transform(X_feature)
            # average W to pass from from (n x h x w x c') to (n x c')
            featureMaps_avg = self._feature_filter(featureMaps)

            # iterate for the different CAVs
            for No in featureIdx:
                # None, None at first call, used to concatenate features for one CAV in different minibatches and different dataloaders
                samples_top, heatmap_top, fm_top = features[No]
                samples_contrast, heatmap_contrast, fm_contrast = features_contrast[No]
                # take the indices of the last 5 vectors (i.e. pieces that activates the CAVs most)
                idx_most = featureMaps_avg[:, No].argsort()[::-1][:imgTopk]
                # take the indices of the first 5 vectors (i.e. pieces that activates the CAVs less)
                idx_less = featureMaps_avg[:, No].argsort()[:imgTopk]
                # take only the top/bottom 5 pieces from the full W matrix
                nheatmap_most = featureMaps[idx_most, :, :, No]
                nheatmap_less = featureMaps[idx_less, :, :, No]
                # take the top/bottom 5 pieces from the input pianorolls matrix
                nsamples_most = comp_pieces[idx_most, ...]
                nsamples_less = comp_pieces[idx_less, ...]

                # find top/bottom 5 across all composer classes
                samples_top, heatmap_top, fm_top = self._update_feature_dict(
                    samples_top, heatmap_top, fm_top,  nsamples_most, nheatmap_most, featureMaps_avg[idx_most,No]
                )
                samples_contrast, heatmap_contrast, fm_contrast = self._update_feature_dict(
                    samples_contrast,
                    heatmap_contrast,
                    fm_contrast,
                    nsamples_less,
                    nheatmap_less,
                    featureMaps_avg[idx_less,No],
                    keep="lowest",
                )
                features[No] = [samples_top, heatmap_top, fm_top]
                features_contrast[No] = [samples_contrast, heatmap_contrast, fm_contrast]
                # features[No].append([nsamples_most, nheatmap_most, featureMaps_avg[idx_most,No] ])
                # features_contrast[No].append([nsamples_less, nheatmap_less, featureMaps_avg[idx_less,No]])

            print(
                "Done with class: {}, {}/{}".format(
                    self.class_names[i], i + 1, len(loaders)
                )
            )


        # create repeat prototypes in case lack of samples
        # TODO : check what is that. It does not seems to enter here
        for no, (x, h, avg) in features.items():
            idx = h.mean(axis=(1, 2)).argmax()
            for i in range(h.shape[0]):
                if h[i].max() == 0:
                    print(
                        "WARNING: creating repeat prototype. Probably the explainer is not good enough."
                    )
                    x[i] = x[idx]
                    h[i] = h[idx]

        self.features.update(features)
        self.features_contrast.update(features_contrast)
        self.save()
        return inter_dict


    def _save_features(
        self, threshold=0.5, background=0.2, smooth=True, contrast=False
    ):
        if not contrast:
            feature_path = self.exp_location / self.title / "feature_imgs"
        else:
            feature_path = self.exp_location / self.title / "feature_contrast_imgs"

        if not os.path.exists(feature_path):
            os.mkdir(feature_path)

        if not contrast:
            features = self.features
        else:
            features = self.features_contrast

        for idx in features.keys():

            
            x, h, avg = features[idx]
            minmax = False
            if self.reducer_type == "PCA":
                minmax = True
            x, h = self.utils.img_filter(
                x,
                h,
                threshold=1,
                background=0,
                smooth=smooth,
                minmax=minmax,
                skip_norm=True,
                skip_mask=True,
            )
            title = f"Explanation for CAV{idx} - {self.class_names[0]}: {self.test_weight[idx][0]} , {self.class_names[1]}: {self.test_weight[idx][1]}"
            if contrast:
                title = "Contrastive " + title
            plotly_fig = self.utils.plotly_plot(x, h, avg, title)
            plotly_fig.write_html(feature_path / (str(idx) + "plotly.html"))
            # save reducer error plot
            if self.reducer_type == "NTD":
                error_plot_path = self.exp_location / self.title / "NTD_error.png"
                plt.plot(self.reducer.reducer_conv)
                plt.savefig(error_plot_path)
                plt.close()

    def _sonify_features(
        self,
        threshold=0.5,
        background=0.01,
        smooth=True,
        unfiltered_midi=False,
        contrast=False,
    ):
        if not contrast:
            feature_path = self.exp_location / self.title / "feature_midis"
            features = self.features
        else:
            feature_path = self.exp_location / self.title / "feature_contrast_midis"
            features = self.features_contrast
        # utils = self.utils

        if not os.path.exists(feature_path):
            os.mkdir(feature_path)

        # iterate over features
        for idx in features.keys():
            x, h, avg = features[idx]
            # x contains the MIDI for 6 different MIDI files
            minmax = False
            if self.reducer_type == "PCA":
                minmax = True
            # filter out notes that are not considered
            x_filt, h = self.utils.midi_filter(
                x,
                h,
                threshold=threshold,
                background=background,
                smooth=smooth,
                minmax=minmax,
            )

            # iterate over MIDI files for a specific feature
            for i in range(x_filt.shape[0]):
                # save midi for x[i]
                midifile_filt_path = Path(feature_path, f"feature{idx}-{i}_filt.mid")
                try:
                    if unfiltered_midi:
                        midifile_path = Path(feature_path, f"feature{idx}-{i}.mid")
                        self.utils.pianoroll2midi(
                            np.transpose(x[i], (0, 2, 1)), midifile_path, channels=1
                        )
                except ValueError:
                    print(f"No midi file generated for feature{idx}-{i}")

    def global_explanations(self):
        title = self.title
        suggested_CAVs = find_contrastive_cavs(self.test_weight, print_sug=False)
        reducer_conv = (
            self.reducer.reducer_conv[-1]
            if self.reducer_type == "NTD"
            else self.reducer.reducer_conv
        )

        print("Generate explanations with fullset condition")
        out_json = {
            "title": self.title,
            "reducer": self.reducer_type,
            "dimension": self.dimension,
            "rank": self.n_components,
            "fidelity": self.reducer_err.tolist(),
            "original_precision": self.original_precision.tolist(),
            "fidelity_avg": float(np.mean(self.reducer_err)),
            "classes": self.class_names,
            "concepts_sensitivity": {
                f"CAV{i}": cav.tolist() for i, cav in enumerate(self.test_weight)
            },
            "suggested_CAVs": suggested_CAVs,
            "reducer_conv": float(reducer_conv),
            "dataset_size" : int(len(np.concatenate(self.X_features))),
            "loaders_size" : list(self.loaders_sizes),
        }
        out_path = self.exp_location / title / "summary.json"
        with open(out_path, "w") as outfile:
            json.dump(out_json, outfile)
