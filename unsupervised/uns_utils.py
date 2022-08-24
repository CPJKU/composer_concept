""" Taken and readapted to MIDI data from the original repo (https://github.com/zhangrh93/InvertibleCE) """
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.transform import resize
import partitura
import tensorly as tl
import plotly
import plotly.express as px
import plotly.graph_objects as go
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import json

from config import asap_root

mean = [103.939, 116.779, 123.68]
SIZE = [224, 224]
EPSILON = 1e-8


class img_utils:
    def __init__(
        self,
        img_size=(224, 224),
        nchannels=3,
        img_format="channels_last",
        mode=None,
        std=None,
        mean=None,
    ):
        self.img_format = img_format
        self.nchannels = nchannels
        self.fsize = list(img_size)
        self.img_size = self.fsize + [self.nchannels]
        # if img_format == 'channels_first':
        #    self.img_size = [self.nchannels] + self.fsize
        # else:
        #    self.img_size = self.fsize + [self.nchannels]

        self.std = std
        self.mean = mean
        self.mode = mode

    def deprocessing(self, x):
        mode = self.mode
        x = np.array(x)
        X = x.copy()

        if self.img_format == "channels_first":
            if X.ndim == 3:
                X = np.transpose(X, (1, 2, 0))
            else:
                X = np.transpose(X, (0, 2, 3, 1))

        if mode is None:
            return X

        if mode == "tf":
            X += 1
            X *= 127.5
            return X

        if mode == "torch":
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        if mode == "caffe":
            mean = [103.939, 116.779, 123.68]
            std = None

        if mode == "norm":
            mean = self.mean
            std = self.std

        if std is not None:
            X[..., 0] *= std[0]
            X[..., 1] *= std[1]
            X[..., 2] *= std[2]
        X[..., 0] += mean[0]
        X[..., 1] += mean[1]
        X[..., 2] += mean[2]

        if mode == "caffe":
            # 'RGB'->'BGR'
            X = X[..., ::-1]

        if mode == "torch":
            X *= 255
        return X

    def resize_img(self, array, smooth=False):
        fsize = self.fsize
        size = array.shape
        if smooth:
            tsize = list(size)
            tsize[1] = fsize[0]
            tsize[2] = fsize[1]
            res = resize(array, tsize, order=1, mode="reflect", anti_aliasing=False)
        else:
            res = []
            for i in range(size[0]):
                temp = array[i]
                temp = np.repeat(temp, fsize[0] // size[1], axis=0)
                temp = np.repeat(temp, fsize[1] // size[2], axis=1)
                res.append(temp)
            res = np.array(res)
        return res

    def flatten(self, array):
        size = array.shape
        return array.reshape(-1, size[-1])

    def show_img(
        self,
        X,
        nrows=1,
        ncols=1,
        heatmaps=None,
        useColorBar=True,
        deprocessing=True,
        save_path=None,
    ):
        X = np.array(X)
        if not heatmaps is None:
            heatmaps = np.array(heatmaps)
        if len(X.shape) < 4:
            print("Dim should be 4")
            return

        X = np.array(X)
        if deprocessing:
            X = self.deprocessing(X)

        if (not X.min() == 0) or X.max() > 1:
            X = X - X.min()
            X = X / X.max()

        if X.shape[0] == 1:
            X = np.squeeze(X)
            X = np.expand_dims(X, axis=0)
        else:
            X = np.squeeze(X)

        if self.nchannels == 1:
            cmap = "Greys"
        else:
            cmap = "viridis"

        l = nrows * ncols
        plt.figure(figsize=(5 * ncols, 5 * nrows))
        for i in range(l):
            plt.subplot(nrows, ncols, i + 1)
            plt.axis("off")
            img = X[i]
            img = np.clip(img, 0, 1)
            plt.imshow(img, cmap=cmap)
            if not heatmaps is None:
                if not heatmaps[i] is None:
                    heapmap = heatmaps[i]
                    plt.imshow(heapmap, cmap="jet", alpha=0.5, interpolation="bilinear")
                    if useColorBar:
                        plt.colorbar()

        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

    def img_filter(
        self,
        x,
        h,
        threshold=0.5,
        background=0.1,
        smooth=False,
        minmax=False,
        skip_norm=False,
        skip_mask=False,
    ):
        x = x.copy()
        h = h.copy()

        if minmax:
            h = h - h.min()

        if not skip_norm:
            h = h * (h > 0)
            for i in range(h.shape[0]):
                h[i] = h[i] / (h[i].max() + EPSILON)

            h = (h - threshold) * (1 / (1 - threshold))

        h = self.resize_img(h, smooth=smooth)

        if not skip_mask:
            h = (h > 0).astype("float") * (1 - background) + background
            h_mask = np.repeat(h, self.nchannels).reshape(list(h.shape) + [-1])
            if self.img_format == "channels_first":
                h_mask = np.transpose(h_mask, (0, 3, 1, 2))
            x = x * h_mask  # Commented to show for MIDI

        h = h - h.min()
        h = h / (h.max() + EPSILON)

        return x, h

    def midi_filter(
        self, x, h, threshold=0.5, background=0.0, smooth=False, minmax=False
    ):
        x = x.copy()
        h = h.copy()

        if minmax:
            h = h - h.min()

        h = h * (h > 0)
        for i in range(h.shape[0]):
            h[i] = h[i] / (h[i].max() + EPSILON)

        h = (h - threshold) * (1 / (1 - threshold))

        # h = h * (h>0)

        h = self.resize_img(h, smooth=smooth)
        h = (h > 0).astype("float") * (1 - background) + background
        h_mask = np.repeat(h, self.nchannels).reshape(list(h.shape) + [-1])
        if self.img_format == "channels_first":
            h_mask = np.transpose(h_mask, (0, 3, 1, 2))
        x = x * h_mask  # Commented to show for MIDI

        h = h - h.min()
        h = h / (h.max() + EPSILON)

        return x, h

    # def pianoroll2midi(
    #     self, pianoroll3d, out_path, samples_per_second=20, piano_range=True
    # ):
    #     """Generate a midi file from a pianoroll.
    #     The expected pianoroll shape is (2,128,x)"""
    #     if pianoroll3d.max() == 0:
    #         raise ValueError(
    #             "There should be at least one note played in the pianoroll"
    #         )

    #     note_array = partitura.utils.pianoroll_to_notearray(
    #         pianoroll3d[1, :, :], time_div=samples_per_second, time_unit="sec"
    #     )
    #     performed_part = partitura.performance.PerformedPart.from_note_array(
    #         note_array, id=None, part_name=None
    #     )
    #     partitura.io.exportmidi.save_performance_midi(performed_part, out_path)

    def pianoroll2midi(self, pianoroll3d, out_path, samples_per_second=20, channels=2):
        """Generate a midi file from a pianoroll. 
        The expected pianoroll shape is (2,128,x) or (2,88,x), or """
        if pianoroll3d.max() == 0:
            raise ValueError(
                "There should be at least one note played in the pianoroll"
            )
        if pianoroll3d.shape[0] != channels or pianoroll3d.shape[1] not in (128, 88):
            raise ValueError("Shape is expected to be ['channels', 128, x]")

        # enlarge to 128 midi pitches if there are only 88.
        # The inverse operation of slicing it with [:,21:109,: ]
        if pianoroll3d.shape[1] == 88:
            pianoroll3d = np.pad(pianoroll3d, ((0, 0), (21, 19), (0, 0)), "constant")

        if channels == 2:  # take only the channel with note duration
            note_array = partitura.utils.pianoroll_to_notearray(
                pianoroll3d[1, :, :], time_div=samples_per_second, time_unit="sec"
            )
        elif channels == 1:  #  take the only channel
            note_array = partitura.utils.pianoroll_to_notearray(
                pianoroll3d[0, :, :], time_div=samples_per_second, time_unit="sec"
            )
        performed_part = partitura.performance.PerformedPart.from_note_array(
            note_array, id=None, part_name=None
        )
        partitura.io.exportmidi.save_performance_midi(performed_part, out_path)

    def contour_img(self, x, h, dpi=400):
        image_size_multiplier = 2  # added by francesco. Set 1 for the original image
        dpi = float(dpi)
        size = x.shape
        if x.max() > 1:
            x = x / x.max()
        fig = plt.figure(
            figsize=(
                image_size_multiplier * size[1] / dpi,
                image_size_multiplier * size[0] / dpi,
            ),
            dpi=image_size_multiplier * dpi * SIZE[0] / size[0],
        )
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        xa = np.linspace(0, size[1] - 1, size[1])
        ya = np.linspace(0, size[0] - 1, size[0])
        X, Y = np.meshgrid(xa, ya)
        if x.shape[-1] == 1:
            x = np.squeeze(x)
            ax.imshow(x, cmap="Greys")
        elif x.shape[-1] == 2:
            # x = np.concatenate([x, np.zeros((x.shape[0], x.shape[1], 1))], axis=2)
            x = np.squeeze(x[:, :, 1])  # display notes in full lenght
            ax.imshow(x)
        else:
            ax.imshow(x)
        ax.contour(X, Y, h, colors="r", linewidths=0.2)

        return fig

    def res_ana(
        self, model, classesLoader, classNos, reducer, layer_name="conv5_block3_out"
    ):

        w, b = model.model.layers[-1].get_weights()
        V_ = reducer._reducer.components_
        w_ = np.dot(V_, w)
        ana = []

        for No in classNos:
            target = No
            tX, ty = classesLoader.load_val([No])
            tX = np.concatenate(tX)
            ty = np.concatenate(ty).astype(int)
            X = model.get_feature(tX, layer_name=layer_name).mean(axis=(1, 2))
            S = reducer.transform(X)
            U = X - np.dot(S, V_)

            C1 = np.dot(X, w[:, target])
            C2 = np.dot(S, w_[:, target]) + np.dot(U, w[:, target])

            C = np.dot(S, w_[:, target])
            res = np.dot(U, w[:, target])

            ana.append(np.array([C1, C2, C, res]))
        return [np.array(ana), reducer]


    def plotly_plot(self, x, h, avg, title):
        number_of_steps = 10
        number_of_figures = 5

        # Create figure
        fig = plotly.subplots.make_subplots(number_of_figures, 1, subplot_titles=[f"average concept presence: {a:.2f}" for a in avg],vertical_spacing = 0.03)
        for i in range(number_of_figures):
            fig.add_trace(
                go.Heatmap(z=x[i, 0, :, :].T / x.max(), colorscale="hot_r"), i + 1, 1
            )
            # Add traces, one for each slider step
            for step in np.linspace(0, 1, number_of_steps):
                fig.add_trace(
                    go.Contour(
                        z=h[i].T,
                        showscale=False,
                        line_color="black",
                        contours=dict(value=step, type="constraint", operation=">="),
                        line_width=2,
                    ),
                    i + 1,
                    1,
                )

        fig.update_traces(showscale=False)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(height=1200)

        # Make 1th trace visible
        # fig.data[1].visible = True

        # Create and add slider
        steps = []
        for i in range(number_of_steps):
            step = dict(
                method="update",
                label=f"{int(i)*10}%",
                args=[
                    {"visible": [False] * len(fig.data)},
                    {"title": title},
                ],  # layout attribute
            )
            for fi in range(number_of_figures):
                # print(i+(fi*(number_of_steps+1)-1))
                step["args"][0]["visible"][
                    i + (fi * (number_of_steps + 1))
                ] = True  # Toggle i'th trace to "visible"
                step["args"][0]["visible"][
                    0 + (fi * (number_of_steps + 1))
                ] = True  # Toggle initial figure to be alway visible
            steps.append(step)

        sliders = [
            dict(
                active=0,
                currentvalue={"prefix": "threshold: "},
                pad={"t": 0},
                steps=steps,
            )
        ]

        fig.update_layout(sliders=sliders)

        return fig

        # fig.show()
        # fig.write_html("test_plot.html")


def find_contrastive_cavs(list_of_concept_sensitivity, print_sug=True):
    difference = [np.abs(sen[1] - sen[0]) for sen in list_of_concept_sensitivity]
    sort_diff_inds = np.argsort(difference)[::-1]
    # remove not contrastive CAVs
    filtered_diff_inds = [
        ind
        for ind in sort_diff_inds.tolist()
        if list_of_concept_sensitivity[ind][0] * list_of_concept_sensitivity[ind][1] < 0
    ]
    if print_sug:
        for ind in filtered_diff_inds:
            print(f"Promising CAV {ind}")
    return filtered_diff_inds


class AsapPianoRollDataset(Dataset):
    """Asap is used instead of Maestro to have more control over the multiple performances of the same pieces"""
    def __init__(
        self,
        composers_idx = [],
        seg_num=2,
        asap_basepath = asap_root,
        frame_length = 0.05,
        piano_range = True,
        window_len = 400
    ):
        assert len(composers_idx)==1
        self.composers_idx = composers_idx
        self.composer_names = [TARGET_COMPOSERS[c] for c in composers_idx]
        self.seg_num = seg_num  # seg num per song

        #get a list of performances such as there are not 2 performances of the same piece
        df = pd.read_csv(Path(asap_basepath,"metadata.csv"))
        df = df.drop_duplicates(subset=["title","composer"])
        df = df[df['composer'].isin(self.composer_names)]
        self.pianorolls = []

        for midi_path, comp in zip(df["midi_performance"].tolist(), df['composer'].tolist() ):
            part = partitura.load_performance_midi(Path(asap_basepath,midi_path))
            part.sustain_pedal_threshold = 127  # discard pedal information
            pr = partitura.utils.compute_pianoroll(
                part,
                time_unit="sec",
                time_div=int(1 / frame_length),
                onset_only=False,
                piano_range=piano_range,
            ).toarray().T
            pr = np.expand_dims(pr,0)
            all_start_idxs = [s*window_len for s in range((pr.shape[1]-1)//window_len)]
            random_start = np.random.choice(all_start_idxs, size=seg_num, replace=False)
            self.pianorolls.extend([pr[:,rs:rs+window_len,:] for rs in random_start])

        
    def __len__(self):
        return len(self.pianorolls)

    def __getitem__(self, idx):
        X = self.pianorolls[idx]

        return [X]


TARGET_COMPOSERS = [
    "Scriabin",
    "Debussy",
    "Scarlatti",
    "Liszt",
    "Schubert",
    "Chopin",
    "Bach",
    "Brahms",
    "Haydn",
    "Beethoven",
    "Schumann",
    "Rachmaninoff",
    "Mozart",
]