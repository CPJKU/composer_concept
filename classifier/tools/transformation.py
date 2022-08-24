""" From original repo (https://github.com/KimSSung/Deep-Composer-Classification/blob/master/tools/transformation.py) """

import torch
import numpy as np
import random

# from .data_loader import MIDIDataset
import random
import math


class Segmentation(object):
    """ basic segmentation
        - randomly crop (2x) 400 x 128
        - window = 600
        - overlap = vary
        - returns {X, Y, pth}
        - pth: composer#/midi#/ver#.npy
    """

    def __init__(self, is_train, seg_num, order):
        self.order = order
        self.seg_num = seg_num
        self.is_train = is_train

    def __call__(self, data):
        X, Y, pth = data["X"], data["Y"], data["pth"]
        duration = len(X[0])
        window = 500

        windows = list()
        if duration > (window * self.seg_num):
            gap = int((duration - (window * self.seg_num)) / (self.seg_num - 1))
            seg_start = 0
            while seg_start <= (duration - window):
                windows.append(seg_start)
                seg_start = seg_start + window + gap
            # print("{}:{}, {}".format("gap", gap, len(windows)))
            # print(duration)
            # print(windows)
        else:
            overlap = int(
                math.ceil(((window * self.seg_num) - duration) / (self.seg_num - 1))
            )
            seg_start = 0
            while seg_start <= (duration - window):
                windows.append(seg_start)
                seg_start = seg_start + window - overlap
            # print("{}:{}, {}".format("overlap", overlap, len(windows)))
            # print(duration)
            # print(windows)

        start = windows[self.order % len(windows)]
        if self.is_train:
            start += random.randint(0, window - 400)
        X_crop = data["X"][:, start : (start + 400), :].copy()
        # IMPORTANT: NOT A VIEW BUT A "COPY"

        return {"X": X_crop, "Y": Y, "pth": pth}


class Transpose(object):
    """ [-n, +n] semitones (default: n=6)
    segment = {X, Y, pth}
    np.roll() COPIES data to a new nd-array"""

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, segment):
        X, Y, pth = segment["X"], segment["Y"], segment["pth"]
        semitone = random.randint(-self.rng, self.rng)  # randomly select [-n ~ +n]
        new_X = list()
        for i in range(2):
            tmp = np.roll(X[i], semitone, axis=1)  # X[i].shape = (400,128)
            if semitone > 0:
                tmp[:, :semitone] = 0
            elif semitone < 0:
                tmp[:, semitone:] = 0
            new_X.append(tmp)

        new_X = np.array(new_X)  # (2,400,128)

        return {"X": new_X, "Y": Y, "pth": pth}


class TempoStretch(object):
    def __call__(self, segment):
        X, Y, pth = segment["X"], segment["Y"], segment["pth"]

        try:
            # Stretch Variable
            multiplier_rand = random.randint(0, 4)
            stretch_start = random.randint(1, 200)
            stretch_duration = random.randint(30, 100)
            stretch_end = stretch_start + stretch_duration
            tempo_list = [1.2, 1.1, 1.0, 1.3, 1.4]
            tempo_mul = tempo_list[multiplier_rand]
            segment_tree = [[{} for i in range(0, 128)] for i in range(0, X.shape[0])]
            # [[{start: [duration,veclocity]},{start: [durariont, velocity}] ....

            # Find stretch numpy, Cut Numpy for stretch

            if tempo_mul <= 1.0:
                # print('Tempo no changed')
                return {"X": X, "Y": Y, "pth": pth}

            mod_npy = X[:, stretch_start:stretch_end, :].copy()  # len(mod_npy) duration

            start = [[-1 for i in range(0, 128)] for i in range(0, X.shape[0])]
            # Masking duration
            for track in range(0, 2):
                for time in range(0, stretch_duration):
                    for note in range(0, 128):

                        # Find the start point
                        if mod_npy[track][time][note] > 0 and start[track][note] == -1:

                            if (
                                time in segment_tree[track][note].keys()
                            ):  # Error Checking
                                print("Something is wrong for Segment Tree...")
                            segment_tree[track][note][time] = [
                                1,
                                mod_npy[track][time][note],
                            ]

                            start[track][
                                note
                            ] = time  # Mark start[note] with start_time

                        # If start[note] is marked
                        elif (
                            mod_npy[track][time][note] > 0 and start[track][note] != -1
                        ):

                            segment_tree[track][note][start[track][note]][
                                0
                            ] += 1  # Add Duration 1 if it is detected

                        # End point detected
                        elif (
                            mod_npy[track][time][note] == 0 and start[track][note] != -1
                        ):

                            start[track][
                                note
                            ] = -1  # Mark one of the Long Duration Note ends

            # Debugging Duration saving

            # for track in range(0,X.shape[0]):
            #     for note in range(0,128):
            #         if  not segment_tree[track][note]:
            #             print(segment_tree[track][note])

            # Stretch all of the duration

            stretched_segment_tree = [
                [{} for i in range(0, 128)] for i in range(0, X.shape[0])
            ]

            for track in range(0, X.shape[0]):
                for note in range(0, 128):
                    if segment_tree[track][note]:
                        for key in segment_tree[track][note].keys():

                            # Modify Duration put new_segment_tree
                            new_key = int(key * tempo_mul)
                            stretched_segment_tree[track][note][new_key] = []
                            stretched_segment_tree[track][note][new_key].append(
                                int(segment_tree[track][note][key][0] * tempo_mul)
                            )

                            # If we consider velocity changed
                            stretched_segment_tree[track][note][new_key].append(
                                int(segment_tree[track][note][key][1] * (2 - tempo_mul))
                            )

            # make numpy

            mod_X = [
                [
                    [0 for i in range(0, 128)]
                    for j in range(0, int(mod_npy.shape[1] * tempo_mul) + 1)
                ]
                for k in range(0, X.shape[0])
            ]
            mod_X = np.array(mod_X)
            for track in range(0, len(mod_X)):
                for time in range(0, len(mod_X[0])):
                    for note in range(0, 128):

                        if stretched_segment_tree[track][note]:
                            for start_time in stretched_segment_tree[track][
                                note
                            ].keys():

                                if (
                                    time
                                    <= start_time
                                    + stretched_segment_tree[track][note][start_time][0]
                                ) and time >= start_time:

                                    mod_X[track][time][note] = stretched_segment_tree[
                                        track
                                    ][note][start_time][1]

            new_X = [
                [[0 for i in range(0, 128)] for j in range(0, 400)] for k in range(0, 2)
            ]

            for track in range(0, X.shape[0]):
                for time in range(0, 400):
                    for note in range(0, 128):

                        if time < stretch_start:

                            new_X[track][time][note] = X[track][time][note]

                        elif stretch_start <= time and time < stretch_start + int(
                            stretch_duration * tempo_mul
                        ):

                            new_X[track][time][note] = mod_X[track][
                                time - stretch_start
                            ][note]
                            flag_end = time
                            iterator = 0

                        else:
                            after_index = int(stretch_duration * (tempo_mul - 1)) - 1
                            new_X[track][time][note] = X[track][time - after_index][
                                note
                            ]

            new_X = np.array(new_X)

        except:
            return {"X": X, "Y": Y, "pth": pth}

        else:
            # Plot after tempo stretched version
            # nzero = new_X[1].nonzero()
            # x1 = nzero[0]
            # y1 = nzero[1]
            #
            # nzero2 = X[1].nonzero()
            # x2 = nzero2[0]
            # y2 = nzero2[1]
            #
            # plt.subplot(2,1,1)
            # plt.ylim(20, 100)
            # plt.title('Original')
            # plt.xlabel("/0.05 sec")
            # plt.ylabel("pitch")
            # plt.axvline(x=stretch_start, color='r', linestyle='--', linewidth=1)
            # plt.axvline(x=stretch_end, color='r', linestyle='--', linewidth=1)
            # plt.scatter(x=x2, y=y2, c="green", s=2)
            #
            # plt.subplot(2,1,2)
            # plt.ylim(20, 100)
            # plt.title('Modified')
            # plt.xlabel("/0.05 sec")
            # plt.ylabel("pitch")
            # plt.axvline(x=stretch_start, color='r', linestyle='--', linewidth=1)
            # plt.axvline(x=stretch_start + stretch_duration * tempo_mul, color='r', linestyle='--', linewidth=1)
            # plt.scatter(x=x1, y=y1, c="green", s=2)
            #
            # plt.show()

            return {"X": new_X, "Y": Y, "pth": pth}


class DoubleTempo(object):
    def __call__(self, segment):
        X, Y, pth = segment["X"], segment["Y"], segment["pth"]

        try:
            # Stretch Variable
            multiplier_rand = random.randint(0, 4)
            stretch_start = random.randint(3, 200)
            stretch_duration = random.randint(10, 30)
            stretch_end = stretch_start + stretch_duration

            # Find stretch numpy, Cut Numpy for stretch

            mod_npy = X[:, stretch_start:stretch_end, :].copy()  # len(mod_npy) duration
            double_npy = np.repeat(mod_npy, 2, axis=1)
            after_npy = X[:, stretch_end : (400 - stretch_duration), :].copy()

            # make numpy

            new_X = np.concatenate(
                (X[:, :stretch_start, :], double_npy, after_npy), axis=1
            )

        except:
            return {"X": X, "Y": Y, "pth": pth}

        else:
            # # Plot after tempo stretched version
            # nzero = new_X[1].nonzero()
            # x1 = nzero[0]
            # y1 = nzero[1]
            #
            # nzero2 = X[1].nonzero()
            # x2 = nzero2[0]
            # y2 = nzero2[1]
            #
            # plt.subplot(2,1,1)
            # plt.ylim(20, 100)
            # plt.title('Original')
            # plt.xlabel("/0.05 sec")
            # plt.ylabel("pitch")
            # plt.axvline(x=stretch_start, color='r', linestyle='--', linewidth=1)
            # plt.axvline(x=stretch_end, color='r', linestyle='--', linewidth=1)
            # plt.scatter(x=x2, y=y2, c="green", s=2)
            #
            # plt.subplot(2,1,2)
            # plt.ylim(20, 100)
            # plt.title('Modified')
            # plt.xlabel("/0.05 sec")
            # plt.ylabel("pitch")
            # plt.axvline(x=stretch_start, color='r', linestyle='--', linewidth=1)
            # plt.axvline(x=stretch_start + stretch_duration * 2, color='r', linestyle='--', linewidth=1)
            # plt.scatter(x=x1, y=y1, c="green", s=2)
            #
            # plt.show()

            return {"X": new_X, "Y": Y, "pth": pth}


class ToTensor(object):
    """Convert numpy.ndarray to tensor
        segment = {X, Y, pth}"""

    def __call__(self, segment):
        X, Y, pth = segment["X"], segment["Y"], segment["pth"]
        new_segment = {
            "X": torch.tensor(X, dtype=torch.float32),
            "Y": torch.tensor(Y, dtype=torch.long),
            "pth": pth,
        }
        return new_segment