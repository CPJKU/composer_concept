""" Adapted from original repo (https://github.com/KimSSung/Deep-Composer-Classification/blob/master/generator.py) """
# (partitura is used in this version instead of music21)

import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import random
import partitura

from config import maestro_root, data_root

random.seed(123)


class Generator:
    def __init__(self):
        self.chars = {
            ",": "",
            ".": "",
            '"': "",
            "'": "",
            "/": "",
            "(": "",
            ")": "",
            "{": "",
            "}": "",
            "[": "",
            "]": "",
            "!": "",
            "?": "",
            "#": "",
            "$": "",
            "%": "",
            "&": "",
            "*": "",
            " ": "",
        }
        self.song_dict = dict()
        self.name_id_map = pd.DataFrame(
            columns=["composer", "composer_id", "orig_name", "midi_id", "saved_fname"]
        )  # df to store mapped info

    def run(self):
        input_path = os.path.join(data_root, 'npy')
        data_list, composers = self.get_data_list(
            'classifier/meta/maestro-v2.0.0-reduced.csv'
        )


        for i, composer in tqdm(enumerate(composers)):
            success = 0  # count files for each composer
            track_list = list()  # for uniq track id

            print(
                "\n################################## {} ####################################\n".format(
                    composer
                )
            )

            for data in data_list:
                track_comp, orig_name, file_name = data[0], data[1], data[2]
                # file_name = os.path.join(maestro_root, file_name)
                if track_comp is composer:
                    try:
                        mid = partitura.load_performance_midi(
                            os.path.join(maestro_root, data[2])
                        )
                        segment = self.generate_segment(mid)
                    except:
                        print("ERROR: failed to open {}\t".format(file_name))
                    else:
                        # assign uniq id to midi
                        version = self.fetch_version(orig_name)
                        track_id = self.fetch_id(track_list, orig_name)

                        fsave_pth = os.path.join(
                            input_path, "composer" + str(i) + "/midi" + str(track_id)
                        )
                        self.save_input(segment, fsave_pth, version)  # TODO: enable
                        self.name_id_map = self.name_id_map.append(
                            {
                                "composer": composer,
                                "composer_id": i,
                                "orig_name": orig_name,
                                "midi_id": track_id,
                                "saved_fname": file_name,
                            },
                            ignore_index=True,
                        )

                        # print result
                        success += 1
                        # print(
                        #     "{} success: {} => {} => midi{}_ver{}".format(
                        #         success, file_name, orig_name, track_id, version
                        #     )
                        # )

        # save mapped list
        self.name_id_map.to_csv(
            os.path.join(input_path, "name_id_map.csv"), sep=","
        ) 
        return

    def get_data_list(self, fdir):  # return preprocessed list of paths
        # data = pd.read_csv(fdir, encoding="euc-kr")  # cleaned csv
        data = pd.read_csv(fdir)  # cleaned csv
        data = data.drop(
            ["split", "year", "audio_filename", "duration"], axis=1
        )  # drop unnecessary columns
        data_list = list(
            zip(
                data["canonical_composer"],
                data["canonical_title"],
                data["midi_filename"],
            )
        )
        composers = data["canonical_composer"].unique()

        return data_list, composers

    def generate_segment(self, mid, frame_length=0.05, piano_range=True):
        """Generate a 3d pianoroll from a partitura performedPart with a specific frame length (in seconds).
        The first channel contains only onsets, while the second contains full length notes and velocity."""
        mid.sustain_pedal_threshold = 127 # don't take pedal into consideration
        matrix1 = partitura.utils.compute_pianoroll(
            mid,
            time_unit="sec",
            time_div=int(1 / frame_length),
            onset_only=True,
            piano_range=piano_range,
        ).toarray()
        matrix1[matrix1 > 0] = 1  # we don't want velocity here
        matrix2 = partitura.utils.compute_pianoroll(
            mid,
            time_unit="sec",
            time_div=int(1 / frame_length),
            onset_only=False,
            piano_range=piano_range,
        ).toarray()
        pr3d = np.stack([matrix1, matrix2])
        pr3d = np.transpose(pr3d, (0, 2, 1))

        return pr3d

    def fetch_version(self, track):
        track = track.lower()  # case-insensitive comparison
        track = track.translate(str.maketrans(self.chars))  # remove symbols
        if track in self.song_dict:
            self.song_dict[track] = self.song_dict[track] + 1  # update
        else:
            self.song_dict.update({track: 0})

        return self.song_dict[track]

    def fetch_id(self, lookup, name):
        name = name.lower()  # case-insensitive comparison
        name = name.translate(str.maketrans(self.chars))  # remove symbols
        if name not in lookup:
            lookup.append(name)

        return lookup.index(name)

    def save_input(self, matrix, save_pth, vn):
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)

        np.save(save_pth + "/ver" + str(vn), matrix)  # save as .npy


if __name__ == "__main__":
    gen = Generator()
    gen.run()
