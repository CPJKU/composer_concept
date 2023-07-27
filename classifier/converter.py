""" Taken from original repo (https://github.com/KimSSung/Deep-Composer-Classification/blob/master/converter.py) """
import py_midicsv
import os
import numpy as np
import pandas as pd
from config import data_root, results_root, maestro_root


class Converter:
    def __init__(self):
        self.atype = ""  # default
        self.csv_printable = False

        ### Set File directory
        # Get the Header and other data at original Midi Data
        self.npy_root_path = os.path.join(data_root, 'npy')
        self.npy_path_list = []  # String list object /data/inputs/composer#/...
        self.midi_header_path_list = (
            []
        )  # Save matched version composer#/midi# -> "/data/3 Etudes, Op.65"
        self.origin_midi_dir = maestro_root
        self.output_file_dir = os.path.join(data_root, 'output')
        self.csv_output_dir = os.path.join(data_root, 'csv')
        self.mapping_csv_dir = os.path.join(self.npy_root_path, 'name_id_map.csv')

        # To get original Header with matching
        self.composer = ""
        self.orig_midi_name = ""
        self.maestro_midi_name = ""
        self.success_num = 0
        self.limit_success_num = 1000000000
        self.epsilon_folder = ""

    # --------------------------------------------------------------------------
    # functions

    def start_track_string(self, track_num):
        return str(track_num) + ", 0, Start_track\n"

    def title_track_string(self, track_num):
        return str(track_num) + ', 0, Title_t, "Test file"\n'

    def program_c_string(self, track_num, channel, program_num):
        return (
            str(track_num)
            + ", 0, Program_c, "
            + str(channel)
            + ", "
            + str(int(program_num))
            + "\n"
        )

    def note_on_event_string(self, track_num, delta_time, channel, pitch, velocity):
        return (
            str(track_num)
            + ", "
            + str(delta_time)
            + ", Note_on_c, "
            + str(channel)
            + ", "
            + str(pitch)
            + ", "
            + str(velocity)
            + "\n"
        )

    def control_change_event_string(
        self, track_num, delta_time, channel, pitch, velocity
    ):

        return (
            str(track_num)
            + ", "
            + str(delta_time)
            + ", Control_c, "
            + str(channel)
            + ", "
            + str(pitch)
            + ", "
            + str(velocity)
            + "\n"
        )

    def note_off_event_string(self, track_num, delta_time, channel, pitch, velocity):
        return (
            str(track_num)
            + ", "
            + str(delta_time)
            + ", Note_off_c, "
            + str(channel)
            + ", "
            + str(pitch)
            + ", "
            + str(velocity)
            + "\n"
        )

    def end_track_string(self, track_num, delta_time):
        return str(track_num) + ", " + str(delta_time) + ", End_track\n"

    def run(self):

        # Get all path for npy_path by root
        self.load_npy_path()

        print(">>>>>> Converting <<<<<<")

        n_successes = 0
        n_fails = 0
        for index, cur_npy in enumerate(self.npy_path_list):

            try:
                self.name_id_map_restore(cur_npy)
                print(self.composer + " " + self.orig_midi_name)
                print("PATH: " + cur_npy + "\n")
                self.convert_file(cur_npy)
                n_successes += 1
                print("Converting Succeed\n")
            except:
                print("Error occured at: " + cur_npy + "\n")
                exit()
                n_fails += 1
                continue

            if self.success_num == self.limit_success_num:
                break

        print("successes: {} / fails: {}".format(n_successes, n_fails))
        return

    def convert_file(self, file):

        # TODO: Modify Discrete Sound
        off_note = 0
        success_num = 0
        new_csv_string = []

        total_track = 0
        track_num = 1  # Set the Track number

        program_num = 0
        delta_time = 0
        channel = 0
        pitch = 60
        velocity = 90

        # FOR CHECKING
        print("file", file)
        if os.path.isfile(file):
            file_name = file.split("/")[-1]

            print("file_name", file_name)
            epsilon_folder = file.split("/")[-2]

            print("epsilon_folder", epsilon_folder)
            self.epsilon_folder = epsilon_folder

            if "orig" in file_name:
                self.atype = "orig"
            elif "noise" in file_name:
                self.atype = "noise"
            else:  # origin input2midi
                self.atype = "att"

            # VP: there was an if/else (atype) but in both branches the following was executed...:
            only_file_name = file_name.replace(".npy", "")

            new_csv_string = []
            load_data = np.load(file)
            load_data = np.squeeze(load_data)

            print("load_data", load_data)

            origin_file = self.origin_midi_dir + self.get_origin_file_name(
                self.composer, self.orig_midi_name
            )

            print("Original file:", origin_file)

            try:
                origin_file_csv = py_midicsv.midi_to_csv(origin_file)
                print("origin_file_csv", origin_file_csv)
            except:
                print("MIDI_TO_CSV ERROR!!")

            else:
                print("MIDI_TO_CSV WORKED!!")
                # print("current file:", file)
                # for string in origin_file_csv:
                #    if 'Program_c' in string: print(string)

                total_track = 2
                current_used_instrument = [-1, -1]
                # find total track num
                for instrument_num, lst in enumerate(
                    load_data
                ):  # instrument_num : 0-127
                    if np.sum(lst) != (off_note) * 400 * 128:
                        total_track += 1
                        current_used_instrument.append(instrument_num)

                # slower by 4.8
                header = origin_file_csv[0].split(", ")
                # print('Before header:', header)
                header[-1] = str(int(int(header[-1][:-1]) * 1.3)) + "\n"
                header[-2] = str(int(total_track))
                # print('After header:', header)
                new_csv_string.append(
                    ", ".join(header)
                )  # header_string(total_track) + change last to 168 (too fast)
                new_csv_string.append(
                    origin_file_csv[1]
                )  # self.start_track_string(track_num)

                for string in origin_file_csv:
                    if "SMPTE_offset" in string:
                        # print(string)
                        continue
                    elif "Time_signature" in string or "Tempo" in string:
                        new_csv_string.append(string)

                    elif "Program_c" in string:
                        break

                new_csv_string.append(self.end_track_string(track_num, delta_time))

                # Set the track_string_list to identify different instrument time line
                track_string_list = [[] for i in range(0, total_track)]
                track_string_list[0].append(-1)  # To Generate Error -> Header File
                track_string_list[1].append(-1)  # To Generate Error -> Meta File

                note_on_list = [[] for i in range(0, total_track)]
                note_on_list[0].append(-1)
                note_on_list[1].append(-1)

                control_change_list = [[] for i in range(0, total_track)]
                control_change_list[0].append(-1)
                control_change_list[1].append(-1)

                note_off_list = [[] for i in range(0, total_track)]
                note_off_list[0].append(-1)
                note_off_list[1].append(-1)

                # print(load_data.shape[0], " ", load_data.shape[1], " ", load_data.shape[2])
                for channel_instrument in range(0, load_data.shape[0]):
                    for row in range(0, load_data.shape[1]):
                        for col in range(0, load_data.shape[2]):

                            if load_data[channel_instrument][row][col] == off_note:
                                continue
                            else:
                                # Set the different condition for attacked Midi Files
                                # print('music21 instrument:', load_data[row][col]) # 0-59
                                # print('py_midicsv instrument:', program_num_map[load_data[row][col]])

                                if (
                                    len(
                                        track_string_list[
                                            current_used_instrument.index(
                                                channel_instrument
                                            )
                                        ]
                                    )
                                    != 0
                                ):
                                    program_num = channel_instrument  # program_num = instrment num
                                    pitch = col
                                    channel = 0
                                    delta_time = 50 * row
                                    end_delta_time = 50 * (row + 1)
                                    velocity = int(
                                        load_data[channel_instrument][row][col]
                                    )

                                    # Check if the note is continuous or not

                                    # Append Note_on when before event don't exist

                                    if row != 0 and (
                                        load_data[channel_instrument][row - 1][col] == 0
                                    ):
                                        note_on_list[track_num].append(
                                            [
                                                track_num,
                                                delta_time,
                                                channel,
                                                pitch,
                                                velocity,
                                            ]
                                        )

                                    elif row != 0 and (
                                        load_data[channel_instrument][row - 1][col] != 0
                                    ):

                                        control_change_list[track_num].append(
                                            [
                                                track_num,
                                                delta_time,
                                                channel,
                                                pitch,
                                                velocity,
                                            ]
                                        )

                                    # Append Note_off when after event don't exist

                                    if row != (load_data.shape[1] - 1) and (
                                        load_data[channel_instrument][row + 1][col] == 0
                                    ):
                                        note_off_list[track_num].append(
                                            [
                                                track_num,
                                                end_delta_time,
                                                channel,
                                                pitch,
                                                velocity,
                                            ]
                                        )

                                else:
                                    # Set the track_string_list new track header - program_c event
                                    track_num = current_used_instrument.index(
                                        channel_instrument
                                    )
                                    if channel_instrument == 128:
                                        program_num = 1
                                    else:
                                        program_num = channel_instrument
                                    channel = 0
                                    pitch = col
                                    delta_time = 50 * row
                                    end_delta_time = 50 * (row + 1)
                                    velocity = int(
                                        load_data[channel_instrument][row][col]
                                    )
                                    track_string_list[track_num].append(
                                        self.start_track_string(track_num)
                                    )
                                    track_string_list[track_num].append(
                                        self.title_track_string(track_num)
                                    )
                                    track_string_list[track_num].append(
                                        self.program_c_string(
                                            track_num, channel, program_num
                                        )
                                    )

                                    if row != 0 and (
                                        load_data[channel_instrument][row - 1][col] == 0
                                    ):
                                        note_on_list[track_num].append(
                                            [
                                                track_num,
                                                delta_time,
                                                channel,
                                                pitch,
                                                velocity,
                                            ]
                                        )

                                    elif row != 0 and (
                                        load_data[channel_instrument][row - 1][col] != 0
                                    ):

                                        control_change_list[track_num].append(
                                            [
                                                track_num,
                                                delta_time,
                                                channel,
                                                pitch,
                                                velocity,
                                            ]
                                        )

                                    if row != (load_data.shape[1] - 1) and (
                                        load_data[channel_instrument][row + 1][col] == 0
                                    ):
                                        note_off_list[track_num].append(
                                            [
                                                track_num,
                                                end_delta_time,
                                                channel,
                                                pitch,
                                                velocity,
                                            ]
                                        )

                        for num in range(2, len(note_on_list)):  # num = track num
                            for notes in range(0, len(note_on_list[num])):
                                track_string_list[num].append(
                                    self.note_on_event_string(
                                        note_on_list[num][notes][0],
                                        note_on_list[num][notes][1],
                                        note_on_list[num][notes][2],
                                        note_on_list[num][notes][3],
                                        note_on_list[num][notes][4],
                                    )
                                )

                        for num in range(
                            2, len(control_change_list)
                        ):  # num = track num
                            for notes in range(0, len(control_change_list[num])):
                                track_string_list[num].append(
                                    self.control_change_event_string(
                                        control_change_list[num][notes][0],
                                        control_change_list[num][notes][1],
                                        control_change_list[num][notes][2],
                                        control_change_list[num][notes][3],
                                        control_change_list[num][notes][4],
                                    )
                                )

                        for num in range(2, len(note_off_list)):
                            for notes in range(0, len(note_off_list[num])):
                                track_string_list[num].append(
                                    self.note_off_event_string(
                                        note_off_list[num][notes][0],
                                        note_off_list[num][notes][1],
                                        note_off_list[num][notes][2],
                                        note_off_list[num][notes][3],
                                        note_off_list[num][notes][4],
                                    )
                                )

                        note_on_list = [[] for i in range(0, total_track)]
                        control_change_list = [[] for i in range(0, total_track)]
                        note_off_list = [[] for i in range(0, total_track)]

                end_delta_time = 400 * 50
                for i in range(2, len(track_string_list)):
                    for j in track_string_list[i]:
                        new_csv_string.append(j)
                    new_csv_string.append(self.end_track_string(i, end_delta_time))
                new_csv_string.append("0, 0, End_of_file\n")  # end of file string
                # print('NEW STRING')

                # data = pd.DataFrame(new_csv_string)
                # data.to_csv(csv_output_dir,index = False)

                midi_object = py_midicsv.csv_to_midi(new_csv_string)

                self.make_directory(
                    os.path.join(self.output_file_dir, str(epsilon_folder), "origin")
                )
                self.make_directory(
                    os.path.join(self.output_file_dir, str(epsilon_folder), "attack")
                )

                new_output_file_dir = os.path.join(self.output_file_dir, str(epsilon_folder))
                if self.atype == "orig":

                    new_output_file_dir = (
                        new_output_file_dir
                        + "/origin/"
                        + self.orig_midi_name
                        + "_"
                        + self.atype
                        + "_"
                        + only_file_name
                        + ".mid"
                    )

                elif self.atype == "att":

                    new_output_file_dir = (
                        new_output_file_dir
                        + "/attack/"
                        + self.orig_midi_name
                        + "_"
                        + self.atype
                        + "_"
                        + only_file_name
                        + ".mid"
                    )

                with open(new_output_file_dir, "wb",) as output_file:
                    midi_writer = py_midicsv.FileWriter(output_file)
                    midi_writer.write(midi_object)
                    # print("Good Midi File")

                    self.success_num += 1

                # For Cheking Error Data, Represent to csv files
                if self.csv_printable:
                    self.checking_csv(only_file_name)

    def checking_csv(self, only_file_name):
        csv_string = py_midicsv.midi_to_csv(
            self.output_file_dir
            + str(self.epsilon_folder)
            + "/"
            + "New_"
            + self.atype
            + "_"
            + self.orig_midi_name
            + "_"
            + only_file_name
            + ".mid"
        )
        tmp_list = []
        for i in range(0, len(csv_string)):
            temp = np.array(csv_string[i].replace("\n", "").replace(" ", "").split(","))
            tmp_list.append(temp)
        data = pd.DataFrame(tmp_list)
        data.to_csv(
            self.csv_output_dir
            + "New_"
            + self.atype
            + "_"
            + self.orig_midi_name
            + "_"
            + only_file_name
            + ".csv",
            header=False,
            index=False,
        )
        print(".csv saved!")

    def load_npy_path(self):
        """
        return: list of all the npy_path(abs_path)
        """

        self.npy_path_list = []
        mapping_csv_df = pd.read_csv(
            self.mapping_csv_dir, encoding="UTF-8", index_col=False
        )  # read mapping csv
        mapping_csv_df = mapping_csv_df.drop(
            mapping_csv_df.columns[[0]], axis="columns"
        )
        # print(mapping_csv_df)

        # Find all of the npy converted files

        for dirpath, dirnames, filenames in os.walk(self.npy_root_path):

            for filename in filenames:

                if filename.endswith(".npy"):

                    current_npy_path = str(dirpath) + "/" + str(filename)
                    self.npy_path_list.append(current_npy_path)
                    # print('Saved file path: ',current_npy_path) # Debug

        return self.npy_path_list

    def name_id_map_restore(self, cur_npy_string):
        print("cur_npy_string", cur_npy_string)
        # Set self.composer, self.orig_midi_name for right place
        split_string_list = cur_npy_string.split("/")  # List
        # print("split_string_list", split_string_list)
        attack_split_string_list = split_string_list[-1].split("_")
        # print("attack_split_string_list", attack_split_string_list)
        composer_num = -1
        midi_num = -1

        # Find the composer num, midi num position
        for index, dir in enumerate(split_string_list):
            if "composer" in dir:
                composer_num = int(dir.replace("composer", ""))

            if "midi" in dir:
                midi_num = int(dir.replace("midi", ""))

        mapping_csv_df = pd.read_csv(
            self.mapping_csv_dir, encoding="UTF-8", index_col=False
        )  # read mapping csv
        mapping_csv_df = mapping_csv_df.drop(
            mapping_csv_df.columns[[0]], axis="columns"
        )

        is_composer = mapping_csv_df["composer_id"] == composer_num
        is_song = mapping_csv_df["midi_id"] == midi_num

        print("is_composer", is_composer.sum())
        print("is_song", is_song.sum())

        subset_df = mapping_csv_df[is_composer & is_song]

        print("subset_df")
        print(subset_df.head())
        # exit()

        self.composer = subset_df.iloc[0].loc["composer"]
        self.orig_midi_name = subset_df.iloc[0].loc["orig_name"]
        print("Current Composer ", self.composer)
        print("Current Song", self.orig_midi_name)

    def get_origin_file_name(self, composer, orig_midi_name):
        """
        Get midi_filename at MAESTRO Dataset CSV
        composer : (str) composer name for csv
        orig_midi_name : (str) Restored Canonical_tilte for csv matching
        return : (str) MAESTRO Midi File name mapped
        """

        # maestro_csv_df = pd.read_csv(
        #     "/data/MAESTRO/maestro-v2.0.0/maestro-v2.0.0_cleaned.csv", encoding="euc-kr"
        # )
        maestro_csv_df = pd.read_csv(
            "classifier/meta/maestro-v2.0.0-reduced.csv" # ???, encoding="euc-kr"
        )

        print("maestro_csv_df")
        print(maestro_csv_df.head())

        print("params", composer, orig_midi_name)

        is_composer = maestro_csv_df["canonical_composer"] == composer
        is_title = maestro_csv_df["canonical_title"] == orig_midi_name

        print("is_composer", is_composer.sum())
        print("is_title", is_title.sum())

        subset_df = maestro_csv_df[is_composer & is_title]

        print("subset_df")
        print(subset_df.head())

        self.maestro_midi_name = subset_df.iloc[0].loc["midi_filename"]

        return self.maestro_midi_name

    def make_directory(self, dir_path):
        """
        Check if the directory have been placed or not
        """

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        return

if __name__ == "__main__":
    conv = Converter()
    conv.run()
