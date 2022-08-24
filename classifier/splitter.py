""" Taken from original repo (https://github.com/KimSSung/Deep-Composer-Classification/blob/master/spliter.py) """

from config import data_root, splits_root

import os
import random


# random.seed(333)  # change this


class Splitter:
    def __init__(self):
        self.input_path = os.path.join(data_root, 'npy')
        self.save_dir = splits_root
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # print("omit:", self.omitlist)

        self.composer = []
        self.composer_map = {}

        self.composer_midi_count = []  # Total midi folder num of each composer

        self.train_percentage = 0.7

        print("###################################")
        print(
            ">> Train : Valid = "
            + str(int(self.train_percentage * 10))
            + " : "
            + str(int((1 - self.train_percentage) * 10))
            + " <<"
        )
        print()

    def run(self):
        self.counts()

    def counts(self):

        for folder in os.listdir(self.input_path):
            if folder == "name_id_map.csv":
                continue

            comp_idx = folder.replace("composer", "")  # str

            # Optional: omit some composers
            # if comp_idx in self.omitlist:
            #     continue
            self.composer.append(int(comp_idx))

            composer_fold = os.path.join(self.input_path, folder)
            count = 0
            midi_count = 0
            seg_per_midi_composer = []
            for midi_f in os.listdir(composer_fold):
                midi_count += 1

            self.composer_midi_count.append(midi_count)

        self.composer_mapping()
        self.prints()
        self.splits()

    def composer_mapping(self):
        # [7, 0, 12, 9, 11, 10, 8, 4, 6, 3, 2, 1, 5]
        for idx, comp in enumerate(self.composer):
            self.composer_map[idx] = self.composer[idx]

    def prints(self):

        print("## composer map:")
        print(self.composer_map)
        print("## composer's total midi counts:")
        print(self.composer_midi_count)
        print("## Total midi count:")
        print(sum(self.composer_midi_count))

        # self.print_3age()

    def print_3age(self):
        pass
        # Consider Age
        # 1. Baroque: Scarlatti / Bach => [2, 6]
        # 2. Classical: Haydn / Mozart / Beethoven / Schubert => [4, 8, 9, 12]
        # 3. Romanticism: Schumann / Chopin / Liszt / Brahms / Debussy
        #                 / Rachmaninoff / Scriabin => [0, 1, 3, 5, 7, 10, 11]

        # baroq_seg, classic_seg, roman_seg = [], [], []
        # baroq_midi, classic_midi, roman_midi = [], [], []

        # for i in range(self.config.composers):
        #     if self.composer_map[i] in [2, 6]:
        #         baroq_seg.append(self.composer_seg_count[i])
        #         baroq_midi.append(self.composer_midi_count[i])
        #     elif self.composer_map[i] in [4, 8, 9, 12]:
        #         classic_seg.append(self.composer_seg_count[i])
        #         classic_midi.append(self.composer_midi_count[i])
        #     else:
        #         roman_seg.append(self.composer_seg_count[i])
        #         roman_midi.append(self.composer_midi_count[i])

        # print("## 3 Age")
        # print("Baroque:")
        # print("Seg:", sum(baroq_seg))
        # print("Midi:", sum(baroq_midi))
        # print()
        # print("Classical:")
        # print("Seg:", sum(classic_seg))
        # print("Midi:", sum(classic_midi))
        # print()
        # print("Romanticism:")
        # print("Seg:", sum(roman_seg))
        # print("Midi:", sum(roman_midi))
        # print()

    def splits(self):

        train_file = open(os.path.join(self.save_dir,"train.txt"), "w")
        valid_file = open(os.path.join(self.save_dir,"valid.txt"), "w")

        midi_idxs = []

        # Select train / valid 'midi' from each composer
        for midi_count in self.composer_midi_count:
            idxlist = random.sample(
                list(range(0, midi_count)), int(midi_count * self.train_percentage)
            )
            midi_idxs.append(idxlist)

        total_train_midi = 0
        total_valid_midi = 0
        composer_idx = 0  # to exclude 'csv' file on composer_idx count

        # if self.config.age == True
        baroq, classic, roman = [], [], []
        baroq_file, classic_file, roman_file = [], [], []
        bcnt, ccnt, rcnt = 0, 0, 0

        for folder in os.listdir(self.input_path):
            if folder == "name_id_map.csv":
                continue
            # elif folder.replace("composer", "") in self.omitlist:
            #     continue

            composer_fold = os.path.join(self.input_path, folder)
            for this_midi_idx, midi_f in enumerate(os.listdir(composer_fold)):

                midi_fold = os.path.join('npy', folder, midi_f)

                # if not self.config.age:  # Not Age
                # Train
                if this_midi_idx in midi_idxs[composer_idx]:
                    train_file.write(midi_fold + "\n")
                    total_train_midi += 1
                # valid
                else:
                    valid_file.write(midi_fold + "\n")
                    total_valid_midi += 1

            composer_idx += 1

        # age_cnt = [len(baroq_file), len(classic_file), len(roman_file)]
        # if self.config.age: # write train / valid file

        # 	# under sampling
        # 	b_idxlist = random.sample(list(range(0, len(baroq_file))), min(age_cnt))
        # 	c_idxlist = random.sample(list(range(0, len(classic_file))), min(age_cnt))
        # 	r_idxlist = random.sample(list(range(0, len(roman_file))), min(age_cnt))

        # 	# split train / valid

        # if self.config.age:
        #     print(
        #         "## Each midi * "
        #         + str(self.each_seg)
        #         + " seg (Baroq / Classic / Roman):",
        #         age_cnt,
        #     )
        #     print()

        print("## After split:")
        # print("total:", sum(self.composer_seg_count))
        # print("goal train num:", int(sum(self.composer_seg_count) * self.train_percentage))
        # print("goal valid num:", int(sum(self.composer_seg_count) * (1 - self.train_percentage)))
        # print()

        print("total train count:", total_train_midi)
        print("total valid count:", total_valid_midi)

        train_file.close()
        valid_file.close()


if __name__ == "__main__":
    temp = Splitter()
    temp.run()