from config import concepts_path
from argparse import ArgumentParser
import os
from classifier.generator import Generator
from data_handling.utils import midi_to_3dpianoroll


if __name__ == '__main__':
    concepts_midi_path = os.path.join(concepts_path, 'midi')

    parser = ArgumentParser()
    parser.add_argument("--concept_name", required=True, type=str)
    args = parser.parse_args()

    selected_concept = args.concept_name
    concept_dir = os.path.join(concepts_midi_path, selected_concept)

    generator = Generator()

    destination_dir = os.path.join(concepts_path, 'npy', selected_concept)

    if os.path.exists(destination_dir):
       print('{} already exists ... quitting ...'.format(destination_dir))
       exit()

    os.makedirs(destination_dir)

    for midi_file in os.listdir(concept_dir):
        in_path = os.path.join(concept_dir, midi_file)
        print("processing {}".format(in_path))
        out_path = os.path.join(destination_dir, midi_file.replace(".mid", ".npy"))
        midi_to_3dpianoroll(in_path, out_path)
