import pickle

import numpy as np
import partitura
import json

import torch


def pianoroll2midi(pianoroll3d, out_path, samples_per_second=20, channels=2):
    """Generate a midi file from a pianoroll. 
    The expected pianoroll shape is (2,128,x) or (2,88,x), or """
    if pianoroll3d.max() == 0:
        raise ValueError("There should be at least one note played in the pianoroll")
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


def midi_to_3dpianoroll(midi_path, out_npy_path, frame_length=0.05, piano_range=True):
    """Generate a 3dpianoroll from a midi file."""
    part = partitura.load_performance_midi(midi_path)
    part.sustain_pedal_threshold = 127  # discard pedal information
    matrix1 = partitura.utils.compute_pianoroll(
        part,
        time_unit="sec",
        time_div=int(1 / frame_length),
        onset_only=True,
        piano_range=piano_range,
    ).toarray()
    matrix1[matrix1 > 0] = 1  # we don't want velocity here
    matrix2 = partitura.utils.compute_pianoroll(
        part,
        time_unit="sec",
        time_div=int(1 / frame_length),
        onset_only=False,
        piano_range=piano_range,
    ).toarray()
    pr3d = np.stack([matrix1, matrix2])
    pr3d = np.transpose(pr3d, (0, 2, 1))
    np.save(str(out_npy_path), pr3d)


def load_concept_mapping(file_path="concepts/concept_mapping.json"):
    """ Loads mapping between a concept and its (fixed) ID. """
    with open(file_path) as json_file:
        concept_to_id = json.load(json_file)
    id_to_concept = {v: k for k, v in concept_to_id.items()}
    return concept_to_id, id_to_concept


def handle_midi_length(x, desired_length):
    """ Pads or crops a midi to the desired length. """
    current_length = x.shape[1]
    if current_length > desired_length:
        center = current_length // 2
        adjust_odd_lengths = desired_length % 2
        x = x[:, center - desired_length // 2: center + desired_length // 2 + adjust_odd_lengths, :]

    if current_length < desired_length:
        x_new = np.zeros((x.shape[0], desired_length, x.shape[2]))
        x_new[:, :current_length, :] = x
        x = x_new

    return x


def get_tensor_from_filename(filename, clip=True, pad=False):
    """ Loads a (preprocessed) midi, ensures correct length and returns tensor. """
    assert clip + pad <= 1, 'you can\'t set clip and pad'
    print("loading {}...".format(filename))
    x = np.load(filename)
    if clip:
        x = handle_midi_length(x, 316)
    if pad:
        x = handle_midi_length(x, 400)
    x = torch.tensor(x).float()
    return x


def pickle_dump(x, path):
    """ Dumps given data to file path. """
    pickle.dump(x, open(path, "wb"))


def pickle_load(path):
    """ Loads pickled data from given file path. """
    return pickle.load(open(path, "rb"))


def npyfy(x):
    return x.detach().cpu().numpy()
