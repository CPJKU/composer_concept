# functions are based on the tutorial https://captum.ai/tutorials/TCAV_Image
# and have been adapted to MIDI data
import os

from captum.concept import Concept
from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset

from data_handling.utils import get_tensor_from_filename


def assemble_padded_concept(name, id, concepts_path, include_onset=True):
    """ Creates dataset (that pads/crops data) and captum-concept of defined concept. """
    def get_padded_tensor_from_filename(filename):
        x = get_tensor_from_filename(filename, clip=False, pad=True)
        if not include_onset:
            x = x[1, :, :].unsqueeze(0)
        return x
    print("assemble padded concept {}".format(name))
    concept_path = os.path.join(concepts_path, name) + '/'
    print("concept_path", concept_path)
    dataset = CustomIterableDataset(get_padded_tensor_from_filename, concept_path)
    concept_iter = dataset_to_dataloader(dataset, batch_size=1)
    return Concept(id=id, name=name, data_iter=concept_iter)
