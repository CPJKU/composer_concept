import os
import torch

from pathlib import Path
from captum.concept import TCAV
from argparse import ArgumentParser
from captum.attr import LayerIntegratedGradients
from captum.concept._utils.common import concepts_to_str

from supervised.tcav_utils import assemble_padded_concept
from config import concepts_path, results_root, splits_root
from data_handling.utils import load_concept_mapping, pickle_dump
from supervised.test_with_cavs import get_model, get_inputs, composer_mapping, manual_tcav, get_classifier


def opts_parser():
    """ Prepares argument parsing. """
    desc = 'Script that tests statistical significance of tcav concept.'
    parser = ArgumentParser(description=desc)
    parser.add_argument('modelfile', metavar='MODELFILE', type=str,
                        help='File to load the learned weights from (.pt format)')
    parser.add_argument('--concept-path', metavar='DIR', type=str,
                        default=Path(concepts_path) / 'npy', help='Path to concept data.')
    parser.add_argument('--concept', default=1, type=int, help='The concept we want to check the significance for.')
    parser.add_argument('--experiment-root', metavar='DIR', type=str,
                        default=results_root, help='Path to experimental folder.')
    parser.add_argument('--valid-file', metavar='FILE', type=str, default=os.path.join(splits_root, 'valid.txt'),
                        help='Path pointing to file containing validation files of trained network.')
    parser.add_argument('--omit-onset', action='store_true', help='Whether to omit onset or not.')
    parser.add_argument('--layers', nargs='+', default=['layer1', 'layer2', 'layer3', 'layer4'],
                        help='The layers we want to compute explanations for.')
    parser.add_argument('--composer', type=str, default='mozart', help='Which composer we want to test for.')
    parser.add_argument('--save-dir', type=str, required=True, help='Defines directory were scores should be stored to.')
    parser.add_argument('--metric', type=str, default='sign_count',
                        help='Either "magnitude" or "sign_count". Which metric to use for significance test.')
    parser.add_argument('--concept-classifier', type=str, default='sgd',
                        help='Specify classifier for concept vs random dataset.')
    return parser


def prepare_experimental_sets(concept_path, concept_id, include_onset):
    """ Prepares experimental concept sets for significance test. """
    concept_to_id, id_to_concept = load_concept_mapping()
    # prepare random concepts
    random_concepts_list, random_concept_paths = [], []
    for rid in [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]:
        random_concept_paths.append(concept_path / id_to_concept[rid])
    for concept in random_concept_paths:
        random_concepts_list.append(
            assemble_padded_concept(concept.name, concept_to_id['random_datasets/' + concept.name],
                                    concepts_path=concept.parent, include_onset=include_onset))
    # prepare non-random concept
    concept = assemble_padded_concept(id_to_concept[concept_id], concept_id,
                                      concepts_path=concept_path, include_onset=include_onset)
    # prepare experimental sets
    experimental_set = [[concept, rc] for rc in random_concepts_list]
    ln = len(experimental_set)
    for rc in random_concepts_list[1:]:
        experimental_set.append([random_concepts_list[0], rc])
    print(experimental_set)
    return experimental_set, ln


def compute_tcav_significance(modelfile, experiment_root, concept_path, valid_file, save_path, options):
    """ Computes tcav scores and performs significance tests on concept. """
    # prepare concept paths and concepts
    experimental_set, ln = prepare_experimental_sets(concept_path, options.concept, not options.omit_onset)

    if options.metric not in ['magnitude', 'sign_count']:
        raise ValueError('{} metric not known, please use "magnitude" or "sign_count"!'.format(options.metric))

    # prepare model
    device = torch.device('cpu')
    net, model_dir = get_model(modelfile, device, options.omit_onset)
    classifier = get_classifier(options.concept_classifier)

    # prepare tcav
    tcav = TCAV(model=net, layers=options.layers, model_id=options.concept_classifier,
                layer_attr_method=LayerIntegratedGradients(net, None, multiply_by_inputs=False),
                save_path=str(experiment_root / 'cav' / model_dir), classifier=classifier)

    if options.composer.lower() not in composer_mapping.keys():
        raise ValueError('Please define other composer, {} is not known!'.format(options.composer))
    composer_idx = composer_mapping[options.composer.lower()]
    inputs = get_inputs(composer_idx, valid_file, not options.omit_onset)

    # compute tcav scores
    tcav_score = {concepts_to_str(c): {l: torch.tensor([]) for l in options.layers} for c in experimental_set}
    for i in inputs:
        # split inputs into 20 sec segments (as during training/testing)
        total_indices = list(range(0, i.shape[1] - 400, 400))
        for ind in range(0, len(total_indices), 20):  # process max 20x 20 sec snippets to avoid too much memory
            ipt = torch.stack([i[:, beg:beg + 400, :] for beg in total_indices[ind:ind + 20]])
            sub_score = manual_tcav(tcav, ipt, experimental_set, composer_idx, 5)
            for c in experimental_set:
                c = concepts_to_str(c)
                tcav_score[c].update({l: torch.cat((tcav_score[c][l], sub_score[c][l]['tcav_score']), dim=0)
                                      for l in options.layers})
    print(tcav_score)
    # store result
    pickle_dump(tcav_score, save_path / 'significance_res.pkl')


def main():
    # parse arguments
    parser = opts_parser()
    options = parser.parse_args()

    # prepare paths, check whether necessary directories exist
    modelfile = Path(options.modelfile)
    if not modelfile.exists():
        raise FileExistsError('Please give a path to an existing model-file!')
    concept_path = Path(options.concept_path)
    valid_file = Path(options.valid_file)
    save_path = Path(results_root) / 'significance_tests' / options.save_dir if options.save_dir else None
    if save_path and not save_path.exists():
        save_path.mkdir(parents=True)
    experiment_root = Path(options.experiment_root)
    if not experiment_root:
        raise NotADirectoryError('Please define valid experimental root directory!')

    compute_tcav_significance(modelfile, experiment_root, concept_path, valid_file, save_path, options)


if __name__ == '__main__':
    main()
