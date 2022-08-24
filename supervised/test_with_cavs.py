import os
import numpy as np

from pathlib import Path
from argparse import ArgumentParser

from captum.concept import TCAV
from captum.attr import LayerIntegratedGradients
from captum.concept._utils.common import concepts_to_str

from supervised.classifiers import *
from classifier.tools.resnet import resnet50
from supervised.tcav_utils import assemble_padded_concept
from config import concepts_path, results_root, splits_root, data_root
from data_handling.utils import load_concept_mapping, get_tensor_from_filename, pickle_dump

composer_mapping = {
    'scriabin': 0,
    'debussy': 1,
    'scarlatti': 2,
    'liszt': 3,
    'schubert': 4,
    'chopin': 5,
    'bach': 6,
    'brahms': 7,
    'haydn': 8,
    'beethoven': 9,
    'schumann': 10,
    'rachmaninoff': 11,
    'mozart': 12
}


def opts_parser():
    """ Prepares argument parsing. """
    desc = 'Script to perform testing with CAVs.'
    parser = ArgumentParser(description=desc)
    parser.add_argument('modelfile', metavar='MODELFILE', type=str,
                        help='File to load the learned weights from (.pt format)')
    parser.add_argument('--concept-path', metavar='DIR', type=str,
                        default=Path(concepts_path) / 'npy', help='Path to concept data.')
    parser.add_argument('--experiment-root', metavar='DIR', type=str,
                        default=results_root, help='Path to experimental folder.')
    parser.add_argument('--valid-file', metavar='FILE', type=str,
                        default=os.path.join(splits_root, 'valid.txt'),
                        help='Path pointing to file containing validation files of trained network.')
    parser.add_argument('--omit-onset', action='store_true',
                        help='Whether to omit onset or not.')
    parser.add_argument('--layers', nargs='+', default=['layer1', 'layer2', 'layer3', 'layer4'],
                        help='The layers we want to compute explanations for.')
    parser.add_argument('--composer', type=str, default='mozart',
                        help='Which composer we want to test for.')
    parser.add_argument('--save-path', type=str, required=True,
                        help='Defines path were scores should be stored to.')
    parser.add_argument('--concepts', nargs='+', default=[1, 6, 7], type=int,
                        help='The concepts we want to compute explanations for. ')
    parser.add_argument('--random-concepts', nargs='+', default=[90], type=int,
                        help='The random concepts used for TCAV.')
    parser.add_argument('--concept-classifier', type=str, default='sgd',
                        help='Specify classifier for concept vs random dataset.')
    return parser


def prepare_experimental_sets(concept_path, options):
    """ Given (non-) random concepts, prepares them and according experimental sets. """
    concept_to_id, id_to_concept = load_concept_mapping()
    random_concept_paths, concept_paths = [], []
    for id in options.random_concepts:
        random_concept_paths.append(concept_path / id_to_concept[id])
    for id in options.concepts:
        concept_paths.append(concept_path / id_to_concept[id])
    # assemble concepts
    random_concepts_list, concepts_list = [], []
    for concept in random_concept_paths:
        random_concepts_list.append(
            assemble_padded_concept(concept.name, concept_to_id['random_datasets/' + concept.name],
                                    concepts_path=concept.parent, include_onset=(not options.omit_onset)))
    for concept in concept_paths:
        concepts_list.append(
            assemble_padded_concept(concept.name, concept_to_id[concept.name], concepts_path=concept.parent,
                                    include_onset=(not options.omit_onset)))
    # prepare combination of concepts w/ random set for different experiments
    experimental_set = []
    for c in concepts_list:
        for rc in random_concepts_list:
            experimental_set.append([c, rc])
    print(experimental_set)
    return experimental_set


def get_model(modelfile, device, omit_onset):
    """ Loads (pre-trained) model. """
    checkpoint = torch.load(modelfile, map_location=device)
    state_dict = {k.replace('module.', ''): checkpoint['model.state_dict'][k] for k in
                  checkpoint['model.state_dict'].keys()}
    if modelfile.name.startswith('resnet50'):
        model = resnet50(in_channels=(1 if omit_onset else 2), num_classes=13).to(device)
    else:
        raise NotImplementedError('Only supports resnet 50 for now')
    model.load_state_dict(state_dict)
    model.eval()

    model_dir = modelfile.parts[-3]
    print('using {} as model_dir'.format(model_dir))
    return model, model_dir


def get_inputs(composer_idx, valid_file, include_onset=True):
    """ Given a particular composer and file containing validation samples, returns inputs we test with. """
    with open(valid_file, 'r') as fp:
        files = [Path(data_root) / f.rstrip() for f in fp if 'composer{}'.format(composer_idx) in f]
    # next line chooses an arbitrary version for each piece
    # if you actually want to be able to reproduce stuff, fix a random seed below - be smarter than me
    # random.seed(21)
    files = [random.choice(list(f.rglob('*.npy'))) for f in files]

    # load inputs, include onsets only if necessary
    inputs = [get_tensor_from_filename(f, clip=False, pad=False) for f in files]

    if not include_onset:
        inputs = [x[1, :, :].unsqueeze(0) for x in inputs]

    return inputs


def get_classifier(classifier_name):
    """ Returns desired classifier for CAV computation (sgd or logistic regression). """
    if classifier_name.lower() == 'sgd':
        return SGDClassifier()
    elif classifier_name.lower() == 'logistic_regression':
        return LogisticRegression()
    else:
        raise NotImplementedError('Please define valid classifier!')


def compute_tcav_scores(modelfile, experiment_root, concept_path, valid_file, save_path, options):
    """ Performs testing-with-cavs and stores full scores. """
    # prepare concept paths and concepts
    experimental_set = prepare_experimental_sets(concept_path, options)

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
        for ind in range(0, len(total_indices), 20):        # process max 20x 20 sec snippets to avoid too much memory
            ipt = torch.stack([i[:, beg:beg + 400, :] for beg in total_indices[ind:ind + 20]])
            sub_score = manual_tcav(tcav, ipt, experimental_set, composer_idx, 5)
            for c in experimental_set:
                c = concepts_to_str(c)
                tcav_score[c].update({l: torch.cat((tcav_score[c][l], sub_score[c][l]['tcav_score']), dim=0)
                                      for l in options.layers})

    # store result
    print(tcav_score)
    pickle_dump(tcav_score, save_path / 'total_tcav_score.pkl')


def save_score(score_dict, file_name, save_path):
    """ Dumps pickled tcav score at given path. """
    save_name = save_path / (file_name.parts[-3] + '_' + file_name.parts[-2] + '_' + file_name.with_suffix('.pkl').name)
    pickle_dump(score_dict, save_name)


def manual_tcav(tcav, inputs, experimental_sets, target, n_steps):
    """ Adapted from captum (see function below for changed functionality). """
    from typing import Dict, cast, Any
    from collections import defaultdict
    from captum._utils.common import _get_module_from_name, _format_tensor_into_tuples

    tcav.compute_cavs(experimental_sets, processes=None)

    scores = defaultdict(lambda: defaultdict())

    # Retrieves the lengths of the experimental sets so that we can sort
    # them by the length and compute TCAV scores in batches.
    exp_set_lens = np.array(list(map(lambda exp_set: len(exp_set), experimental_sets)), dtype=object)
    exp_set_lens_arg_sort = np.argsort(exp_set_lens)

    # compute offsets using sorted lengths using their indices
    exp_set_lens_sort = exp_set_lens[exp_set_lens_arg_sort]
    exp_set_offsets_bool = [False] + list(exp_set_lens_sort[:-1] == exp_set_lens_sort[1:])
    exp_set_offsets = []
    for i, offset in enumerate(exp_set_offsets_bool):
        if not offset:
            exp_set_offsets.append(i)
    exp_set_offsets.append(len(exp_set_lens))

    # sort experimental sets using the length of the concepts in each set
    experimental_sets_sorted = np.array(experimental_sets, dtype=object)[exp_set_lens_arg_sort]

    for layer in tcav.layers:
        layer_module = _get_module_from_name(tcav.model, layer)
        tcav.layer_attr_method.layer = layer_module

        attribs = tcav.layer_attr_method.attribute(inputs, target=target, n_steps=n_steps)

        attribs = _format_tensor_into_tuples(attribs)
        # n_inputs x n_features (2 dimensions)
        attribs = torch.cat([torch.reshape(attrib, (attrib.shape[0], -1)) for attrib in attribs], dim=1)

        # n_experiments x n_concepts x n_features (3 dimensions)
        cavs = []
        classes = []
        for concepts in experimental_sets:
            concepts_key = concepts_to_str(concepts)
            cavs_stats = cast(Dict[str, Any], tcav.cavs[concepts_key][layer].stats)
            cavs.append(cavs_stats["weights"].float().detach().tolist())
            classes.append(cavs_stats["classes"])

        # sort cavs and classes using the length of the concepts in each set
        cavs_sorted = np.array(cavs, dtype=object)[exp_set_lens_arg_sort]
        classes_sorted = np.array(classes, dtype=object)[exp_set_lens_arg_sort]

        i = 0
        while i < len(exp_set_offsets) - 1:
            cav_subset = np.array(cavs_sorted[exp_set_offsets[i]: exp_set_offsets[i + 1]], dtype=object).tolist()
            classes_subset = classes_sorted[exp_set_offsets[i]: exp_set_offsets[i + 1]].tolist()

            # n_experiments x n_concepts x n_features (3 dimensions)
            cav_subset = torch.tensor(cav_subset)
            cav_subset = cav_subset.to(attribs.device)
            assert len(cav_subset.shape) == 3, "cav should have 3 dimensions: n_experiments x n_concepts x n_features."

            experimental_subset_sorted = experimental_sets_sorted[exp_set_offsets[i]: exp_set_offsets[i + 1]]
            compute_tcav_score(scores, layer, attribs, cav_subset, classes_subset, experimental_subset_sorted)
            i += 1

    return scores


def compute_tcav_score(scores, layer, attribs, cavs, classes, experimental_sets):
    """ Adapted from captum (to save also tcav_score in scores-dict). """
    # n_inputs x n_concepts (2 dimensions)
    tcav_score = torch.matmul(attribs.float(), torch.transpose(cavs, 1, 2))
    assert len(tcav_score.shape) == 3, "tcav_score should have 3 dimensions: n_experiments x n_inputs x n_concepts."

    assert attribs.shape[0] == tcav_score.shape[1], (
        "attrib and tcav_score should have the same 1st and 2nd dimensions respectively (n_inputs).")
    # n_experiments x n_concepts
    sign_count_score = (torch.sum(tcav_score > 0.0, dim=1).float() / tcav_score.shape[1])

    magnitude_score = torch.sum(torch.abs(tcav_score * (tcav_score > 0.0).float()), dim=1) \
                                                                            / torch.sum(torch.abs(tcav_score), dim=1)

    for i, (cls_set, concepts) in enumerate(zip(classes, experimental_sets)):
        concepts_key = concepts_to_str(concepts)

        # sort classes / concepts in the order specified in concept_keys
        concept_ord = {concept.id: ci for ci, concept in enumerate(concepts)}
        new_ord = torch.tensor([concept_ord[cls] for cls in cls_set], device=tcav_score.device)

        # sort based on classes
        scores[concepts_key][layer] = {
            "tcav_score": tcav_score[i],
            "sign_count": torch.index_select(sign_count_score[i, :], dim=0, index=new_ord),
            "magnitude": torch.index_select(magnitude_score[i, :], dim=0, index=new_ord),
        }


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
    save_path = Path(options.save_path) if options.save_path else None
    if save_path and not save_path.exists():
        save_path.mkdir(parents=True)
    experiment_root = Path(options.experiment_root)
    if not experiment_root:
        raise NotADirectoryError('Please define valid experimental root directory!')

    compute_tcav_scores(modelfile, experiment_root, concept_path, valid_file, save_path, options)


if __name__ == '__main__':
    main()
