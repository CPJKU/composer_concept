import torch
import os
import numpy as np
import click
import json
import os
from pathlib import Path

# append folder to ease the debugging
import sys
sys.path.insert(1, os.path.realpath(os.path.curdir))

from unsupervised.ModelWrapper import PytorchModelWrapper
from unsupervised.Explainer import Explainer
from unsupervised.uns_utils import img_utils, AsapPianoRollDataset
from config import asap_root, results_root
from unsupervised.uns_utils import TARGET_COMPOSERS
from config import concepts_path
from classifier.tools.resnet import resnet50



concepts_path = os.path.join(concepts_path, "npy")


def prepare_data(device, target_classes, batch_size, layer, rank):

    classes_names = [TARGET_COMPOSERS[i] for i in target_classes]
    title = (
        "{}_r{}[".format(layer, rank)
        + "_".join(classes_names)
        + "]"
    )
    if not Path(results_root).exists():
        Path(results_root).mkdir()
    # delete old files
    if Path(results_root,title).exists():
        raise Exception("Already computing for", title)
    else:
        Path(results_root, title).mkdir()

    seed = 10
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    model_loc = "classifier/meta/2202180921/model/"
    model_name = "resnet50.pt"

    model = resnet50(in_channels=1, num_classes=13)
    checkpoint = torch.load(os.path.join(model_loc, model_name), map_location=device)
    state_dict = {
        k.replace("module.", ""): checkpoint["model.state_dict"][k]
        for k in checkpoint["model.state_dict"].keys()
    }

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    loaders = []
    datasets = []
    for target in target_classes:
        tdataset = AsapPianoRollDataset(
            composers_idx=[target],
            seg_num=2
        )
        datasets.append(tdataset)
        loaders.append(
            torch.utils.data.DataLoader(
                tdataset, batch_size=batch_size, shuffle=False, num_workers=2
            )
        )
    return (loaders, classes_names, model, title)


def train_explainer(
    model,
    batch_size,
    target_classes,
    classes_names,
    layer_name,
    n_components,
    loaders,
    iter_max,
    dimension,
    reducer,
    title,
    use_cuda
):

    wm = PytorchModelWrapper(
        model,
        batch_size=batch_size,
        predict_target=target_classes,
        input_size=[1, 400, 88],
        input_channel_first=True,
        model_channel_first=True,
        use_cuda = use_cuda
    )

    print("title:{}".format(title))
    print("target_classes:{}".format(target_classes))
    print("classes_names:{}".format(classes_names))
    print("n_components:{}".format(n_components))
    print("layer_name:{}".format(layer_name))

    # create an Explainer
    exp = Explainer(
        title=title,
        layer_name=layer_name,
        class_names=classes_names,
        utils=img_utils(
            img_size=(400, 88), nchannels=1, img_format="channels_first"
        ),
        n_components=n_components,
        reducer_type=reducer,
        nmf_initialization=None,
        dimension=dimension,
        iter_max=iter_max,
    )
    # train reducer based on target classes
    exp.train_model(wm, loaders)
    return exp, wm


def build_explanation(exp, wm, loaders):
    # generate features
    exp.generate_features(wm, loaders)
    # generate global explanations
    exp.global_explanations()
    # generate midi of global explanation
    exp._sonify_features(unfiltered_midi=True)
    exp._sonify_features(contrast=True, unfiltered_midi=True)
    # save the explainer, use load() to load it with the same title
    exp.save()


@click.command()
@click.option("--reducer", help="Either NMF or NTD", default="NMF", type=str)
@click.option("--max-iter", default=1000, type=int)
@click.option("--device", default="cpu", type=str)
@click.option(
    "--targets",
    help="A list of integers (target classes) as string",
    default="[5,6]",
    type=str,
)
@click.option(
    "--dimension", help="An integer, considered only for NTD", default=4, type=int
)
@click.option(
    "--rank", help="An integer, or list of integers as string", default="5", type=str,
)
@click.option(
    "--layer", help="The name of the target layer", default="layer4", type=str
)
@click.option("--batch-size", default=10, type=int)
def start_experiment(
    reducer, max_iter, device, targets, dimension, rank, layer, batch_size
):
    # convert targets string to list
    target_classes = json.loads(targets)
    rank = json.loads(rank)

    loaders, classes_names, model, title = prepare_data(
        device, target_classes, batch_size, layer, rank
    )
    exp, wm = train_explainer(
        model,
        batch_size,
        target_classes,
        classes_names,
        layer,
        rank,
        loaders,
        max_iter,
        dimension,
        reducer,
        title,
        device != "cpu"
    )
    build_explanation(exp, wm, loaders)


def start_experiment_noclick(
    reducer, max_iter, gpu_number, targets, dimension, rank, layer, batch_size
):

    # convert targets string to list
    target_classes = json.loads(targets)
    rank = json.loads(rank)    
    
    loaders, classes_names, model, title = prepare_data(
        str(gpu_number), target_classes, batch_size, layer, rank
    )
    exp, wm = train_explainer(
        model,
        batch_size,
        target_classes,
        classes_names,
        layer,
        rank,
        loaders,
        max_iter,
        dimension,
        reducer,
        title,
    )
    build_explanation(exp, wm, loaders)


if __name__ == "__main__":
    start_experiment()
