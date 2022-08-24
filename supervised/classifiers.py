# based on DefaultClassifier in captum.concept._utils.classifer - precise documentation
# can be found there!
from captum.concept._utils.classifier import *


class SGDClassifier(Classifier):
    """ SGD Classifier, similar to DefaultClassifier of Captum with small adaptations. """

    def __init__(self):
        self.lm = model.SkLearnSGDClassifier(alpha=0.0001, max_iter=5000, tol=1e-3)

    def train_and_eval(
        self, dataloader: DataLoader, test_split_ratio: float = 0.33, **kwargs: Any
    ) -> Union[Dict, None]:
        inputs = []
        labels = []
        for input, label in dataloader:
            inputs.append(input)
            labels.append(label)

        device = "cpu" if input is None else input.device
        self.lm.device = device
        # we can train the CAV on all data (no extra test set necessary)
        self.lm.fit(DataLoader(TensorDataset(torch.cat(inputs), torch.cat(labels)), shuffle=True))

        predict = self.lm(torch.cat(inputs))
        # fixing an issue with predictions here
        predict = self.lm.classes()[torch.tensor([0 if p.item() < 0. else 1 for p in predict])]
        score = predict.long() == torch.cat(labels).long().cpu()

        accs = score.float().mean()

        return {"accs": accs}

    def weights(self) -> Tensor:
        weights = self.lm.representation()
        if weights.shape[0] == 1:
            # if there are two concepts, there is only one label. We split it in two.
            return torch.stack([-1 * weights[0], weights[0]])
        else:
            return weights

    def classes(self) -> List[int]:
        return self.lm.classes().detach().numpy()


class LogisticRegression(Classifier):
    """ Logistic regression classifier, based on DefaultClassifier of captum. """

    def __init__(self):
        self.lm = model.SkLearnLogisticRegression()

    def train_and_eval(
        self, dataloader: DataLoader, test_split_ratio: float = 0.33, **kwargs: Any
    ) -> Union[Dict, None]:
        inputs = []
        labels = []
        for input, label in dataloader:
            inputs.append(input)
            labels.append(label)

        device = "cpu" if input is None else input.device
        self.lm.device = device
        # we can train the CAV on all data (no extra test set necessary)
        self.lm.fit(DataLoader(TensorDataset(torch.cat(inputs), torch.cat(labels)), shuffle=True))

        predict = self.lm(torch.cat(inputs))
        # fixing an issue with predictions here
        predict = self.lm.classes()[torch.tensor([0 if p.item() < 0. else 1 for p in predict])]
        score = predict.long() == torch.cat(labels).long().cpu()

        accs = score.float().mean()

        return {"accs": accs}

    def weights(self) -> Tensor:
        weights = self.lm.representation()
        if weights.shape[0] == 1:
            # if there are two concepts, there is only one label. We split it in two.
            return torch.stack([-1 * weights[0], weights[0]])
        else:
            return weights

    def classes(self) -> List[int]:
        return self.lm.classes().detach().numpy()
