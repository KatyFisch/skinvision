from unittest import TestCase

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

from classifiers.finetune_routed import FROZEN_LAYERS, RoutedResNet18, train_fold


class RoutedFineTuneTest(TestCase):
    def test_cached_views_reproduce_full_forward_and_keep_the_trunk_frozen(self) -> None:
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(42)
        clinical, dermoscopy = torch.rand((2, 4, 3, 32, 40), generator=generator)
        encoder = resnet18().eval().requires_grad_(False)
        encoder.fc = nn.Identity()
        model = RoutedResNet18(encoder).eval()
        trunk = nn.Sequential(*(encoder.get_submodule(name) for name in FROZEN_LAYERS)).eval()
        with torch.no_grad():
            pairs = torch.stack((trunk(clinical), trunk(dermoscopy)), dim=1)
            torch.testing.assert_close(model.classify(model.embed_cached(pairs)), model(clinical, dermoscopy))

        before = {name: value.clone() for name, value in model.state_dict().items()}
        train_fold(
            model,
            pairs.numpy(),
            np.arange(4),
            torch.tensor([1.0, 0.0, 0.0, 1.0]),
            torch.tensor([-1, 0, 1, -1]),
            torch.device("cpu"),
            4,
            1,
            42,
        )
        after = model.state_dict()
        for name, original in before.items():
            if name.startswith("encoder.") and (
                not name.startswith("encoder.layer4.") or name.endswith(("running_mean", "running_var", "num_batches_tracked"))
            ):
                torch.testing.assert_close(after[name], original, rtol=0, atol=0)
        for name in ("encoder.layer4.0.conv1.weight", "router.linear.weight", "other.linear.weight"):
            self.assertFalse(torch.equal(before[name], after[name]), name)
