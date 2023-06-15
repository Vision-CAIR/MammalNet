#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .ava_dataset import Ava  # noqa
from .build import DATASET_REGISTRY, build_dataset  # noqa
from .charades import Charades  # noqa
from .imagenet import Imagenet  # noqa
from .kinetics import Kinetics  # noqa
from .ssv2 import Ssv2  # noqa
from .animalnet_animal import Animalnet_animal
from .animalnet_behavior import Animalnet_behavior
from .animalnet_behavior_composition import Animalnet_behavior_composition



try:
    from .ptv_datasets import Ptvcharades, Ptvkinetics, Ptvssv2  # noqa
except Exception:
    print("Please update your PyTorchVideo to latest master")
