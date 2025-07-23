import os
from collections import OrderedDict
from typing import Any

import torch

try:
    from src.model.inception import InceptionNetworkWithDownsampling
except ImportError:
    from inception import InceptionNetworkWithDownsampling


class InceptionEnsemble(torch.nn.Module):
    def __init__(
        self, weights_dir: str, weights_startswith: str = "load_and_train_", inception_kwargs: dict[str, Any] = {}
    ) -> None:
        super(InceptionEnsemble, self).__init__()
        models_list = []
        all_weights = [f for f in os.listdir(weights_dir) if f.startswith(weights_startswith)]
        for weight_file in all_weights:
            model = InceptionNetworkWithDownsampling(**inception_kwargs)
            model.load_weights(os.path.join(weights_dir, weight_file))
            models_list.append(model)
        self.models: torch.nn.ModuleList = torch.nn.ModuleList(models_list)
        self.num_in_channels = self.models[0].num_in_channels
        print(
            f"InceptionEnsemble: Loaded {len(self.models)} models from {weights_dir} with prefix {weights_startswith}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs, dim=1)

    def load_weights(self, path: str) -> None:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(loaded, OrderedDict):
            # remove the keys "_orig_mod" as it is added by torch.compile
            loaded = OrderedDict((k.replace("_orig_mod.", ""), v) for k, v in loaded.items())
            self.load_state_dict(loaded)
        elif isinstance(loaded, tuple):
            loaded = loaded[0]
            loaded = OrderedDict((k.replace("_orig_mod.", ""), v) for k, v in loaded.items())
            self.load_state_dict(loaded)
        else:
            print("Could not load weights")
