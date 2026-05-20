from typing import Dict, Tuple, Union, Optional
import copy
import os
import torch
import torch.nn as nn
import torchvision
from diffusion_policy.model.vision.tactile_cnn_encoder import TactileCNNEncoder
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class MultiTactileObsEncoder(ModuleAttrMixin):
    def __init__(
        self,
        shape_meta: dict,
        tactile_model: Union[nn.Module, Dict[str, nn.Module]],
        resize_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        # replace BatchNorm with GroupNorm
        use_group_norm: bool = False,
        # use single tactile model for all tactile inputs
        share_tactile_model: bool = False,
        # pretrained model path
        pretrained_path: Optional[str] = None,
    ):
        """
        Multi-view tactile force array encoder.
        Assumes tactile input: B,C,H,W (e.g., B,3,20,20)
        Assumes low_dim input: B,D

        Args:
            shape_meta: Dictionary containing shape information for observations
            tactile_model: Tactile encoder model or dict of models for each key
            resize_shape: Optional resize shape for tactile inputs
            use_group_norm: Replace BatchNorm with GroupNorm
            share_tactile_model: Use single model for all tactile inputs
            pretrained_path: Path to pretrained weights (e.g., 'best_model.pt')
        """
        super().__init__()

        tactile_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        # handle sharing tactile backbone
        if share_tactile_model:
            assert isinstance(tactile_model, nn.Module)
            key_model_map["tactile"] = tactile_model

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            type = attr.get("type", "low_dim")
            key_shape_map[key] = shape

            if type == "tactile":
                tactile_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_tactile_model:
                    if isinstance(tactile_model, dict):
                        # have provided model for each key
                        this_model = tactile_model[key]
                    else:
                        assert isinstance(tactile_model, nn.Module)
                        # have a copy of the tactile model
                        this_model = copy.deepcopy(tactile_model)

                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features // 16,
                                num_channels=x.num_features,
                            ),
                        )
                    key_model_map[key] = this_model

                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(size=(h, w))
                    input_shape = (shape[0], h, w)

                key_transform_map[key] = this_resizer

            elif type == "low_dim":
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        tactile_keys = sorted(tactile_keys)
        low_dim_keys = sorted(low_dim_keys)

        # Load pretrained weights
        # Default to best_model.pt in the same directory as this file
        if pretrained_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            default_path = os.path.join(current_dir, "best_model.pt")
            if os.path.exists(default_path):
                pretrained_path = default_path

        if pretrained_path is not None:
            self._load_pretrained_weights(
                pretrained_path, key_model_map, share_tactile_model
            )

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_tactile_model = share_tactile_model
        self.tactile_keys = tactile_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

    def _load_pretrained_weights(
        self, pretrained_path, key_model_map, share_tactile_model
    ):
        """Load pretrained weights from checkpoint"""
        try:
            checkpoint = torch.load(pretrained_path, map_location="cpu")

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Load weights for encoder part only
            if share_tactile_model:
                model = key_model_map["tactile"]
                # Extract encoder weights
                encoder_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("encoder."):
                        # Remove 'encoder.' prefix if present
                        new_key = key.replace("encoder.", "")
                        encoder_state_dict[new_key] = value
                    elif not key.startswith("decoder.") and not key.startswith("fc"):
                        # Include weights that don't have encoder/decoder prefix
                        encoder_state_dict[key] = value

                if encoder_state_dict:
                    model.load_state_dict(encoder_state_dict, strict=False)
                    print(
                        f"✓ Loaded pretrained tactile encoder weights from {pretrained_path}"
                    )
                else:
                    # If no encoder prefix, try loading full state dict
                    model.load_state_dict(state_dict, strict=False)
                    print(
                        f"✓ Loaded pretrained tactile model weights from {pretrained_path}"
                    )
            else:
                # Load weights for each tactile key
                for key in key_model_map.keys():
                    if key in ["tactile"] or key.startswith("tactile"):
                        model = key_model_map[key]
                        encoder_state_dict = {}
                        for state_key, value in state_dict.items():
                            if state_key.startswith("encoder."):
                                new_key = state_key.replace("encoder.", "")
                                encoder_state_dict[new_key] = value
                            elif not state_key.startswith(
                                "decoder."
                            ) and not state_key.startswith("fc"):
                                encoder_state_dict[state_key] = value

                        if encoder_state_dict:
                            model.load_state_dict(encoder_state_dict, strict=False)
                        else:
                            model.load_state_dict(state_dict, strict=False)
                        print(f"✓ Loaded pretrained weights for {key}")

        except Exception as e:
            print(
                f"⚠️  Warning: Could not load pretrained weights from {pretrained_path}: {e}"
            )

    def forward(self, obs_dict):
        batch_size = None
        features = list()

        # process tactile input
        if self.share_tactile_model:
            # pass all tactile obs to tactile model
            tactile_arrays = list()
            for key in self.tactile_keys:
                tactile = obs_dict[key]
                if batch_size is None:
                    batch_size = tactile.shape[0]
                else:
                    assert batch_size == tactile.shape[0]
                assert tactile.shape[1:] == self.key_shape_map[key]
                tactile = self.key_transform_map[key](tactile)
                tactile_arrays.append(tactile)
            # (N*B,C,H,W)
            tactile_arrays = torch.cat(tactile_arrays, dim=0)
            # (N*B,D)
            feature = self.key_model_map["tactile"](tactile_arrays)
            # (N,B,D)
            feature = feature.reshape(-1, batch_size, *feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature, 0, 1)
            # (B,N*D)
            feature = feature.reshape(batch_size, -1)
            features.append(feature)
        else:
            # run each tactile obs to independent models
            for key in self.tactile_keys:
                tactile = obs_dict[key]
                if batch_size is None:
                    batch_size = tactile.shape[0]
                else:
                    assert batch_size == tactile.shape[0]
                assert tactile.shape[1:] == self.key_shape_map[key]
                tactile = self.key_transform_map[key](tactile)
                feature = self.key_model_map[key](tactile)
                features.append(feature)

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)

        # concatenate all features
        result = torch.cat(features, dim=-1)
        return result

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta["obs"]
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            this_obs = torch.zeros(
                (batch_size,) + shape, dtype=self.dtype, device=self.device
            )
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape
