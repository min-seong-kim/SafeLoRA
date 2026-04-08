# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SafeLoRAConfig:
    """
    This is the configuration class to store the configuration of a safeLoRA.
    """

    base_model_path: str = field(
        default=None,
        metadata={"help": "The path of the base model for obtaining the aligned matrix"},
    )

    aligned_model_path: str = field(
        default=None,
        metadata={"help": "The path of the aligned model for obtaining the aligned matrix"},
    )

    hf_token: Optional[str] = field(
        default=None,
        metadata={"help": "Optional Hugging Face token for gated/private checkpoints."},
    )

    select_layers_type: str = field(
        default="number",
        metadata={"help": "How to select projection layers? options: [threshold, number]"},
    )

    threshold: float = field(
        default=0.5,
        metadata={"help": "The threshold of cosine similarity."},
    )

    num_proj_layers: int = field(
        default=10,
        metadata={"help": "The number of projected layers."},
    )

    devices: str = field(
        default="cuda",
        metadata = {"help": "Devices are used in SafeLoRA. (gpu or cpu)"}

    )

    use_approximation: bool = field(
        default=True,
        metadata={"help": "Use the fast approximation C = VV^T / ||V||_F instead of the exact projector."},
    )

    projection_eps: float = field(
        default=1e-8,
        metadata={"help": "Numerical stability epsilon for projection and cosine similarity."},
    )

    def __post_init__(self):
        if self.base_model_path is None:
            raise ValueError("base_model_path cannot be None.")
        if self.aligned_model_path is None:
            raise ValueError("aligned_model_path cannot be None.")
        if self.select_layers_type not in {"threshold", "number"}:
            raise ValueError("select_layers_type must be either 'threshold' or 'number'.")
        if self.select_layers_type == "number" and self.num_proj_layers < 0:
            raise ValueError("num_proj_layers must be non-negative.")
