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

import gc
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def _hf_auth_kwargs(token: Optional[str]) -> Dict[str, str]:
    if not token:
        return {}
    return {"token": token}


class SafeLoRA:
    def __init__(self, peft_model: torch.nn.Module, config):
        super().__init__()
        self.peft_model = peft_model
        self.config = config
        self.peft_config = peft_model.peft_config["default"]
        self.target_modules = set(self.peft_config.target_modules)
        self.device = torch.device(
            config.devices if config.devices == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self.original_lora_params = self._clone_lora_params()
        self.lora_modules = self._collect_lora_modules()
        self.projectors = self.get_alignment_projectors()
        self.module_metrics = self._compute_module_metrics()
        self.selected_modules = self._select_modules()
        self.model = self._apply_projection()
        sorted_metrics = sorted(
            (
                {
                    "module": module_name,
                    "selected": module_name in set(self.selected_modules),
                    **metric,
                }
                for module_name, metric in self.module_metrics.items()
            ),
            key=lambda item: item["cosine"],
        )
        self.stats = {
            "selected_modules": self.selected_modules,
            "metrics": self.module_metrics,
            "sorted_metrics": sorted_metrics,
            "num_projected_layers": len(self.selected_modules),
            "num_candidate_layers": len(self.module_metrics),
            "selection_mode": self.config.select_layers_type,
            "threshold": self.config.threshold,
            "num_proj_layers": self.config.num_proj_layers,
            "use_approximation": self.config.use_approximation,
        }

    def _clone_lora_params(self) -> Dict[str, torch.Tensor]:
        params = {}
        for name, param in self.peft_model.named_parameters():
            if ".lora_" in name and name.endswith(".weight"):
                params[name] = param.detach().clone().cpu()
        return params

    def _collect_lora_modules(self) -> Dict[str, Dict[str, str]]:
        modules: Dict[str, Dict[str, str]] = {}
        for name, _ in self.peft_model.named_parameters():
            if ".lora_A." in name and name.endswith(".weight"):
                prefix = name.split(".lora_A.", 1)[0]
                modules.setdefault(prefix, {})["A"] = name
            elif ".lora_B." in name and name.endswith(".weight"):
                prefix = name.split(".lora_B.", 1)[0]
                modules.setdefault(prefix, {})["B"] = name

        invalid = [prefix for prefix, values in modules.items() if "A" not in values or "B" not in values]
        if invalid:
            raise ValueError(f"Incomplete LoRA module pairs found: {invalid}")
        return modules

    def _is_target_weight(self, name: str) -> bool:
        if not name.endswith(".weight"):
            return False
        parts = name.split(".")
        return len(parts) >= 2 and parts[-2] in self.target_modules

    def _build_model_weight_map(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        weight_map = {}
        for name, param in model.named_parameters():
            if self._is_target_weight(name):
                weight_map[name[:-len(".weight")]] = param.detach().cpu()
        return weight_map

    def _match_projector_key(self, lora_prefix: str, projector_keys: List[str]) -> str:
        matched = [key for key in projector_keys if lora_prefix.endswith(key)]
        if len(matched) != 1:
            raise ValueError(
                f"Could not uniquely match LoRA module '{lora_prefix}' to an alignment weight. "
                f"Candidates: {matched}"
            )
        return matched[0]

    def _build_projection_matrix(self, alignment_delta: torch.Tensor) -> torch.Tensor:
        alignment_delta = alignment_delta.to(self.device, dtype=torch.float32)
        norm = torch.linalg.matrix_norm(alignment_delta, ord="fro")
        if norm.item() <= self.config.projection_eps:
            raise ValueError("Encountered a near-zero alignment delta; cannot build a projection matrix.")

        if self.config.use_approximation:
            projection = alignment_delta @ alignment_delta.T
            projection = projection / (norm + self.config.projection_eps)
        else:
            gram = alignment_delta.T @ alignment_delta
            gram_pinv = torch.linalg.pinv(gram)
            projection = alignment_delta @ gram_pinv @ alignment_delta.T

        return projection.detach().cpu()

    def get_alignment_projectors(self) -> Dict[str, torch.Tensor]:
        load_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_path,
            return_dict=True,
            torch_dtype=load_dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
            **_hf_auth_kwargs(self.config.hf_token),
        )
        aligned_model = AutoModelForCausalLM.from_pretrained(
            self.config.aligned_model_path,
            return_dict=True,
            torch_dtype=load_dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
            **_hf_auth_kwargs(self.config.hf_token),
        )

        base_weights = self._build_model_weight_map(base_model)
        projectors = {}
        for name, param in aligned_model.named_parameters():
            if not self._is_target_weight(name):
                continue

            module_name = name[:-len(".weight")]
            if module_name not in base_weights:
                raise ValueError(f"Aligned target weight '{module_name}' is missing from the base model.")

            alignment_delta = param.detach().cpu() - base_weights.pop(module_name)
            projectors[module_name] = self._build_projection_matrix(alignment_delta)

        if base_weights:
            missing = sorted(base_weights)[:5]
            raise ValueError(f"Base target weights missing from aligned model: {missing}")

        del base_model
        del aligned_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return projectors

    def _compute_module_metrics(self) -> Dict[str, Dict[str, Union[float, str]]]:
        metrics = {}
        projector_keys = list(self.projectors)

        for lora_prefix in sorted(self.lora_modules):
            projector_key = self._match_projector_key(lora_prefix, projector_keys)
            a_name = self.lora_modules[lora_prefix]["A"]
            b_name = self.lora_modules[lora_prefix]["B"]

            a_weight = self.original_lora_params[a_name].to(self.device).float()
            b_weight = self.original_lora_params[b_name].to(self.device).float()
            projector = self.projectors[projector_key].to(self.device).float()

            original_delta = b_weight @ a_weight
            projected_b = projector @ b_weight
            projected_delta = projected_b @ a_weight

            cosine = F.cosine_similarity(
                projected_delta.reshape(1, -1),
                original_delta.reshape(1, -1),
                dim=1,
                eps=self.config.projection_eps,
            ).item()

            delta_shift = torch.norm(projected_delta - original_delta).item()
            metrics[lora_prefix] = {
                "cosine": cosine,
                "projector_key": projector_key,
                "delta_shift": delta_shift,
            }

        return metrics

    def _select_modules(self) -> List[str]:
        sorted_modules = sorted(self.module_metrics, key=lambda name: self.module_metrics[name]["cosine"])
        if self.config.select_layers_type == "threshold":
            return [
                name
                for name in sorted_modules
                if self.module_metrics[name]["cosine"] < self.config.threshold
            ]

        num_layers = min(self.config.num_proj_layers, len(sorted_modules))
        return sorted_modules[:num_layers]

    def _apply_projection(self):
        selected = set(self.selected_modules)
        with torch.no_grad():
            for name, param in self.peft_model.named_parameters():
                if name not in self.original_lora_params:
                    continue

                original = self.original_lora_params[name].to(param.device, dtype=param.dtype)
                if ".lora_B." not in name or not name.endswith(".weight"):
                    param.copy_(original)
                    continue

                lora_prefix = name.split(".lora_B.", 1)[0]
                if lora_prefix not in selected:
                    param.copy_(original)
                    continue

                projector_key = self.module_metrics[lora_prefix]["projector_key"]
                projector = self.projectors[projector_key].to(param.device, dtype=torch.float32)
                projected = projector @ self.original_lora_params[name].to(param.device, dtype=torch.float32)
                param.copy_(projected.to(dtype=param.dtype))

        if self.module_metrics:
            mean_cos = sum(metric["cosine"] for metric in self.module_metrics.values()) / len(self.module_metrics)
        else:
            mean_cos = 0.0

        selection_desc = (
            f"threshold={self.config.threshold}"
            if self.config.select_layers_type == "threshold"
            else f"num_proj_layers={len(self.selected_modules)}"
        )
        print(
            f"{len(self.selected_modules)} / {len(self.module_metrics)} LoRA layers projected "
            f"using {selection_desc}; mean cosine={mean_cos:.4f}."
        )
        if self.selected_modules:
            print("Selected modules (lowest cosine first):")
            for module_name in self.selected_modules:
                metric = self.module_metrics[module_name]
                print(
                    f"  - {module_name}: cosine={metric['cosine']:.6f}, "
                    f"delta_shift={metric['delta_shift']:.6f}, projector={metric['projector_key']}"
                )
        else:
            print("No LoRA modules were selected for projection.")
        return self.peft_model
