import torch.nn as nn


LINEAR_PROJECTION = "linear"
MLP_PROJECTION = "mlp"
AUTO_PROJECTION = "auto"

_LINEAR_KEYS = {
    "llama_proj.weight",
    "llama_proj.bias",
}
_MLP_KEYS = {
    "llama_proj.fc1.weight",
    "llama_proj.fc1.bias",
    "llama_proj.fc2.weight",
    "llama_proj.fc2.bias",
}


class MLPProjection(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=4096, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


def checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary.")

    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint 'model' entry must be a state dictionary.")
    return state_dict


def infer_projection_type(state_dict):
    linear_keys = _LINEAR_KEYS.intersection(state_dict)
    mlp_keys = _MLP_KEYS.intersection(state_dict)

    if linear_keys and mlp_keys:
        raise ValueError("Checkpoint contains both linear and MLP projection weights.")
    if linear_keys:
        if linear_keys != _LINEAR_KEYS:
            missing = sorted(_LINEAR_KEYS - linear_keys)
            raise ValueError(f"Incomplete linear projection weights; missing: {missing}")
        return LINEAR_PROJECTION
    if mlp_keys:
        if mlp_keys != _MLP_KEYS:
            missing = sorted(_MLP_KEYS - mlp_keys)
            raise ValueError(f"Incomplete MLP projection weights; missing: {missing}")
        return MLP_PROJECTION
    return None


def resolve_projection_type(requested_type, state_dict=None):
    requested_type = (requested_type or AUTO_PROJECTION).lower()
    supported = {AUTO_PROJECTION, LINEAR_PROJECTION, MLP_PROJECTION}
    if requested_type not in supported:
        raise ValueError(
            f"Unsupported projection_type '{requested_type}'. "
            f"Expected one of {sorted(supported)}."
        )

    detected_type = infer_projection_type(state_dict) if state_dict is not None else None
    if requested_type == AUTO_PROJECTION:
        return detected_type or LINEAR_PROJECTION
    if detected_type is not None and detected_type != requested_type:
        raise ValueError(
            f"Configured projection_type '{requested_type}' does not match "
            f"checkpoint projection type '{detected_type}'."
        )
    return requested_type


def build_projection(projection_type, in_dim, out_dim, state_dict=None, dropout=0.1):
    projection_type = resolve_projection_type(projection_type, state_dict)

    if projection_type == LINEAR_PROJECTION:
        projection = nn.Linear(in_dim, out_dim)
        if state_dict is not None and "llama_proj.weight" in state_dict:
            expected = (out_dim, in_dim)
            actual = tuple(state_dict["llama_proj.weight"].shape)
            bias_shape = tuple(state_dict["llama_proj.bias"].shape)
            if actual != expected or bias_shape != (out_dim,):
                raise ValueError(
                    "Linear projection shape mismatch: "
                    f"expected weight={expected}, bias={(out_dim,)}; "
                    f"got weight={actual}, bias={bias_shape}."
                )
        return projection

    hidden_dim = 4096
    if state_dict is not None and "llama_proj.fc1.weight" in state_dict:
        fc1_shape = tuple(state_dict["llama_proj.fc1.weight"].shape)
        fc2_shape = tuple(state_dict["llama_proj.fc2.weight"].shape)
        fc1_bias_shape = tuple(state_dict["llama_proj.fc1.bias"].shape)
        fc2_bias_shape = tuple(state_dict["llama_proj.fc2.bias"].shape)
        hidden_dim = fc1_shape[0]
        expected_fc1 = (hidden_dim, in_dim)
        expected_fc2 = (out_dim, hidden_dim)
        shapes_match = (
            fc1_shape == expected_fc1
            and fc2_shape == expected_fc2
            and fc1_bias_shape == (hidden_dim,)
            and fc2_bias_shape == (out_dim,)
        )
        if not shapes_match:
            raise ValueError(
                "MLP projection shape mismatch: "
                f"expected fc1={expected_fc1}/{(hidden_dim,)}, "
                f"fc2={expected_fc2}/{(out_dim,)}; "
                f"got fc1={fc1_shape}/{fc1_bias_shape}, "
                f"fc2={fc2_shape}/{fc2_bias_shape}."
            )

    return MLPProjection(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


def infer_lora_rank(state_dict):
    ranks = set()
    for key, value in state_dict.items():
        if key.endswith("lora_A.weight"):
            ranks.add(value.shape[0])
        elif key.endswith("lora_B.weight"):
            ranks.add(value.shape[1])

    if len(ranks) > 1:
        raise ValueError(f"Checkpoint contains inconsistent LoRA ranks: {sorted(ranks)}")
    return next(iter(ranks), None)


def validate_adaptation_load(load_result, state_dict):
    critical_checkpoint_keys = {
        key
        for key in state_dict
        if key.startswith("llama_proj.") or ".lora_" in key
    }
    skipped = sorted(
        key for key in load_result.unexpected_keys if key in critical_checkpoint_keys
    )
    if skipped:
        raise RuntimeError(
            "Checkpoint adaptation weights were not loaded: " + ", ".join(skipped)
        )
