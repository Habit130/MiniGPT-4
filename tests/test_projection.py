import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "minigpt4" / "models" / "projection.py"
)
SPEC = importlib.util.spec_from_file_location("projection", MODULE_PATH)
projection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(projection)


class ProjectionCompatibilityTest(unittest.TestCase):
    def test_detects_and_loads_linear_projection(self):
        source = torch.nn.Linear(12, 8)
        state_dict = {
            "llama_proj.weight": source.weight.detach().clone(),
            "llama_proj.bias": source.bias.detach().clone(),
        }

        projection_type = projection.resolve_projection_type("auto", state_dict)
        target = projection.build_projection(projection_type, 12, 8, state_dict)
        result = target.load_state_dict(
            {
                key.removeprefix("llama_proj."): value
                for key, value in state_dict.items()
            }
        )

        self.assertEqual(projection_type, "linear")
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])

    def test_detects_and_loads_mlp_projection(self):
        source = projection.MLPProjection(12, 8, hidden_dim=10)
        state_dict = {
            f"llama_proj.{key}": value.detach().clone()
            for key, value in source.state_dict().items()
        }

        projection_type = projection.resolve_projection_type("auto", state_dict)
        target = projection.build_projection(projection_type, 12, 8, state_dict)
        result = target.load_state_dict(
            {
                key.removeprefix("llama_proj."): value
                for key, value in state_dict.items()
            }
        )

        self.assertEqual(projection_type, "mlp")
        self.assertEqual(target.fc1.out_features, 10)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])

    def test_rejects_incomplete_projection(self):
        with self.assertRaisesRegex(ValueError, "Incomplete MLP"):
            projection.infer_projection_type(
                {"llama_proj.fc1.weight": torch.empty(10, 12)}
            )

    def test_rejects_explicit_projection_mismatch(self):
        state_dict = {
            "llama_proj.weight": torch.empty(8, 12),
            "llama_proj.bias": torch.empty(8),
        }
        with self.assertRaisesRegex(ValueError, "does not match"):
            projection.resolve_projection_type("mlp", state_dict)

    def test_infers_lora_rank(self):
        state_dict = {
            "layer.q_proj.lora_A.weight": torch.empty(64, 4096),
            "layer.q_proj.lora_B.weight": torch.empty(4096, 64),
        }
        self.assertEqual(projection.infer_lora_rank(state_dict), 64)

    def test_rejects_silently_skipped_adaptation_weights(self):
        load_result = SimpleNamespace(
            missing_keys=[],
            unexpected_keys=["llama_proj.fc1.weight"],
        )
        with self.assertRaisesRegex(RuntimeError, "were not loaded"):
            projection.validate_adaptation_load(
                load_result,
                {"llama_proj.fc1.weight": torch.empty(10, 12)},
            )


if __name__ == "__main__":
    unittest.main()
