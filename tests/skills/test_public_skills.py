# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static contract tests for the public CodonFM skills.

These tests intentionally use only the Python standard library so command
usage can be checked before installing the GPU runtime.
"""

import argparse
import ast
import json
import re
import shlex
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = REPO_ROOT / "skills"
PUBLIC_SKILLS = {
    "codonfm-setup",
    "codonfm-score",
    "codonfm-embed",
    "codonfm-finetune",
}


def _frontmatter(skill_text: str) -> dict[str, str]:
    match = re.match(r"\A---\n(.*?)\n---\n", skill_text, re.DOTALL)
    if match is None:
        raise AssertionError("SKILL.md is missing YAML frontmatter")
    result = {}
    for line in match.group(1).splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            raise AssertionError(f"Invalid frontmatter line: {line}")
        result[key.strip()] = value.strip()
    return result


def _runner_commands(skill_text: str) -> list[list[str]]:
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", skill_text, re.DOTALL):
        normalized = block.replace("\\\n", " ")
        tokens = shlex.split(normalized, comments=True)
        for index in range(len(tokens) - 3):
            if tokens[index:index + 3] == ["python", "-m", "src.runner"]:
                commands.append(tokens[index + 3:])
                break
    return commands


def _public_runner_parser() -> argparse.ArgumentParser:
    """Build the checked-in parser without importing the runner's GPU deps."""
    tree = ast.parse((REPO_ROOT / "src/runner.py").read_text())
    get_parser = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_parser"
    )
    parser_module = ast.Module(body=[get_parser], type_ignores=[])
    namespace = {"argparse": argparse}
    exec(compile(parser_module, "src/runner.py", "exec"), namespace)
    return namespace["get_parser"]()


def _evaluate_function(namespace):
    """Load only tasks.evaluate so it can be tested without GPU packages."""
    tree = ast.parse((REPO_ROOT / "src/tasks.py").read_text())
    evaluate = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate"
    )
    module = ast.Module(body=[evaluate], type_ignores=[])
    exec(compile(module, "src/tasks.py", "exec"), namespace)
    return namespace["evaluate"]


class PublicSkillContractTests(unittest.TestCase):
    def test_expected_public_skill_set(self):
        actual = {
            path.name
            for path in SKILLS_ROOT.iterdir()
            if path.is_dir() and (path / "SKILL.md").exists()
        }
        self.assertTrue(PUBLIC_SKILLS.issubset(actual))
        self.assertFalse((SKILLS_ROOT / "codonfm-optimize").exists())

    def test_frontmatter_and_ui_metadata(self):
        for skill_name in PUBLIC_SKILLS:
            with self.subTest(skill=skill_name):
                skill_dir = SKILLS_ROOT / skill_name
                text = (skill_dir / "SKILL.md").read_text()
                metadata = _frontmatter(text)
                self.assertEqual(set(metadata), {"name", "description"})
                self.assertEqual(metadata["name"], skill_name)
                self.assertNotIn("TODO", text)

                ui = (skill_dir / "agents/openai.yaml").read_text()
                self.assertIn("display_name:", ui)
                self.assertIn("short_description:", ui)
                self.assertIn(f"${skill_name}", ui)

    def test_evals_are_valid_and_named(self):
        for skill_name in PUBLIC_SKILLS:
            with self.subTest(skill=skill_name):
                eval_path = SKILLS_ROOT / skill_name / "evals/evals.json"
                payload = json.loads(eval_path.read_text())
                self.assertEqual(payload["skill_name"], skill_name)
                self.assertGreater(len(payload["evals"]), 0)
                ids = [case["id"] for case in payload["evals"]]
                self.assertEqual(len(ids), len(set(ids)))

    def test_documented_runner_commands_parse(self):
        parser = _public_runner_parser()
        commands = []
        for skill_name in PUBLIC_SKILLS:
            text = (SKILLS_ROOT / skill_name / "SKILL.md").read_text()
            commands.extend((skill_name, command) for command in _runner_commands(text))

        self.assertEqual({name for name, _ in commands}, {
            "codonfm-score",
            "codonfm-embed",
            "codonfm-finetune",
        })
        for skill_name, command in commands:
            with self.subTest(skill=skill_name):
                parser.parse_args(command)

    def test_fragile_command_requirements(self):
        score = _runner_commands(
            (SKILLS_ROOT / "codonfm-score/SKILL.md").read_text()
        )[0]
        self.assertIn("--mask_mutation", score)
        self.assertIn("--extract-seq", score)
        self.assertIn("--dryrun", score)
        self.assertEqual(score[score.index("--num_gpus") + 1], "1")

        embed = _runner_commands(
            (SKILLS_ROOT / "codonfm-embed/SKILL.md").read_text()
        )[0]
        self.assertIn("--dryrun", embed)
        self.assertEqual(embed[embed.index("--num_gpus") + 1], "1")

        finetune = _runner_commands(
            (SKILLS_ROOT / "codonfm-finetune/SKILL.md").read_text()
        )[0]
        self.assertIn("--pretrained_ckpt_path", finetune)
        self.assertNotIn("--checkpoint_path", finetune)
        for flag in (
            "--lr",
            "--check_val_every_n_epoch",
            "--checkpoints_dir",
            "--use_downstream_head",
            "--dryrun",
        ):
            self.assertIn(flag, finetune)
        self.assertEqual(
            finetune[finetune.index("--check_val_every_n_epoch") + 1], "1"
        )

    def test_no_unavailable_feature_in_runner_commands(self):
        forbidden = {
            "MissenseDataset",
            "missense_prediction",
            "missense_synom_agg",
            "missense_inference",
            "missense_seq",
            "decodon_200m",
            "decodon_1b",
            "mutation_pred_clm",
        }
        for skill_name in PUBLIC_SKILLS:
            text = (SKILLS_ROOT / skill_name / "SKILL.md").read_text()
            for command in _runner_commands(text):
                with self.subTest(skill=skill_name, command=command):
                    self.assertTrue(forbidden.isdisjoint(command))

    def test_setup_usage_matches_public_script(self):
        text = (SKILLS_ROOT / "codonfm-setup/SKILL.md").read_text()
        self.assertIn("bash run_dev.sh", text)
        self.assertIn("--data-dir", text)
        self.assertIn("--checkpoints-dir", text)
        self.assertIn("/data/checkpoints", text)
        self.assertIn("hf download nvidia/NV-CodonFM-Encodon-1B-v1", text)
        self.assertIn("python3.11 -m venv .venv", text)
        self.assertIn("python -m pip install -r requirements.txt", text)
        self.assertIn("export MPLCONFIGDIR=", text)
        self.assertIn("Use only the checked-in public code", text)

    def test_safetensors_eval_is_not_loaded_as_a_lightning_checkpoint(self):
        calls = {"torch_load": 0, "predict": 0}

        class FakeTorch:
            @staticmethod
            def load(*args, **kwargs):
                calls["torch_load"] += 1
                raise AssertionError("torch.load must not read safetensors")

        class FakeLogger:
            def log_hyperparams(self, config):
                self.config = config

        class FakeData:
            init_global_step = 0

            def setup(self, stage):
                self.stage = stage

            def load_state_dict(self, state):
                self.state = state

        class FakeModel:
            prediction_counter = 0

            def configure_model(self):
                self.configured = True

        class FakeTrainer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def predict(self, *args, **kwargs):
                calls["predict"] += 1

        namespace = {
            "Any": object,
            "Dict": dict,
            "Path": Path,
            "Trainer": FakeTrainer,
            "logging": type("Logging", (), {"info": staticmethod(lambda message: None)}),
            "os": __import__("os"),
            "seed_everything": lambda *args, **kwargs: None,
            "torch": FakeTorch,
        }
        evaluate = _evaluate_function(namespace)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.safetensors"
            model_path.touch()
            data = FakeData()
            model = FakeModel()
            evaluate(
                config={
                    "log": FakeLogger(),
                    "data": data,
                    "trainer": {},
                    "model": model,
                    "callbacks": {},
                },
                config_dict={},
                model_ckpt_path=str(model_path),
                out_dir=temp_dir,
            )

        self.assertEqual(calls["torch_load"], 0)
        self.assertEqual(calls["predict"], 1)
        self.assertEqual(data.stage, "test")
        self.assertTrue(model.configured)

    def test_checked_in_references_exist(self):
        for relative_path in (
            "notebooks/4-EnCodon-Downstream-Task-riboNN.ipynb",
            "notebooks/5-EnCodon-Downstream-Task-mRFP-expression.ipynb",
            "notebooks/6-EnCodon-Downstream-Task-mRNA-stability.ipynb",
            "src/data/codon_bert_dataset.py",
            "src/data/mutation_dataset.py",
            "src/inference/encodon.py",
        ):
            self.assertTrue((REPO_ROOT / relative_path).is_file(), relative_path)


if __name__ == "__main__":
    unittest.main()
