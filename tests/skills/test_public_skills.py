# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static contract tests for the public CodonFM skills.

These tests intentionally use only the Python standard library so command
usage can be checked before installing the GPU runtime.
"""

import argparse
import ast
import csv
import hashlib
import importlib.util
import io
import json
import re
import shlex
import ssl
import subprocess
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


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
        if line.startswith(" "):
            continue
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


def _ribonn_helper():
    path = SKILLS_ROOT / "codonfm-finetune/scripts/prepare_ribonn.py"
    spec = importlib.util.spec_from_file_location("prepare_ribonn", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
                self.assertEqual(set(metadata), {"name", "description", "metadata"})
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

    def test_eval_inputs_exist_and_source_fixtures_match_repository(self):
        expected_archive = None
        for skill_name in sorted(PUBLIC_SKILLS):
            evals = SKILLS_ROOT / skill_name / "evals"
            payload = json.loads((evals / "evals.json").read_text())
            for case in payload["evals"]:
                for relative in case.get("files", []):
                    path = (evals / relative).resolve()
                    self.assertTrue(path.is_relative_to(evals.resolve()))
                    self.assertTrue(path.is_file(), str(path))
            archive_path = evals / "files/codonfm_source.zip"
            data = archive_path.read_bytes()
            if expected_archive is not None:
                self.assertEqual(data, expected_archive, "Trials must receive identical public source")
            expected_archive = data
            with zipfile.ZipFile(archive_path) as archive:
                manifest = json.loads(archive.read("source-manifest.json"))
                for name, sha in manifest["sha256"].items():
                    self.assertEqual(archive.read(name), (REPO_ROOT / name).read_bytes(),
                                     "Refresh source fixtures with python skills/stage_eval_context.py")
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), sha)
                self.assertFalse(any(name.startswith("skills/") or name.endswith(".safetensors")
                                     for name in archive.namelist()))

    def test_ribonn_preparation_preserves_data_without_runtime_dependencies(self):
        files = SKILLS_ROOT / "codonfm-finetune/evals/files"
        helper = SKILLS_ROOT / "codonfm-finetune/scripts/prepare_ribonn.py"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "prepared.csv"
            result = subprocess.run([sys.executable, "-S", str(helper), "--input",
                                     str(files / "ribonn_smoke.tsv"), "--output", str(output)],
                                    cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["split_counts"], {"train": 8, "val": 2, "test": 2})
            self.assertEqual(json.loads(output.with_suffix(".metadata.json").read_text()), report)
            with (files / "ribonn_smoke.tsv").open() as handle:
                raw = {row["transcript_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
            with output.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 12)
            for row in rows:
                source = raw[row["id"]]
                start, length = int(source["utr5_size"]), int(source["cds_size"])
                self.assertEqual(row["ref_seq"], source["tx_sequence"][start:start + length])
                self.assertEqual(float(row["value"]), float(source["mean_te"]))
                fold = int(source["fold"])
                self.assertEqual(row["split"], "val" if fold == 8 else "test" if fold == 9 else "train")

    def test_ribonn_download_requires_direct_https_success(self):
        helper = _ribonn_helper()
        fixture = (SKILLS_ROOT / "codonfm-finetune/evals/files/ribonn_smoke.tsv").read_bytes()
        for status in (200, 301, 302, 303, 307, 308, 404, 500):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "prepared.csv"
                response = io.BytesIO(fixture)
                response.status = status
                with (
                    patch.object(helper.http.client, "HTTPSConnection") as https,
                    patch.object(sys, "argv", ["prepare_ribonn.py", "--output", str(output)]),
                    redirect_stdout(io.StringIO()) as stdout,
                    redirect_stderr(io.StringIO()) as stderr,
                ):
                    connection = https.return_value
                    connection.getresponse.return_value = response
                    if status == 200:
                        helper.main()
                        report = json.loads(stdout.getvalue())
                        self.assertEqual(report["split_counts"], {"train": 8, "val": 2, "test": 2})
                        self.assertEqual(report["source"], helper.DATA_URL)
                        self.assertTrue(output.is_file())
                    else:
                        with self.assertRaises(SystemExit) as failure:
                            helper.main()
                        self.assertEqual(failure.exception.code, 1)
                        self.assertIn(f"HTTP {status}", stderr.getvalue())
                        self.assertEqual(list(Path(tmp).iterdir()), [])
                    https.assert_called_once_with("raw.githubusercontent.com", timeout=10)
                    connection.request.assert_called_once_with("GET", helper.DATA_PATH)
                    connection.close.assert_called_once_with()
                    self.assertTrue(response.closed)

    def test_ribonn_download_handles_transport_errors_without_output(self):
        helper = _ribonn_helper()
        errors = (TimeoutError("read timed out"), ssl.SSLCertVerificationError("untrusted certificate"),
                  helper.http.client.BadStatusLine("invalid HTTP response"))
        for error in errors:
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "prepared.csv"
                with (
                    patch.object(helper.http.client, "HTTPSConnection") as https,
                    patch.object(sys, "argv", ["prepare_ribonn.py", "--output", str(output)]),
                    redirect_stderr(io.StringIO()) as stderr,
                ):
                    https.return_value.getresponse.side_effect = error
                    with self.assertRaises(SystemExit) as failure:
                        helper.main()
                    self.assertEqual(failure.exception.code, 1)
                    self.assertIn("Preparation failed:", stderr.getvalue())
                    self.assertEqual(list(Path(tmp).iterdir()), [])
                    https.assert_called_once()
                    https.return_value.close.assert_called_once_with()

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
        self.assertEqual(score[score.index("--num_gpus") + 1], "1")

        embed = _runner_commands(
            (SKILLS_ROOT / "codonfm-embed/SKILL.md").read_text()
        )[0]
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
