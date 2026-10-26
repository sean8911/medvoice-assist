from pathlib import Path
import ast
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "thesis2_biobert_server"


class Thesis2BioBertServerPackageTest(unittest.TestCase):
    def test_biobert_package_contains_expected_files(self):
        expected_paths = [
            "train.sh",
            "train_biobert_and_compare.py",
            "requirements.txt",
            "README.md",
            "data/README.md",
            "baseline_outputs/tfidf_logistic_regression_classification_report.csv",
            "baseline_outputs/fine_tuned_bert_classification_report.csv",
            "logs/.gitkeep",
            "outputs/.gitkeep",
        ]

        missing = [
            relative_path
            for relative_path in expected_paths
            if not (PACKAGE_DIR / relative_path).exists()
        ]

        self.assertEqual(missing, [])

    def test_package_does_not_contain_notebooks(self):
        self.assertEqual(list(PACKAGE_DIR.rglob("*.ipynb")), [])

    def test_train_sh_is_simple_slurm_script_for_biobert_only(self):
        train_sh = (PACKAGE_DIR / "train.sh").read_text(encoding="utf-8")
        lines = train_sh.splitlines()

        self.assertEqual(lines[0], "#!/bin/bash")
        self.assertEqual(lines[1], "#SBATCH --job-name=thesis2_biobert")
        self.assertEqual(lines[2], "#SBATCH --output=logs/out_%j.txt")
        self.assertEqual(lines[3], "#SBATCH --error=logs/error_%j.txt")
        self.assertEqual(lines[4], "#SBATCH --gres=gpu:1")

        executable_lines = [
            line for line in lines
            if line.strip()
            and not line.startswith("#!")
            and not line.startswith("#SBATCH")
        ]
        self.assertEqual(len(executable_lines), 1)
        self.assertTrue(
            executable_lines[0].startswith("python train_biobert_and_compare.py")
        )
        self.assertIn("--dataset-dir data/disease-diagnosis-dataset", executable_lines[0])
        self.assertIn("--baseline-dir baseline_outputs", executable_lines[0])
        self.assertIn("--epochs 1", executable_lines[0])
        self.assertIn("--save-steps 1000", executable_lines[0])
        self.assertIn("--resume-from-checkpoint auto", executable_lines[0])

        for command in ["set -euo", "mkdir", "module load", "source ", "conda ", "pip install", "if ["]:
            self.assertNotIn(command, train_sh)

    def test_biobert_script_trains_biobert_and_builds_three_model_comparison(self):
        script_path = PACKAGE_DIR / "train_biobert_and_compare.py"
        source = script_path.read_text(encoding="utf-8")
        module = ast.parse(source)
        function_names = {
            node.name
            for node in module.body
            if isinstance(node, ast.FunctionDef)
        }

        expected_functions = {
            "require_torch_26_or_newer",
            "load_dataset",
            "resolve_resume_checkpoint",
            "train_biobert_model",
            "report_to_metric_row",
            "write_three_model_comparison",
            "main",
        }

        self.assertTrue(expected_functions.issubset(function_names))
        self.assertIn("BioBERT", source)
        self.assertIn("TF-IDF + Logistic Regression", source)
        self.assertIn("Fine-tuned BERT", source)
        self.assertIn("three_model_comparison.csv", source)
        self.assertIn("starting_from_base_model", source)
        self.assertIn("model_source", source)

    def test_requirements_allow_checkpoint_resume_with_torch_25_or_newer(self):
        requirements = (PACKAGE_DIR / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("torch>=2.5", requirements)

    def test_auto_resume_selects_highest_numbered_checkpoint(self):
        import shutil
        import sys

        sys.path.insert(0, str(PACKAGE_DIR))
        import train_biobert_and_compare

        checkpoint_root = PACKAGE_DIR / "outputs" / "test_checkpoints"
        if checkpoint_root.exists():
            shutil.rmtree(checkpoint_root)
        (checkpoint_root / "checkpoint-1000").mkdir(parents=True)
        (checkpoint_root / "checkpoint-3000").mkdir(parents=True)
        (checkpoint_root / "checkpoint-2000").mkdir(parents=True)

        selected = train_biobert_and_compare.resolve_resume_checkpoint(
            checkpoint_root,
            "auto",
        )

        self.assertTrue(selected.endswith("checkpoint-3000"))
        shutil.rmtree(checkpoint_root)


if __name__ == "__main__":
    unittest.main()
