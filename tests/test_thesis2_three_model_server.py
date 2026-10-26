from pathlib import Path
import ast
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "thesis2_three_model_server"


class Thesis2ServerPackageTest(unittest.TestCase):
    def test_server_package_contains_expected_files(self):
        expected_paths = [
            "train.sh",
            "train_three_models.py",
            "requirements.txt",
            "README.md",
            "data/README.md",
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
        notebooks = list(PACKAGE_DIR.rglob("*.ipynb"))
        self.assertEqual(notebooks, [])

    def test_training_script_exposes_three_model_pipeline(self):
        script_path = PACKAGE_DIR / "train_three_models.py"
        source = script_path.read_text(encoding="utf-8")
        module = ast.parse(source)
        function_names = {
            node.name
            for node in module.body
            if isinstance(node, ast.FunctionDef)
        }

        expected_functions = {
            "load_dataset",
            "train_tfidf_logistic_regression",
            "train_transformer_model",
            "compute_metric_row",
            "write_comparison_outputs",
            "main",
        }

        self.assertTrue(expected_functions.issubset(function_names))
        self.assertIn("TF-IDF + Logistic Regression", source)
        self.assertIn("Fine-tuned BERT", source)
        self.assertIn("BioBERT", source)
        self.assertIn("Accuracy", source)
        self.assertIn("Macro Precision", source)
        self.assertIn("Macro Recall", source)
        self.assertIn("Macro F1", source)
        self.assertIn("Weighted Precision", source)
        self.assertIn("Weighted Recall", source)
        self.assertIn("Weighted F1", source)
        self.assertIn("Selected models:", source)
        self.assertIn("Starting", source)

    def test_all_model_alias_expands_to_three_models(self):
        import sys

        sys.path.insert(0, str(PACKAGE_DIR))
        import train_three_models

        self.assertEqual(
            train_three_models.normalize_model_selection(["all"]),
            ["tfidf", "bert", "biobert"],
        )

    def test_train_sh_uses_slurm_and_python_script(self):
        train_sh = (PACKAGE_DIR / "train.sh").read_text(encoding="utf-8")
        lines = train_sh.splitlines()

        self.assertEqual(lines[0], "#!/bin/bash")
        self.assertEqual(lines[1], "#SBATCH --job-name=thesis2_3models")
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
        self.assertTrue(executable_lines[0].startswith("python train_three_models.py"))
        self.assertIn("--dataset-dir data/disease-diagnosis-dataset", executable_lines[0])
        self.assertIn("--models tfidf bert biobert", executable_lines[0])

        disallowed_commands = [
            "set -euo",
            "mkdir",
            "module load",
            "source ",
            "conda ",
            "pip install",
            "if [",
        ]
        for command in disallowed_commands:
            self.assertNotIn(command, train_sh)
        self.assertNotIn("fi", executable_lines)


if __name__ == "__main__":
    unittest.main()
