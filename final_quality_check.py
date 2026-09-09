from __future__ import annotations

from pathlib import Path
import sys

try:
    import joblib
except ImportError:
    print("[FAIL] joblib is not installed.")
    print("Run: pip install -r requirements.txt")
    raise SystemExit(1)


PROJECT_ROOT = Path(__file__).resolve().parent
APP_DIR = PROJECT_ROOT / "app"
PAGES_DIR = APP_DIR / "pages"
MODELS_DIR = PROJECT_ROOT / "models"

EXCLUDED_DIRS = {
    ".venv",
    "venv",
    "env",
    ".git",
    "__pycache__",
    ".idea",
    ".vscode",
    "site-packages",
}

EXPECTED_PYTHON_FILES = [
    APP_DIR / "main.py",
    PAGES_DIR / "1_Diabetes.py",
    PAGES_DIR / "2_Heart_Disease.py",
    PAGES_DIR / "3_Kidney_Disease.py",
    PAGES_DIR / "4_Liver_Disease.py",
    PAGES_DIR / "5_Respiratory_Disease.py",
    PAGES_DIR / "6_Cancer_Disease.py",
]

EXPECTED_MODELS = [
    MODELS_DIR / "diabetes_model.joblib",
    MODELS_DIR / "heart_disease_model.joblib",
    MODELS_DIR / "kidney_disease_model.joblib",
    MODELS_DIR / "liver_disease_model.joblib",
    MODELS_DIR / "respiratory_disease_model.joblib",
    MODELS_DIR / "cancer_disease_model.joblib",
]

FAILURES: list[str] = []
WARNINGS: list[str] = []


def pass_msg(message: str) -> None:
    print(f"[PASS] {message}")


def warn_msg(message: str) -> None:
    WARNINGS.append(message)
    print(f"[WARN] {message}")


def fail_msg(message: str) -> None:
    FAILURES.append(message)
    print(f"[FAIL] {message}")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def check_required_files() -> None:
    print("\n=== 1. Required Streamlit files ===")
    for path in EXPECTED_PYTHON_FILES:
        if path.exists():
            pass_msg(rel(path))
        else:
            fail_msg(f"Missing required file: {rel(path)}")

    print("\n=== 2. Required trained models ===")
    for path in EXPECTED_MODELS:
        if path.exists():
            pass_msg(rel(path))
        else:
            fail_msg(f"Missing required model: {rel(path)}")


def check_python_syntax() -> None:
    print("\n=== 3. Python syntax ===")

    for path in EXPECTED_PYTHON_FILES:
        if not path.exists():
            continue

        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
            pass_msg(rel(path))
        except Exception as exc:
            fail_msg(f"Syntax error in {rel(path)}: {exc}")


def check_runtime_imports() -> None:
    print("\n=== 4. Runtime packages ===")

    packages = {
        "streamlit": "streamlit",
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "joblib": "joblib",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "scipy": "scipy",
    }

    for import_name, package_name in packages.items():
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            pass_msg(f"{package_name} ({version})")
        except Exception as exc:
            fail_msg(f"Package import failed: {package_name} -> {exc}")


def check_model_artifacts() -> None:
    print("\n=== 5. Model artifacts ===")

    for path in EXPECTED_MODELS:
        if not path.exists():
            continue

        try:
            artifact = joblib.load(path)

            if isinstance(artifact, dict):
                model = artifact.get("model")

                if model is None:
                    fail_msg(f"{path.name}: artifact has no 'model' key")
                    continue

                if not hasattr(model, "predict"):
                    fail_msg(f"{path.name}: saved model has no predict() method")
                    continue

                if "features" in artifact:
                    features = artifact["features"]
                    if not isinstance(features, (list, tuple)) or not features:
                        fail_msg(f"{path.name}: invalid features metadata")
                        continue

                if "threshold" in artifact:
                    try:
                        threshold = float(artifact["threshold"])
                    except Exception:
                        fail_msg(f"{path.name}: threshold is not numeric")
                        continue

                    if not 0.0 <= threshold <= 1.0:
                        fail_msg(
                            f"{path.name}: threshold {threshold} is outside [0, 1]"
                        )
                        continue

                pass_msg(f"{path.name} loaded successfully")

            elif hasattr(artifact, "predict"):
                pass_msg(f"{path.name} loaded as direct estimator")
            else:
                fail_msg(
                    f"{path.name}: unsupported saved object type "
                    f"{type(artifact).__name__}"
                )

        except Exception as exc:
            fail_msg(f"{path.name}: could not load -> {exc}")


def is_excluded(path: Path) -> bool:
    return any(part.lower() in EXCLUDED_DIRS for part in path.parts)


def check_release_hygiene() -> None:
    print("\n=== 6. Release hygiene ===")

    main_candidates = []

    for path in PROJECT_ROOT.rglob("*.py"):
        if is_excluded(path):
            continue

        if path.resolve() == (APP_DIR / "main.py").resolve():
            continue

        if "main" in path.stem.lower():
            main_candidates.append(path)

    if main_candidates:
        warn_msg(
            "Extra main-like Python files were found outside the virtual environment."
        )
        for path in main_candidates:
            print(f"       - {rel(path)}")
    else:
        pass_msg("No duplicate main-like project files found")


def print_summary() -> int:
    print("\n" + "=" * 76)
    print("FINAL QUALITY SUMMARY")
    print("=" * 76)

    if FAILURES:
        print(f"\nBlocking issues: {len(FAILURES)}")
        for index, issue in enumerate(FAILURES, start=1):
            print(f"  {index}. {issue}")
    else:
        print("\nBlocking issues: 0")

    if WARNINGS:
        print(f"\nWarnings: {len(WARNINGS)}")
        for index, issue in enumerate(WARNINGS, start=1):
            print(f"  {index}. {issue}")
    else:
        print("\nWarnings: 0")

    print("\n" + "-" * 76)

    if not FAILURES:
        print("FINAL RESULT: PASS")
        print(
            "Core structure, Python syntax, dependencies, and saved models passed."
        )
        print(
            "Next step: manually test all six Streamlit pages end-to-end."
        )
        return 0

    print(f"FINAL RESULT: FAIL — {len(FAILURES)} blocking issue(s) found.")
    print("Fix the numbered issues shown above and run this script again.")
    return 1


def main() -> int:
    print("=" * 76)
    print("MULTI-DISEASE ML SYSTEM — FINAL QUALITY CHECK v2")
    print("=" * 76)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {sys.version.split()[0]}")

    check_required_files()
    check_python_syntax()
    check_runtime_imports()
    check_model_artifacts()
    check_release_hygiene()

    return print_summary()


if __name__ == "__main__":
    raise SystemExit(main())
