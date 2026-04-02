from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from extract_complexity import analyze_complexity
from extract_history import analyze_history

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
DATA_DIR = ROOT / "data"
DEFAULT_REPOS_DIR = ROOT / "repos"

REPOS = {
    "django": "https://github.com/django/django.git",
    "transformers": "https://github.com/huggingface/transformers.git",
}


def run(cmd: list[str], cwd: Path | None = None) -> None:
    location = cwd if cwd is not None else Path.cwd()
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=location, check=True)


def ensure_git_available() -> None:
    if shutil.which("git") is None:
        raise RuntimeError("git is required but was not found in PATH")


def clone_or_update_repo(name: str, url: str, repos_dir: Path, refresh: bool) -> Path:
    repo_dir = repos_dir / name

    if repo_dir.exists():
        if not (repo_dir / ".git").exists():
            raise RuntimeError(f"{repo_dir} exists but is not a git repository")

        if refresh:
            print(f"Refreshing {name} in {repo_dir}...")
            run(["git", "pull", "--ff-only"], cwd=repo_dir)
        else:
            print(f"Using existing clone for {name}: {repo_dir}")
        return repo_dir

    print(f"Cloning {name} into {repo_dir}...")
    run(["git", "clone", url, str(repo_dir)])
    return repo_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone/update subject repositories, generate CSVs, and build plots."
    )
    parser.add_argument(
        "--repos-dir",
        default=str(DEFAULT_REPOS_DIR),
        help="Directory where the django/ and transformers/ repos should live (default: ./repos).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Run git pull --ff-only on existing clones before analysis.",
    )
    parser.add_argument(
        "--years-back",
        type=int,
        default=2,
        help="History window in years for extract_history.py logic (default: 2).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only generate CSVs; do not run generate_plots.py.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repos_dir = Path(args.repos_dir).resolve()

    ensure_git_available()
    repos_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    repo_paths: dict[str, Path] = {}
    for name, url in REPOS.items():
        repo_paths[name] = clone_or_update_repo(name, url, repos_dir, args.refresh)

    print("\nGenerating complexity CSVs...")
    analyze_complexity(
        str(repo_paths["django"]), str(DATA_DIR / "django_complexity.csv")
    )
    analyze_complexity(
        str(repo_paths["transformers"]),
        str(DATA_DIR / "transformers_complexity.csv"),
    )

    print("\nGenerating history CSVs...")
    analyze_history(
        str(repo_paths["django"]),
        str(DATA_DIR / "django_history.csv"),
        years_back=args.years_back,
    )
    analyze_history(
        str(repo_paths["transformers"]),
        str(DATA_DIR / "transformers_history.csv"),
        years_back=args.years_back,
    )

    if not args.skip_plots:
        print("\nGenerating plots...")
        run([sys.executable, str(SCRIPTS_DIR / "generate_plots.py")], cwd=ROOT)

    print("\nPipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
