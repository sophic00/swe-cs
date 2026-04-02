from pathlib import Path

import lizard
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPOS_DIR = ROOT / "repos"
DATA_DIR = ROOT / "data"


def analyze_complexity(repo_path, output_csv):
    repo_path = Path(repo_path).resolve()
    output_csv = Path(output_csv)
    print(f"Analyzing complexity for {repo_path}...")

    # Run lizard on the repo, ignoring tests and docs
    analysis = lizard.analyze(
        [str(repo_path)], exclude_pattern=["*/tests/*", "*/docs/*", "*/test_*"]
    )

    data = []
    for file_info in analysis:
        # Average cyclomatic complexity of all functions in the file
        avg_ccn = file_info.average_cyclomatic_complexity

        data.append(
            {
                "filename": str(
                    Path(file_info.filename).resolve().relative_to(repo_path)
                ),
                "nloc": file_info.nloc,
                "token_count": file_info.token_count,
                "avg_complexity": avg_ccn,
                "num_functions": len(file_info.function_list),
            }
        )

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} files to {output_csv}")


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    analyze_complexity(REPOS_DIR / "django", DATA_DIR / "django_complexity.csv")
    analyze_complexity(
        REPOS_DIR / "transformers", DATA_DIR / "transformers_complexity.csv"
    )
