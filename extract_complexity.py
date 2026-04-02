import os
import lizard
import pandas as pd

def analyze_complexity(repo_path, output_csv):
    print(f"Analyzing complexity for {repo_path}...")
    
    # Run lizard on the repo, ignoring tests and docs
    analysis = lizard.analyze([repo_path], exclude_pattern=["*/tests/*", "*/docs/*", "*/test_*"])
    
    data = []
    for file_info in analysis:
        # Average cyclomatic complexity of all functions in the file
        avg_ccn = file_info.average_cyclomatic_complexity
        
        data.append({
            "filename": os.path.relpath(file_info.filename, repo_path),
            "nloc": file_info.nloc,
            "token_count": file_info.token_count,
            "avg_complexity": avg_ccn,
            "num_functions": len(file_info.function_list)
        })
        
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} files to {output_csv}")

if __name__ == "__main__":
    analyze_complexity("./django", "django_complexity.csv")
    analyze_complexity("./transformers", "transformers_complexity.csv")
