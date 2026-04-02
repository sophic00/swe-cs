import os
import pandas as pd
from pydriller import Repository
from datetime import datetime, timedelta
from tqdm import tqdm

def analyze_history(repo_path, output_csv, years_back=2):
    print(f"Mining history for {repo_path}...")
    start_date = datetime.now() - timedelta(days=years_back * 365)
    
    # Dictionary to store file metrics
    file_metrics = {}
    
    # Keywords often found in bug-fixing commits
    bug_keywords = ['fix', 'bug', 'issue', 'resolve', 'patch', 'error']

    repo = Repository(repo_path, since=start_date)
    
    for commit in tqdm(repo.traverse_commits(), desc=f"Parsing {repo_path}"):
        is_bug_fix = any(keyword in commit.msg.lower() for keyword in bug_keywords)
        
        for modified_file in commit.modified_files:
            # We only care about Python files and want to skip tests
            if modified_file.filename.endswith('.py') and 'test' not in modified_file.new_path.lower() if modified_file.new_path else False:
                filepath = modified_file.new_path
                
                if filepath not in file_metrics:
                    file_metrics[filepath] = {"churn": 0, "bug_fixes": 0, "lines_added": 0, "lines_deleted": 0}
                
                file_metrics[filepath]["churn"] += 1
                file_metrics[filepath]["lines_added"] += modified_file.added_lines
                file_metrics[filepath]["lines_deleted"] += modified_file.deleted_lines
                
                if is_bug_fix:
                    file_metrics[filepath]["bug_fixes"] += 1

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(file_metrics, orient='index').reset_index()
    df.rename(columns={'index': 'filename'}, inplace=True)
    
    # Clean up filename paths to match the complexity script
    df['filename'] = df['filename'].apply(lambda x: x.replace('\\', '/'))
    
    df.to_csv(output_csv, index=False)
    print(f"Saved history data to {output_csv}")

if __name__ == "__main__":
    analyze_history("./django", "django_history.csv")
    analyze_history("./transformers", "transformers_history.csv")
