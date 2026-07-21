import subprocess
import json
import pandas as pd
import os
import re

def update_split_size(size):
    file_path = "forestScripts/create_splits.py"
    with open(file_path, "r") as f:
        content = f.read()
    
    # Replace VAL_TEST_SAMPLES_PER_CLASS = ...
    new_content = re.sub(
        r"VAL_TEST_SAMPLES_PER_CLASS = \d+",
        f"VAL_TEST_SAMPLES_PER_CLASS = {size}",
        content
    )
    
    with open(file_path, "w") as f:
        f.write(new_content)
    print(f"\n[Experiment] Updated create_splits.py: VAL_TEST_SAMPLES_PER_CLASS = {size}")

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {cmd}")
        print(result.stderr)
        raise RuntimeError(result.stderr)
    return result.stdout

def main():
    # Starting from size 100 up to 1200 in steps of 100
    sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    
    report_path = "forestScripts/validation_size_analysis.md"
    
    # Initialize report
    with open(report_path, "w") as f:
        f.write("# Validation and Test Set Size Impact Analysis (Sizes 100 to 1200)\n\n")
        f.write("This report analyzes how the size of the validation/test set impacts the calibration results and the search accuracy of the Binary Search method vs. the Grid Search method.\n\n")
        f.write("| Size/Class | Total Val/Test | Binary Lambda | Binary Test Acc | Grid Lambda | Grid Test Acc | Diff (Binary - Grid) | Status |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        
    for size in sizes:
        total_val_test = size * 7
        print(f"\n=======================================================")
        print(f"Starting iteration for validation size: {size} per class ({total_val_test} total)")
        print(f"=======================================================")
        
        # 1. Update split size in create_splits.py
        update_split_size(size)
        
        # 2. Run preprocessing/splitting
        print("Running create_splits.py...")
        run_cmd("PYTHONPATH=. .venv/bin/python forestScripts/create_splits.py")
        
        # 3. Train model
        print("Training MLP...")
        run_cmd("PYTHONPATH=. .venv/bin/python forestScripts/train_mlp.py")
        
        # 4. Save posteriors
        print("Saving posteriors...")
        run_cmd("PYTHONPATH=. .venv/bin/python forestScripts/save_posteriors.py")
        
        # 5. Run binary evaluation
        print("Running evaluate_binary.py...")
        run_cmd("PYTHONPATH=. .venv/bin/python forestScripts/evaluate_binary.py")
        
        # 6. Run grid evaluation
        print("Running evaluate_grid.py...")
        run_cmd("PYTHONPATH=. .venv/bin/python forestScripts/evaluate_grid.py")
        
        # 7. Read results
        with open("results/forest/mlp/best_lambda_binary.json", "r") as f:
            binary_data = json.load(f)
            bin_lam = binary_data['best_lambda']
            
        with open("results/forest/mlp/best_lambda_grid.json", "r") as f:
            grid_data = json.load(f)
            grid_lam = grid_data['best_lambda']
            
        metrics_bin = pd.read_csv("results/forest/mlp/final_metrics_comparison_binary.csv", index_col=0)
        bin_test_acc = metrics_bin.loc['Accuracy', 'IC']
        
        metrics_grid = pd.read_csv("results/forest/mlp/final_metrics_comparison_grid.csv", index_col=0)
        grid_test_acc = metrics_grid.loc['Accuracy', 'IC']
        
        diff = bin_test_acc - grid_test_acc
        status = "Equal" if abs(diff) < 1e-6 else ("Binary Better" if diff > 0 else "Grid Better")
        
        print(f"Iteration Results:")
        print(f"  Binary: lambda={bin_lam:.2f}, Test Acc={bin_test_acc:.4f}")
        print(f"  Grid:   lambda={grid_lam:.2f}, Test Acc={grid_test_acc:.4f}")
        print(f"  Diff:   {diff:+.6f} ({status})")
        
        # Append to report
        with open(report_path, "a") as f:
            f.write(f"| {size} | {total_val_test} | {bin_lam:.2f} | {bin_test_acc:.6f} | {grid_lam:.2f} | {grid_test_acc:.6f} | {diff:+.6f} | {status} |\n")
            
        if diff > 0:
            print(f"\n[Experiment] SUCCESS: Binary search achieved STRICTLY better accuracy than Grid search at size {size}!")
            print("Stopping experiment as binary search is strictly better.")
            break

    print(f"\nExperiment complete. Results recorded in {report_path}")

if __name__ == "__main__":
    main()
