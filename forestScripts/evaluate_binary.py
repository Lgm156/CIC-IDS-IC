import numpy as np
import pandas as pd
import json
import os
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.imbalance_calibration import calibrate

def search_lambda_binary(pd_val, y_val, ps):
    num_classes = pd_val.shape[1]
    pt = np.ones(num_classes) / num_classes
    
    history = []
    def evaluate_lam(lam):
        pf = calibrate(pd_val, ps, pt, lam)
        preds = np.argmax(pf, axis=1)
        acc = np.mean(preds == y_val)
        history.append({'lambda': lam, 'accuracy': acc})
        return acc

    # Modified binary search from the paper (Algorithm 2)
    L = 0.0
    H = 2.0
    prec = 0.01
    
    # Boundary check
    if evaluate_lam(L + prec) < evaluate_lam(L):
        best_lambda = L
    else:
        # Enlarge search range if necessary
        while evaluate_lam(H) >= evaluate_lam(L):
            H += 5 * prec
            if H > 10.0:
                break
        
        def find_max_lambda(low, high):
            low_steps = int(round(low / prec))
            high_steps = int(round(high / prec))
            
            if low_steps >= high_steps:
                return low_steps * prec
            
            if high_steps == low_steps + 1:
                if evaluate_lam(low_steps * prec) >= evaluate_lam(high_steps * prec):
                    return low_steps * prec
                else:
                    return high_steps * prec
            
            mid_steps = (low_steps + high_steps) // 2
            M = mid_steps * prec
            m_val = evaluate_lam(M)
            
            # Plateau Resolution (Look-Ahead Loop)
            # If flat on the right, look further right
            step_r = prec
            while abs(evaluate_lam(M + step_r) - m_val) < 1e-9 and (M + step_r) <= high:
                step_r += prec
            m_plus_actual = evaluate_lam(M + step_r)
            
            # If flat on the left, look further left
            step_l = prec
            while abs(evaluate_lam(M - step_l) - m_val) < 1e-9 and (M - step_l) >= low:
                step_l += prec
            m_minus_actual = evaluate_lam(M - step_l)
            
            if m_val >= m_plus_actual and m_val >= m_minus_actual:
                return M  # Peak plateau center found
            elif m_minus_actual > m_val:
                return find_max_lambda(low, M - prec)
            else:
                return find_max_lambda(M + prec, high)

        best_lambda = find_max_lambda(L, H)
    
    best_acc = evaluate_lam(best_lambda)
    return best_lambda, best_acc, history

def main():
    model_name = "mlp"
    print(f"Evaluating {model_name} on Forest Cover Dataset using BINARY SEARCH for Lambda...")
    
    # Load posteriors and targets
    pd_val = np.load("models/forest/mlp/validation_posteriors.npy")
    y_val = np.load("models/forest/mlp/validation_targets.npy")
    pd_test = np.load("models/forest/mlp/test_posteriors.npy")
    y_test = np.load("models/forest/mlp/test_targets.npy")
    ps = np.load("results/forest/source_prior.npy")
    
    num_classes = pd_test.shape[1]
    pt = np.ones(num_classes) / num_classes
    
    le = joblib.load("models/forest/label_encoder.joblib")
    classes = le.classes_
    
    # Run lambda search (using the binary search method)
    best_lambda, best_val_acc, lambda_history = search_lambda_binary(pd_val, y_val, ps)
    print(f"\nBest Lambda (Binary Search) for {model_name}: {best_lambda:.4f} (Val Acc: {best_val_acc:.4f})")
    
    # Save lambda search results
    output_dir = "results/forest/mlp"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "best_lambda_binary.json"), "w") as f:
        json.dump({'best_lambda': best_lambda, 'best_validation_accuracy': best_val_acc}, f)
    pd.DataFrame(lambda_history).to_csv(os.path.join(output_dir, "lambda_search_history_binary.csv"), index=False)
    
    # Baseline Predictions (standard softmax argmax)
    preds_baseline = np.argmax(pd_test, axis=1)
    
    # Calibrated Predictions
    pf_test = calibrate(pd_test, ps, pt, best_lambda)
    preds_ic = np.argmax(pf_test, axis=1)
    
    # Compute Metrics
    def get_metrics(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        per_class_recall = precision_recall_fscore_support(y_true, y_pred, average=None)[1]
        
        metrics = {
            'Accuracy': acc,
            'Balanced Accuracy': bal_acc,
            'Macro Precision': precision,
            'Macro Recall': recall,
            'Macro F1': f1
        }
        for i, name in enumerate(classes):
            metrics[f'Recall_{name}'] = per_class_recall[i]
        return metrics

    baseline_metrics = get_metrics(y_test, preds_baseline)
    ic_metrics = get_metrics(y_test, preds_ic)
    
    df_metrics = pd.DataFrame([baseline_metrics, ic_metrics], index=['Baseline', 'IC']).T
    df_metrics.to_csv(os.path.join(output_dir, "final_metrics_comparison_binary.csv"))
    print("\nFinal Metrics Comparison (Binary Search):")
    print(df_metrics.to_string())
    
    # Confusion Matrices
    def plot_cm(y_true, y_pred, title, filename):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        
    plot_cm(y_test, preds_baseline, f"Baseline Confusion Matrix ({model_name})", "confusion_matrix_baseline_binary.png")
    plot_cm(y_test, preds_ic, f"IC Confusion Matrix ({model_name} - Binary)", "confusion_matrix_ic_binary.png")
    
    # Plot Lambda Search Curve
    df_history = pd.DataFrame(lambda_history).sort_values('lambda')
    plt.figure(figsize=(8, 6))
    plt.plot(df_history['lambda'], df_history['accuracy'], marker='o')
    plt.axvline(best_lambda, color='r', linestyle='--', label=f'Best Lambda: {best_lambda:.4f}')
    plt.title(f"Lambda Search History (Binary Search) - {model_name}")
    plt.xlabel("Lambda")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "lambda_search_curve_binary.png"))
    plt.close()
    
    print(f"\nEvaluation complete (Binary Search). Outputs saved to {output_dir}/")

if __name__ == "__main__":
    main()
