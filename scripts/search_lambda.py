import numpy as np
import json
import pandas as pd
from scripts.imbalance_calibration import calibrate

def search_lambda():
    pd_val = np.load("results/validation_posteriors.npy")
    y_val = np.load("results/validation_targets.npy")
    ps = np.load("results/source_prior.npy")
    
    num_classes = pd_val.shape[1]
    pt = np.ones(num_classes) / num_classes
    
    history = []
    def evaluate(lam):
        pf = calibrate(pd_val, ps, pt, lam)
        preds = np.argmax(pf, axis=1)
        acc = np.mean(preds == y_val)
        history.append({'lambda': lam, 'accuracy': acc})
        return acc

    # Algorithm 2 from the paper: Modified Binary Search
    L = 0.0
    H = 2.0
    prec = 0.01 # Using the precision specified in project requirements
    
    # Boundary check
    if evaluate(L + prec) < evaluate(L):
        best_lambda = L
    else:
        # Enlarge search range if necessary to ensure optimal lambda is contained
        while evaluate(H) >= evaluate(L):
            H += 5 * prec
            if H > 10.0: # Safety break
                break
        
        # Binary search procedure
        def find_max_lambda(low, high):
            # Quantize values to precision steps
            low_steps = int(round(low / prec))
            high_steps = int(round(high / prec))
            
            if low_steps >= high_steps:
                return low_steps * prec
            
            if high_steps == low_steps + 1:
                if evaluate(low_steps * prec) >= evaluate(high_steps * prec):
                    return low_steps * prec
                else:
                    return high_steps * prec
            
            mid_steps = (low_steps + high_steps) // 2
            M = mid_steps * prec
            
            m_val = evaluate(M)
            m_plus = evaluate(M + prec)
            m_minus = evaluate(M - prec)
            
            if m_val > m_plus and m_val > m_minus:
                return M
            elif m_val > m_plus and m_val < m_minus:
                # Optimal lambda lies on the left side
                return find_max_lambda(low, M - prec)
            else:
                # Optimal lambda lies on the right side
                return find_max_lambda(M + prec, high)

        best_lambda = find_max_lambda(L, H)
    
    best_acc = evaluate(best_lambda)
    print(f"Best Lambda (Algorithm 2): {best_lambda:.4f}")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    
    with open("results/best_lambda.json", "w") as f:
        json.dump({'best_lambda': best_lambda, 'best_validation_accuracy': best_acc}, f)
        
    pd.DataFrame(history).to_csv("results/lambda_search_history.csv", index=False)
    print("Saved results/best_lambda.json and results/lambda_search_history.csv")

if __name__ == "__main__":
    search_lambda()
