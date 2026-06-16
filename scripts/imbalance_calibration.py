import numpy as np

def calibrate(pd, ps, pt, lam):
    """
    Apply Imbalance Calibration.
    pd: discriminative posterior P_d(y|x)
    ps: source prior P_s(y)
    pt: target prior P_t(y)
    lam: lambda parameter
    """
    # Rebalanced Posterior: Pr = Pd * (Pt / Ps)
    pr = pd * (pt / ps)
    pr = pr / pr.sum(axis=1, keepdims=True)
    
    # Final Calibrated Posterior: Pf = Pd^(1-lam) * Pr^lam
    # Using log-space for numerical stability
    log_pd = np.log(pd + 1e-12)
    log_pr = np.log(pr + 1e-12)
    
    log_pf = (1 - lam) * log_pd + lam * log_pr
    pf = np.exp(log_pf)
    pf = pf / pf.sum(axis=1, keepdims=True)
    
    return pf
