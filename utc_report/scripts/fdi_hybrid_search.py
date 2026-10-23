"""
Hybrid search: VAE + statistical model + MAD threshold.

Same data, same VAE, same injected attacks as fdi_maintex_pipeline.py. Only the scoring stage
between the VAE and the threshold changes.

  VAE            trained on clean data, gives a residual vector r = x - x' and a latent mean per record
  Statistical    turns r (or the latent mean) into one anomaly score per record
  MAD threshold  flag if score > median + K * 1.4826 * MAD, with median and MAD from CLEAN scores
                 (no labels, no assumed attack share). K = 3 is the usual convention and is fixed.

Statistical models compared
  error          plain VAE reconstruction error (the standalone VAE baseline)
  robust_z       per feature robust z score of the residual (median and MAD per feature), sum of squares
  mahalanobis    Mahalanobis distance of the residual vector, robust covariance (MCD) from clean residuals
  latent_maha    Mahalanobis distance of the latent mean, robust covariance (MCD)
  ewma_error     EWMA control chart on the VAE error
  ewma_maha      EWMA control chart on the Mahalanobis residual distance
  cusum_maha     one sided CUSUM on the robust standardized Mahalanobis distance
  hampel_maha    rolling median (Hampel style) of the Mahalanobis residual distance

All calibration (medians, MADs, covariances) uses clean data only.
Seeds 0 to 4 are used to compare designs. Seeds 5 to 14 are a confirmation set that plays no part
in choosing a design.

Usage:  python fdi_hybrid_search.py [path/to/WRM_F.csv] [output_dir]
"""
import sys, os
import numpy as np, pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from fdi_maintex_pipeline import CONFIG, load_data, train_vae, inject, metrics, percentile_threshold, cv_threshold

K_MAD = 3.0            # MAD threshold multiplier (convention, not tuned)
EWMA_LAMBDA = 0.2      # usual control chart setting
CUSUM_K = 0.5          # usual CUSUM reference value
HAMPEL_W = 11          # rolling median window (records)
DEV_SEEDS, CONFIRM_SEEDS = list(range(5)), list(range(5, 15))

def robust_fit(v):
    med = np.median(v, axis=0); mad = 1.4826 * np.median(np.abs(v - med), axis=0) + 1e-12
    return med, mad

def ewma(s, lam=EWMA_LAMBDA):
    out = np.empty_like(s, dtype=float); acc = s[0]
    for i, v in enumerate(s):
        acc = lam * v + (1 - lam) * acc; out[i] = acc
    return out

def cusum(z, k=CUSUM_K):
    out = np.empty_like(z, dtype=float); c = 0.0
    for i, v in enumerate(z):
        c = max(0.0, c + v - k); out[i] = c
    return out

def rolling_median(s, w=HAMPEL_W):
    return pd.Series(s).rolling(w, center=True, min_periods=1).median().values

class Scorers:
    """Fitted on clean data, then applied to any dataset."""
    def __init__(self, vae, X_clean):
        self.vae = vae
        r, mu = self._parts(X_clean)
        self.rz = robust_fit(r)
        self.mcd_r = MinCovDet(random_state=0).fit(r)
        self.mcd_l = MinCovDet(random_state=0).fit(mu)
        self.maha_fit = robust_fit(np.sqrt(self.mcd_r.mahalanobis(r)))
    def _parts(self, X):
        mu, _ = self.vae.encode(X); mu = mu.numpy()
        return X - self.vae.dec(mu).numpy(), mu
    def score(self, X):
        r, mu = self._parts(X)
        err = (r ** 2).sum(1)
        maha = np.sqrt(self.mcd_r.mahalanobis(r))
        zm = (maha - self.maha_fit[0]) / self.maha_fit[1]
        return {
            'error': err,
            'robust_z': (((r - self.rz[0]) / self.rz[1]) ** 2).sum(1),
            'mahalanobis': maha,
            'latent_maha': np.sqrt(self.mcd_l.mahalanobis(mu)),
            'ewma_error': ewma(err),
            'ewma_maha': ewma(maha),
            'cusum_maha': cusum(zm),
            'hampel_maha': rolling_median(maha),
        }

def mad_threshold(clean_score):
    med = np.median(clean_score); mad = 1.4826 * np.median(np.abs(clean_score - med))
    return med + K_MAD * mad

def main(csv_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df, _ = load_data(csv_path); cols = list(df.columns)
    X = StandardScaler().fit_transform(df.values)
    rows = []
    for seed in DEV_SEEDS + CONFIRM_SEEDS:
        vae = train_vae(X, seed)
        sc = Scorers(vae, X)
        clean = sc.score(X)
        iso = IsolationForest(contamination=0.05, random_state=seed).fit(X)
        for level, n_win in CONFIG['windows'].items():
            Xi, y, _ = inject(X, cols, n_win, np.random.default_rng(1000 * seed + n_win))
            scores = sc.score(Xi); scores_all = {'IF': -iso.score_samples(Xi), **scores}
            clean_all = {'IF': -iso.score_samples(X), **clean}
            for name, s in scores_all.items():
                m_mad = metrics(y, (s > mad_threshold(clean_all[name])).astype(int))
                rows.append(dict(seed=seed, split='dev' if seed in DEV_SEEDS else 'confirm',
                                 level=f"{int(level*100)}%", scorer=name,
                                 auc=roc_auc_score(y, s), ap=average_precision_score(y, s),
                                 f1_mad=m_mad['f1'], prec_mad=m_mad['precision'], rec_mad=m_mad['recall'],
                                 f1_pct=metrics(y, percentile_threshold(s))['f1'],
                                 f1_cv=metrics(y, cv_threshold(s, y, seed))['f1']))
        print(f"seed {seed} done", flush=True)
    res = pd.DataFrame(rows); res.to_csv(os.path.join(out_dir, 'hybrid_search_all_seeds.csv'), index=False)
    num = ['auc', 'ap', 'f1_mad', 'prec_mad', 'rec_mad', 'f1_pct', 'f1_cv']
    summ = res.groupby(['split', 'level', 'scorer'], sort=False)[num].mean().round(3).reset_index()
    summ.to_csv(os.path.join(out_dir, 'hybrid_search_summary.csv'), index=False)
    pd.set_option('display.width', 220)
    print(summ.to_string(index=False))

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'WRM_F.csv', sys.argv[2] if len(sys.argv) > 2 else 'hybrid_search_results')
