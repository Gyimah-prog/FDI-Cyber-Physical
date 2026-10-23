"""
Route 1: attacks injected on the RAW measured values, then standardized with the scaler
fitted on clean data. Everything else follows fdi_maintex_pipeline.py and fdi_hybrid_search.py.

Two attack designs are run. Both use the three techniques of main.tex on Gage, Warp62, Crosslevel,
one technique per 46 record window.

  raw        spike     value +/- (2 to 3) x feature standard deviation
             multiply  value x 1.5 or x 2                      (as in the notebooks)
             noise     value + Gaussian noise, sd = 3 x feature standard deviation
  deviation  same spike and noise, but multiply scales the DEVIATION from the feature median
             by 1.5 or 2. This is the subtle version: multiplying a 56.6 inch gage by 1.5 is not
             a plausible reading, while stretching its deviation from nominal is.

Scorers: IF, VAE error, VAE + EWMA, VAE + Mahalanobis, VAE + robust z, VAE + rolling median.
Thresholds: MAD rule on clean scores (K = 3), 95th percentile, cross validation.
Seeds 0 to 4 compare designs, seeds 5 to 14 confirm.

Usage:  python fdi_route1_raw_injection.py [path/to/WRM_F.csv] [output_dir]
"""
import sys, os
import numpy as np, pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from fdi_maintex_pipeline import CONFIG, load_data, train_vae, metrics, percentile_threshold, cv_threshold
from fdi_hybrid_search import robust_fit, ewma, rolling_median, mad_threshold, DEV_SEEDS, CONFIRM_SEEDS

def inject_raw(R_clean, cols, n_windows, rng, design):
    R = R_clean.copy(); n = len(R); L = CONFIG['window_len']
    y = np.zeros(n, dtype=int); kind = np.array([''] * n, dtype=object)
    spread = (n - n_windows * L) // n_windows
    idx = np.array([cols.index(c) for c in CONFIG['targets']])
    sd = R_clean[:, idx].std(0); med = np.median(R_clean[:, idx], axis=0)
    for i in range(n_windows):
        s = i * (spread + L); e = s + L
        block = R[s:e, :][:, idx]
        method = str(rng.choice(['spike', 'multiply', 'noise']))
        if method == 'spike':
            block = block + rng.choice([-1, 1]) * rng.uniform(2, 3) * sd
        elif method == 'multiply':
            f = rng.choice([1.5, 2.0])
            block = block * f if design == 'raw' else med + (block - med) * f
        else:
            block = block + rng.normal(0, CONFIG['noise_std'], size=block.shape) * sd
        R[s:e, idx] = block; y[s:e] = 1; kind[s:e] = method
    return R, y, kind

class Scorers:
    def __init__(self, vae, X_clean):
        self.vae = vae
        r = self._res(X_clean)
        self.rz = robust_fit(r); self.mcd = MinCovDet(random_state=0).fit(r)
    def _res(self, X):
        mu, _ = self.vae.encode(X); return X - self.vae.dec(mu).numpy()
    def score(self, X):
        r = self._res(X); err = (r ** 2).sum(1); maha = np.sqrt(self.mcd.mahalanobis(r))
        return {'VAE': err, 'VAE+EWMA': ewma(err), 'VAE+Mahalanobis': maha,
                'VAE+robust z': (((r - self.rz[0]) / self.rz[1]) ** 2).sum(1),
                'VAE+rolling median': rolling_median(err)}

def main(csv_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df, _ = load_data(csv_path); cols = list(df.columns); R = df.values.astype(float)
    scaler = StandardScaler().fit(R); X = scaler.transform(R)
    rows, rec = [], []
    for seed in DEV_SEEDS + CONFIRM_SEEDS:
        vae = train_vae(X, seed); sc = Scorers(vae, X)
        iso = IsolationForest(contamination=0.05, random_state=seed).fit(X)
        clean = {'IF': -iso.score_samples(X), **sc.score(X)}
        for design in ['raw', 'deviation']:
            for level, n_win in CONFIG['windows'].items():
                Ri, y, kind = inject_raw(R, cols, n_win, np.random.default_rng(1000 * seed + n_win), design)
                Xi = scaler.transform(Ri)
                scores = {'IF': -iso.score_samples(Xi), **sc.score(Xi)}
                for name, s in scores.items():
                    flag = (s > mad_threshold(clean[name])).astype(int); m = metrics(y, flag)
                    mp, mc = metrics(y, percentile_threshold(s)), metrics(y, cv_threshold(s, y, seed))
                    rows.append(dict(seed=seed, split='dev' if seed in DEV_SEEDS else 'confirm', design=design,
                                     level=f"{int(level*100)}%", model=name, auc=roc_auc_score(y, s),
                                     ap=average_precision_score(y, s),
                                     f1_mad=m['f1'], prec_mad=m['precision'], rec_mad=m['recall'],
                                     f1_pct=mp['f1'], prec_pct=mp['precision'], rec_pct=mp['recall'],
                                     f1_cv=mc['f1'], prec_cv=mc['precision'], rec_cv=mc['recall'], acc_cv=mc['accuracy']))
                    for k in ['spike', 'multiply', 'noise']:
                        if (kind == k).any():
                            rec.append(dict(seed=seed, design=design, level=f"{int(level*100)}%", model=name,
                                            attack=k, recall_mad=flag[kind == k].mean()))
        print(f"seed {seed} done", flush=True)
    res = pd.DataFrame(rows); res.to_csv(os.path.join(out_dir, 'route1_all_seeds.csv'), index=False)
    pd.DataFrame(rec).to_csv(os.path.join(out_dir, 'route1_recall_by_attack.csv'), index=False)
    num = [c for c in res.columns if c not in ('seed', 'split', 'design', 'level', 'model')]
    summ = res.groupby(['split', 'design', 'level', 'model'], sort=False)[num].mean().round(3).reset_index()
    summ.to_csv(os.path.join(out_dir, 'route1_summary.csv'), index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(summ[summ.split == 'confirm'].drop(columns='split').to_string(index=False))

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'WRM_F.csv', sys.argv[2] if len(sys.argv) > 2 else 'route1_results')
