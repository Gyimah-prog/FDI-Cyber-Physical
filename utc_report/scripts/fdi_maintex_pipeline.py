"""
FDI detection pipeline written to follow the methodology text of main.tex
("Hybrid VAE-Based Cyber Intrusion Detection in Rail Track Geometry from SCADA Systems").

What main.tex specifies, and how it is implemented here
  Data        18,290 records, 14 selected features, standardized (zero mean, unit variance)
  Models      Isolation Forest, VAE (14 -> 7 latent), VAE+MAD, all trained on clean data only
  VAE score   reconstruction error ||x - x'||^2
  MAD score   |e - median(E)| / MAD(E), with E the clean reconstruction errors
  Injection   on the standardized clean data, non-overlapping windows of 46 records,
              20 windows for 5% (920 records). Each window uses ONE technique on
              Gage, Warp62, Crosslevel:
                additive spike        +/- 2 to 3 standard deviations
                multiplicative        x1.5 or x2
                random noise          Gaussian, high variance
  Thresholds  percentile: 95th percentile of the score distribution
              cross validation: threshold chosen to maximize F1 on labeled folds
  Levels      5% with both thresholds, 2% and 10% with cross validation
  Metrics     precision, recall, F1, accuracy

Choices main.tex leaves open (set in CONFIG, stated in the output)
  hidden layer sizes, noise standard deviation, number of windows at 2% and 10%,
  number of folds, random seeds.

One extra row is reported for comparison only: "VAE+MAD (weighted)". It applies the
1.5 weight on Gage and alignment features that FDI.ipynb uses and main.tex does not mention.

Usage:  python fdi_maintex_pipeline.py [path/to/WRM_F.csv] [output_dir]
"""
import sys, os, json
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from tensorflow import keras

CONFIG = dict(
    features=['LProf62', 'RProf62', 'LAlign62', 'RAlign62', 'Gage', 'Warp62',
              'BFI_L_Value', 'BFI_C_Value', 'BFI_R_Value', 'Crosslevel',
              'TotalLatForce', 'Left Cant', 'Right Cant', 'Speed'],
    targets=['Gage', 'Warp62', 'Crosslevel'],
    window_len=46,
    windows={0.02: 8, 0.05: 20, 0.10: 40},      # 368, 920, 1840 records
    latent_dim=7, hidden=(64, 32, 16), epochs=100, batch_size=64, patience=10,
    noise_std=3.0,                               # "high variance" Gaussian noise
    percentile=95, n_folds=5,
    seeds=[0, 1, 2, 3, 4],
    weighted_features=['Gage', 'LAlign62', 'RAlign62'], weight=1.5,
)

# ------------------------------------------------------------------ data
def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns={'BFI L': 'BFI_L_Value', 'BFI C ': 'BFI_C_Value', 'BFI R ': 'BFI_R_Value'})
    df = df[CONFIG['features']].apply(pd.to_numeric, errors='coerce')
    n_raw = len(df)
    df = df.dropna().reset_index(drop=True)
    return df, n_raw

# ------------------------------------------------------------------ VAE
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mu, log_var = inputs
        eps = keras.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps

class VAE(keras.Model):
    def __init__(self, input_dim, latent_dim, hidden):
        super().__init__()
        self.enc = keras.Sequential([keras.layers.Dense(h, activation='relu') for h in hidden])
        self.mu = keras.layers.Dense(latent_dim)
        self.log_var = keras.layers.Dense(latent_dim)
        self.sample = Sampling()
        self.dec = keras.Sequential([keras.layers.Dense(h, activation='relu') for h in hidden[::-1]]
                                    + [keras.layers.Dense(input_dim)])
    def encode(self, x):
        h = self.enc(x); return self.mu(h), self.log_var(h)
    def call(self, x):
        mu, log_var = self.encode(x)
        x_hat = self.dec(self.sample([mu, log_var]))
        recon = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_hat), axis=1))
        kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1))
        self.add_loss(recon + kl)
        return x_hat
    def reconstruct(self, x):          # deterministic reconstruction through the latent mean
        mu, _ = self.encode(x)
        return self.dec(mu).numpy()

def train_vae(X, seed):
    keras.utils.set_random_seed(seed)
    vae = VAE(X.shape[1], CONFIG['latent_dim'], CONFIG['hidden'])
    vae.compile(optimizer=keras.optimizers.Adam(1e-3))
    vae.fit(X, X, epochs=CONFIG['epochs'], batch_size=CONFIG['batch_size'], validation_split=0.2,
            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=CONFIG['patience'],
                                                     restore_best_weights=True)], verbose=0)
    return vae

# ------------------------------------------------------------------ injection (main.tex, FDI procedure)
def inject(X_clean, cols, n_windows, rng):
    X = X_clean.copy(); n = len(X); L = CONFIG['window_len']
    y = np.zeros(n, dtype=int); log = []
    spread = (n - n_windows * L) // n_windows
    idx = np.array([cols.index(c) for c in CONFIG['targets']])
    for i in range(n_windows):
        s = i * (spread + L); e = s + L
        block = X[s:e, :][:, idx]
        method = rng.choice(['spike', 'multiply', 'noise'])
        if method == 'spike':
            block = block + rng.choice([-1, 1]) * rng.uniform(2, 3)
        elif method == 'multiply':
            block = block * rng.choice([1.5, 2.0])
        else:
            block = block + rng.normal(0, CONFIG['noise_std'], size=block.shape)
        X[s:e, idx] = block
        y[s:e] = 1; log.append(str(method))
    return X, y, log

# ------------------------------------------------------------------ scores
def mad_score(e, e_ref):
    med = np.median(e_ref); mad = np.median(np.abs(e_ref - med)) + 1e-12
    return np.abs(e - med) / mad

def metrics(y, yp):
    return dict(precision=precision_score(y, yp, zero_division=0), recall=recall_score(y, yp, zero_division=0),
                f1=f1_score(y, yp, zero_division=0), accuracy=accuracy_score(y, yp))

def percentile_threshold(score):
    return (score > np.percentile(score, CONFIG['percentile'])).astype(int)

def best_f1_threshold(score, y):
    """Score cut that maximizes F1 on labeled data (every distinct score is a candidate)."""
    order = np.argsort(-score); s, yy = score[order], y[order]
    tp = np.cumsum(yy); k = np.arange(1, len(s) + 1)          # flag the top k records
    f1 = 2 * tp / (k + yy.sum())
    j = int(np.argmax(f1))
    return (s[j] + s[j + 1]) / 2 if j + 1 < len(s) else s[j] - 1e-12

def cv_threshold(score, y, seed):
    """Threshold chosen on the training folds (labels used), applied to the held out fold."""
    yp = np.zeros_like(y)
    for tr, te in StratifiedKFold(CONFIG['n_folds'], shuffle=True, random_state=seed).split(score, y):
        yp[te] = (score[te] > best_f1_threshold(score[tr], y[tr])).astype(int)
    return yp

# ------------------------------------------------------------------ main
def main(csv_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df, n_raw = load_data(csv_path)
    cols = list(df.columns)
    X = StandardScaler().fit_transform(df.values)
    w = np.array([CONFIG['weight'] if c in CONFIG['weighted_features'] else 1.0 for c in cols])
    print(f"Records read {n_raw}, used {len(df)}, features {len(cols)}")
    rows = []
    for seed in CONFIG['seeds']:
        iso = IsolationForest(contamination=0.05, random_state=seed).fit(X)       # clean data only
        vae = train_vae(X, seed)                                                   # clean data only
        sq_clean = (X - vae.reconstruct(X)) ** 2
        e_clean, ew_clean = sq_clean.sum(1), (sq_clean * w).mean(1)
        for level, n_win in CONFIG['windows'].items():
            Xi, y, log = inject(X, cols, n_win, np.random.default_rng(1000 * seed + n_win))
            sq = (Xi - vae.reconstruct(Xi)) ** 2
            e, ew = sq.sum(1), (sq * w).mean(1)
            scores = {'IF': -iso.score_samples(Xi),
                      'VAE': e,
                      'VAE+MAD': mad_score(e, e_clean),
                      'VAE+MAD (weighted, FDI.ipynb variant)': mad_score(ew, ew_clean)}
            for model, s in scores.items():
                base = dict(seed=seed, level=f"{int(level*100)}%", attacked=int(y.sum()), model=model,
                            auc=roc_auc_score(y, s))
                if level == 0.05:
                    rows.append({**base, 'threshold': 'percentile', **metrics(y, percentile_threshold(s))})
                rows.append({**base, 'threshold': 'cross validation', **metrics(y, cv_threshold(s, y, seed))})
        print(f"seed {seed} done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, 'maintex_rerun_all_seeds.csv'), index=False)
    g = res.groupby(['level', 'threshold', 'model'], sort=False)[['precision', 'recall', 'f1', 'accuracy', 'auc']]
    summary = g.mean().round(3).join(g.std().round(3), rsuffix='_sd').reset_index()
    summary.to_csv(os.path.join(out_dir, 'maintex_rerun_summary.csv'), index=False)
    json.dump(CONFIG, open(os.path.join(out_dir, 'maintex_rerun_config.json'), 'w'), indent=1, default=str)
    pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'WRM_F.csv', sys.argv[2] if len(sys.argv) > 2 else 'maintex_rerun_results')
