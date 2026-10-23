# ============================================================================
# ADD-ON CELLS for FDI_Attack_Aware_Feature_Susceptibility_Framework_Cleaned (2).ipynb
#
# Two upgrades, split into 4 new cells. Paste each block into a NEW cell at
# the indicated location and run in order. Nothing in the existing notebook
# needs to change, these are additive.
#
#   BLOCK A (REVISED) -> paste as a new cell right after cell 32 (Section
#              5.1, the one-feature static attack test). Supersedes the
#              earlier version of this block. Does TWO fixes at once:
#              (1) subsets the static data to the 7 features common across
#              all three attack types (Gage, Crosslevel, Warp62, LProf62,
#              RProf62, LAlign62, RAlign62), dropping the 14-feature scope
#              and the 7 statistical-only extras entirely, per the decision
#              to keep one consistent 7-feature scope throughout, and
#              (2) adds a proper held-out split, mirroring cell 40's
#              temporal pattern, so the static VAE/Isolation Forest are
#              retrained on the 7-feature train split only, and the
#              one-feature test runs on the 7-feature held-out test split.
#
#   BLOCK B -> paste as a new cell right after Section 10 (cell 57), before
#              Section 11. Defines the reusable bootstrap CI helper.
#
#   BLOCK C -> paste as a new cell right after BLOCK B. Per-feature bootstrap
#              CIs for the temporal feature-specific test (persistent,
#              one-shot, gradual x LSTM-VAE, Transformer-VAE).
#
#   BLOCK D -> paste as a new cell right after BLOCK C. Per-feature bootstrap
#              CIs for the replay feature-specific test (Stagnation,
#              LSTM-VAE, Transformer-VAE).
#
# All four reuse functions/variables already defined earlier in the notebook
# (build_vae, vae_reconstruction_error, mad_score_from_train,
# train_percentile_threshold, inject_one_feature_static_scaled,
# inject_one_feature_temporal, inject_replay_partial, stagnation_score,
# lstm_reconstruction_error, STATIC_COLS, FEAT_COLS, FEAT_NAMES, X_train_t,
# X_test_t, train_stag, lstm_enc/dec, trans_enc/dec, SEED). Nothing here
# redefines those, it only reads them.
# ============================================================================


# ============================================================================
# BLOCK A (REVISED) -- paste after cell 32 (Section 5.1)
# Subsets to the 7 common features, adds a leak-free 80/20 held-out split
# (mirroring cell 40's temporal split), retrains the static VAE + Isolation
# Forest on the 7-feature train split only, then reruns the one-feature
# attack test on the 7-feature held-out test split only.
# Produces: static_feature_df_holdout (7 features, held out)
# ============================================================================

from sklearn.model_selection import train_test_split

# ── Same 7 features used throughout the temporal and replay pipelines ──
COMMON_FEATURES = ['Gage', 'Crosslevel', 'Warp62', 'LProf62', 'RProf62', 'LAlign62', 'RAlign62']
df_static_common = df_static[COMMON_FEATURES].reset_index(drop=True)

# ── Split BEFORE scaling -- prevents test-row leakage ──────────────────
idx_all_static = np.arange(len(df_static_common))
idx_train_static, idx_test_static = train_test_split(
    idx_all_static, test_size=0.2, random_state=SEED
)

df_static_train = df_static_common.iloc[idx_train_static].reset_index(drop=True)
df_static_test  = df_static_common.iloc[idx_test_static].reset_index(drop=True)

scaler_static_ho = StandardScaler()
X_static_train_ho = scaler_static_ho.fit_transform(df_static_train)
X_static_test_ho  = scaler_static_ho.transform(df_static_test)

print(f"Static train: {X_static_train_ho.shape}   Static test: {X_static_test_ho.shape}")
print("Scaler fitted on training rows only -- no leakage")
print(f"Features used: {COMMON_FEATURES}")

# ── Feature weights on this 7-feature space (all 7 are safety-critical) ─
fw_static_ho = np.array([1.5 if c in ('Gage', 'LAlign62', 'RAlign62', 'Warp62', 'Crosslevel')
                          else 1.0 for c in COMMON_FEATURES])

# ── Retrain the static VAE and Isolation Forest on the TRAIN split only ─
vae_s_ho, enc_s_ho, dec_s_ho = build_vae(X_static_train_ho.shape[1], latent_dim=7)
cb_ho = [EarlyStopping(patience=10, restore_best_weights=True),
         ReduceLROnPlateau(patience=5, factor=0.5, verbose=0)]
history_ho = vae_s_ho.fit(
    X_static_train_ho, X_static_train_ho,
    epochs=100, batch_size=64,
    validation_split=0.15,
    callbacks=cb_ho, verbose=0
)
print("Held-out static VAE training complete. Final val loss:",
      round(history_ho.history['val_loss'][-1], 4))

iso_s_ho = IsolationForest(contamination=0.05, random_state=SEED)
iso_s_ho.fit(X_static_train_ho)

# ── Rerun the one-feature static attack test on the HELD-OUT test split ─
def static_feature_sensitivity_holdout(
    X_train, X_test, feature_names, enc, dec, iso, feature_weights,
    injection_pct=0.05, magnitude_z=2.5, seed=SEED
):
    train_vae = vae_reconstruction_error(enc, dec, X_train, feature_weights)
    train_iso = -iso.score_samples(X_train)
    train_mad = mad_score_from_train(train_vae, train_vae)

    thresholds = {
        'IF': train_percentile_threshold(train_iso, 95),
        'VAE': train_percentile_threshold(train_vae, 95),
        'VAE+MAD': train_percentile_threshold(train_mad, 95),
    }

    rows = []
    for j, feature in enumerate(feature_names):
        X_inj, y_true = inject_one_feature_static_scaled(
            X_test, feature_idx=j, injection_pct=injection_pct,
            magnitude_z=magnitude_z, seed=seed
        )
        score_vae = vae_reconstruction_error(enc, dec, X_inj, feature_weights)
        score_iso = -iso.score_samples(X_inj)
        score_mad = mad_score_from_train(train_vae, score_vae)

        score_map = {'IF': score_iso, 'VAE': score_vae, 'VAE+MAD': score_mad}
        for model_name, scores in score_map.items():
            y_pred = (scores > thresholds[model_name]).astype(int)
            rows.append({
                'Feature': feature, 'Model': model_name,
                'Attack prevalence': injection_pct, 'Magnitude (z)': magnitude_z,
                'Precision': precision_score(y_true, y_pred, zero_division=0),
                'Recall': recall_score(y_true, y_pred, zero_division=0),
                'F1': f1_score(y_true, y_pred, zero_division=0),
                'False positive rate': (
                    ((y_pred == 1) & (y_true == 0)).sum() / max(1, (y_true == 0).sum())
                )
            })
    return pd.DataFrame(rows)

static_feature_df_holdout = static_feature_sensitivity_holdout(
    X_static_train_ho, X_static_test_ho, COMMON_FEATURES,
    enc_s_ho, dec_s_ho, iso_s_ho, fw_static_ho,
    injection_pct=0.05, magnitude_z=2.5, seed=SEED
)

display(static_feature_df_holdout.sort_values(['Model', 'Recall'], ascending=[True, False]))


# ============================================================================
# BLOCK B -- paste after Section 10 (cell 57), before Section 11
# Generalizes bootstrap_auc_f1 (cell 67) to report Precision/Recall/F1/AUC
# together, and to support "lower score = anomaly" detectors (Stagnation).
# ============================================================================

from sklearn.utils import resample

def bootstrap_metric_ci(y_true, scores, threshold, higher_is_anomaly=True,
                          n_boot=2000, seed=SEED):
    """Bootstrap 95% CIs for Precision, Recall, F1, and ROC-AUC."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    precisions, recalls, f1s, aucs = [], [], [], []

    for _ in range(n_boot):
        idx = resample(np.arange(len(y_true)), random_state=rng)
        yt, sc = y_true[idx], scores[idx]
        if len(np.unique(yt)) < 2:
            continue
        pred = (sc > threshold).astype(int) if higher_is_anomaly \
               else (sc < threshold).astype(int)
        auc_input = sc if higher_is_anomaly else -sc
        precisions.append(precision_score(yt, pred, zero_division=0))
        recalls.append(recall_score(yt, pred, zero_division=0))
        f1s.append(f1_score(yt, pred, zero_division=0))
        aucs.append(roc_auc_score(yt, auc_input))

    def summarize(vals):
        return (np.mean(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5))

    p_m, p_lo, p_hi = summarize(precisions)
    r_m, r_lo, r_hi = summarize(recalls)
    f_m, f_lo, f_hi = summarize(f1s)
    a_m, a_lo, a_hi = summarize(aucs)

    return {
        'Precision': p_m, 'Precision_CI_Lower': p_lo, 'Precision_CI_Upper': p_hi,
        'Recall': r_m, 'Recall_CI_Lower': r_lo, 'Recall_CI_Upper': r_hi,
        'F1': f_m, 'F1_CI_Lower': f_lo, 'F1_CI_Upper': f_hi,
        'ROC_AUC': a_m, 'ROC_AUC_CI_Lower': a_lo, 'ROC_AUC_CI_Upper': a_hi,
    }


# ============================================================================
# BLOCK C -- paste after BLOCK B
# Per-feature bootstrap CIs for the temporal feature-specific test
# (persistent, one-shot, gradual x LSTM-VAE, Transformer-VAE).
# Produces: temporal_feature_ci_df
# NOTE: 7 features x 3 attack types x 2 models x 2000 bootstrap resamples
# -- expect this cell to take a few minutes.
# ============================================================================

temporal_ci_rows = []
model_specs_temporal = {
    'LSTM-VAE': (lstm_enc, lstm_dec),
    'Transformer-VAE': (trans_enc, trans_dec),
}

model_train_err_ci = {}
model_threshold_ci = {}
for model_name, (encoder, decoder) in model_specs_temporal.items():
    train_err = lstm_reconstruction_error(encoder, decoder, X_train_t)
    train_mad = mad_score_from_train(train_err, train_err)
    model_train_err_ci[model_name] = train_err
    model_threshold_ci[model_name] = train_percentile_threshold(train_mad, 95)

for attack_type in ['persistent', 'one_shot', 'gradual']:
    for feature_idx, feature in enumerate(FEAT_COLS):
        X_inj, y_true, _ = inject_one_feature_temporal(
            X_test_t, feature_idx=feature_idx, injection_pct=0.05,
            attack_type=attack_type, persist_months=6, magnitude_z=2.5, seed=SEED
        )
        for model_name, (encoder, decoder) in model_specs_temporal.items():
            test_err = lstm_reconstruction_error(encoder, decoder, X_inj)
            test_mad = mad_score_from_train(model_train_err_ci[model_name], test_err)
            ci = bootstrap_metric_ci(
                y_true, test_mad, model_threshold_ci[model_name],
                higher_is_anomaly=True, n_boot=2000, seed=SEED
            )
            ci.update({'Feature': feature, 'Attack type': attack_type, 'Model': model_name})
            temporal_ci_rows.append(ci)

temporal_feature_ci_df = pd.DataFrame(temporal_ci_rows)
display(temporal_feature_ci_df.sort_values(
    ['Attack type', 'Model', 'Recall'], ascending=[True, True, False]))


# ============================================================================
# BLOCK D -- paste after BLOCK C
# Per-feature bootstrap CIs for the replay feature-specific test
# (Stagnation, LSTM-VAE, Transformer-VAE), single-feature partial replay
# only (matches cell 57's single-feature loop).
# Produces: replay_feature_ci_df
# ============================================================================

replay_ci_rows = []

train_lstm_scores_r  = temporal_reconstruction_error(lstm_enc,  lstm_dec,  X_train_t)
train_trans_scores_r = temporal_reconstruction_error(trans_enc, trans_dec, X_train_t)
thresh_lstm_replay  = np.percentile(train_lstm_scores_r, 95)
thresh_trans_replay = np.percentile(train_trans_scores_r, 95)
thresh_stag_replay  = np.percentile(train_stag, 5)   # train_stag already defined in cell 54

for i, name in enumerate(FEAT_NAMES):
    X_inj, y_inj = inject_replay_partial(X_test_t, 0.05, feat_idx=[i])

    stag_scores  = stagnation_score(X_inj, window=6)
    lstm_scores  = lstm_reconstruction_error(lstm_enc,  lstm_dec,  X_inj)
    trans_scores = lstm_reconstruction_error(trans_enc, trans_dec, X_inj)

    ci_stag = bootstrap_metric_ci(y_inj, stag_scores, thresh_stag_replay,
                                    higher_is_anomaly=False, n_boot=2000, seed=SEED)
    ci_stag.update({'Feature': name, 'Model': 'Stagnation'})
    replay_ci_rows.append(ci_stag)

    ci_lstm = bootstrap_metric_ci(y_inj, lstm_scores, thresh_lstm_replay,
                                    higher_is_anomaly=True, n_boot=2000, seed=SEED)
    ci_lstm.update({'Feature': name, 'Model': 'LSTM-VAE'})
    replay_ci_rows.append(ci_lstm)

    ci_trans = bootstrap_metric_ci(y_inj, trans_scores, thresh_trans_replay,
                                     higher_is_anomaly=True, n_boot=2000, seed=SEED)
    ci_trans.update({'Feature': name, 'Model': 'Transformer-VAE'})
    replay_ci_rows.append(ci_trans)

replay_feature_ci_df = pd.DataFrame(replay_ci_rows)
display(replay_feature_ci_df.sort_values(['Model', 'ROC_AUC']))


# ============================================================================
# BLOCK E -- paste after BLOCK D
# Consolidated cross-attack-type feature ranking, heatmap for the paper.
# Uses the proposed detector for each attack type (VAE+MAD for statistical,
# mean of LSTM-VAE/Transformer-VAE across all three subtypes for temporal,
# Stagnation for replay), ranks features 1 (hardest to detect) to 7
# (easiest) within each attack type, and plots the result. Requires
# static_feature_df_holdout (Block A), temporal_feature_ci_df (Block C),
# and replay_feature_ci_df (Block D) to already exist.
# Produces: feature_rank_table (DataFrame), feature_rank_heatmap.pdf
# ============================================================================

import matplotlib.pyplot as plt

FEATURES = ['Gage', 'Crosslevel', 'Warp62', 'LProf62', 'RProf62', 'LAlign62', 'RAlign62']

static_vae = (static_feature_df_holdout[static_feature_df_holdout['Model'] == 'VAE+MAD']
              .set_index('Feature')['F1'].reindex(FEATURES))

temporal_mean_auc = (temporal_feature_ci_df
                      .groupby('Feature')['ROC_AUC'].mean().reindex(FEATURES))

replay_stag = (replay_feature_ci_df[replay_feature_ci_df['Model'] == 'Stagnation']
               .set_index('Feature')['ROC_AUC'].reindex(FEATURES))

feature_rank_table = pd.DataFrame({
    'Statistical (F1)': static_vae,
    'Temporal (mean AUC)': temporal_mean_auc,
    'Replay (AUC)': replay_stag,
})
rank_table = feature_rank_table.rank()
rank_table['Mean rank'] = rank_table.mean(axis=1)
feature_rank_table['Mean rank'] = rank_table['Mean rank']
feature_rank_table = feature_rank_table.sort_values('Mean rank')
display(feature_rank_table.round(3))

plot_df = rank_table.drop(columns='Mean rank').loc[feature_rank_table.index]
plot_df.columns = ['Statistical', 'Temporal', 'Replay']

fig, ax = plt.subplots(figsize=(5.5, 4.2))
im = ax.imshow(plot_df.values, cmap='RdYlGn', vmin=1, vmax=7, aspect='auto')
ax.set_xticks(range(3)); ax.set_xticklabels(plot_df.columns, fontsize=10)
ax.set_yticks(range(len(plot_df))); ax.set_yticklabels(plot_df.index, fontsize=10)
for i in range(plot_df.shape[0]):
    for j in range(plot_df.shape[1]):
        ax.text(j, i, f"{plot_df.values[i, j]:.0f}", ha='center', va='center', fontsize=10)
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Detectability rank (1 = hardest to detect, 7 = easiest)', fontsize=8)
ax.set_title('Feature Detectability Rank Across Attack Types', fontsize=11)
plt.tight_layout()
plt.savefig('feature_rank_heatmap.pdf')
plt.savefig('feature_rank_heatmap.png', dpi=150)
print("Saved feature_rank_heatmap.pdf and .png to the Colab working directory")
print("Download them from the Colab file browser (folder icon, left sidebar)")


# ============================================================================
# BLOCK F -- paste after BLOCK E
# Reruns the AGGREGATE statistical FDI test (2%, 5%, 10% injection into
# Gage, Warp62, Crosslevel jointly, using the existing inject_static and
# run_static_detection functions) on the 7-feature HELD-OUT static
# baseline from Block A, instead of the original 14-feature in-sample
# enc_s/dec_s/iso_s/X_static. This keeps the aggregate/robustness numbers
# consistent with the 7-feature, held-out feature-specific numbers.
# Produces: all_static_ho (dict, mirrors the notebook's original all_static)
# ============================================================================

all_static_ho = {}
for pct in [0.02, 0.05, 0.10]:
    df_inj_ho, y_inj_ho = inject_static(df_static_test, injection_pct=pct)
    X_inj_ho = scaler_static_ho.transform(df_inj_ho)
    all_static_ho[pct] = run_static_detection(
        X_static_train_ho, X_inj_ho, y_inj_ho,
        enc_s_ho, dec_s_ho, iso_s_ho, fw_static_ho,
        f'{int(pct*100)}pct-ho'
    )

print("Aggregate statistical FDI (7-feature, held-out) at 2%/5%/10%:")
for pct in [0.02, 0.05, 0.10]:
    for detector in ['Isolation Forest', 'VAE with MAD thresholding']:
        r = all_static_ho[pct][detector]
        print(f"  {int(pct*100)}% {detector:28s}  P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}")


# ============================================================================
# BLOCK G -- paste after BLOCK F
# Assembles Table 2 (canonical attack-family-level comparison) and
# Figure 2 (statistical + temporal detection performance across 2%, 5%,
# 10% injection). Statistical uses the 7-feature held-out baseline
# (Block F). Temporal and replay reuse the notebook's existing,
# already-held-out results (y_p5/err_p5 from Section 8, test_cu_p5/
# train_cu from Section 10.1, y_r5/test_stag/train_stag from Section 9),
# recomputed fresh here so this block does not depend on cell 67 having
# already run.
# Produces: canonical_table_v2 (DataFrame), intensity_robustness.pdf
# ============================================================================

canonical_rows_v2 = []

def record_result_v2(family, variant, detector, y, score, threshold, higher_is_anomaly=True):
    y = np.asarray(y); score = np.asarray(score)
    if higher_is_anomaly:
        pred = (score > threshold).astype(int); auc_input = score
    else:
        pred = (score < threshold).astype(int); auc_input = -score
    canonical_rows_v2.append({
        'Attack Family': family, 'Attack Variant': variant, 'Detector': detector,
        'Precision': precision_score(y, pred, zero_division=0),
        'Recall': recall_score(y, pred, zero_division=0),
        'F1': f1_score(y, pred, zero_division=0),
        'Accuracy': np.mean(y == pred),
        'ROC_AUC': roc_auc_score(y, auc_input),
    })

# ---- Statistical: 7-feature held-out (Block F), 5% mixed ----
df_inj5_ho, y_true5_ho = inject_static(df_static_test, injection_pct=0.05)
X_inj5_ho = scaler_static_ho.transform(df_inj5_ho)
train_if_ho  = -iso_s_ho.score_samples(X_static_train_ho)
test_if_ho   = -iso_s_ho.score_samples(X_inj5_ho)
train_vae_ho = vae_reconstruction_error(enc_s_ho, dec_s_ho, X_static_train_ho, fw_static_ho)
test_vae_ho  = vae_reconstruction_error(enc_s_ho, dec_s_ho, X_inj5_ho, fw_static_ho)
train_mad_ho = mad_score_from_train(train_vae_ho, train_vae_ho)
test_mad_ho  = mad_score_from_train(train_vae_ho, test_vae_ho)

record_result_v2('Statistical', 'Mixed statistical 5%', 'Isolation Forest',
                  y_true5_ho, test_if_ho, np.percentile(train_if_ho, 95))
record_result_v2('Statistical', 'Mixed statistical 5%', 'VAE with MAD thresholding',
                  y_true5_ho, test_mad_ho, np.percentile(train_mad_ho, 95))

# ---- Temporal: recomputed fresh, no dependency on cell 67 ----
train_lstm_err_g   = temporal_reconstruction_error(lstm_enc, lstm_dec, X_train_t)
train_lstm_score_g = mad_score_from_train(train_lstm_err_g, train_lstm_err_g)
train_trans_err_g   = temporal_reconstruction_error(trans_enc, trans_dec, X_train_t)
train_trans_score_g = mad_score_from_train(train_trans_err_g, train_trans_err_g)

record_result_v2('Temporal', 'Persistent 5%', 'LSTM-VAE with MAD thresholding',
                  y_p5, mad_score_from_train(train_lstm_err_g, err_p5), np.percentile(train_lstm_score_g, 95))
record_result_v2('Temporal', 'One-shot 5%', 'LSTM-VAE with MAD thresholding',
                  y_o5, mad_score_from_train(train_lstm_err_g, err_o5), np.percentile(train_lstm_score_g, 95))
record_result_v2('Temporal', 'Persistent 5%', 'Transformer-VAE with MAD thresholding',
                  y_tp5, mad_score_from_train(train_trans_err_g, err_tp5), np.percentile(train_trans_score_g, 95))
record_result_v2('Temporal', 'One-shot 5%', 'Transformer-VAE with MAD thresholding',
                  y_to5, mad_score_from_train(train_trans_err_g, err_to5), np.percentile(train_trans_score_g, 95))
record_result_v2('Temporal', 'Persistent 5%', 'CUSUM baseline',
                  y_p5, test_cu_p5, np.percentile(train_cu, 95))

# ---- Replay: reuse the existing, already held-out result ----
record_result_v2('Replay', 'Exact replay 5%', 'Stagnation detector',
                  y_r5, test_stag, np.percentile(train_stag, 5), higher_is_anomaly=False)

canonical_table_v2 = pd.DataFrame(canonical_rows_v2).sort_values(
    ['Attack Family', 'Attack Variant', 'F1'], ascending=[True, True, False]
).reset_index(drop=True)
display(canonical_table_v2.round(3))

attack_family_table_v2 = canonical_table_v2.pivot_table(
    index='Detector', columns='Attack Family', values='F1', aggfunc='mean'
)
display(attack_family_table_v2.round(3))

# ---- Figure 2: intensity robustness (statistical + temporal, 2/5/10%) ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
for detector in ['Isolation Forest', 'VAE with MAD thresholding']:
    values = [all_static_ho[pct][detector]['f1'] for pct in [0.02, 0.05, 0.10]]
    ax.plot(['2%', '5%', '10%'], values, marker='o', label=detector)
ax.set_title('Statistical'); ax.set_ylim(0, 1)
ax.set_xlabel('Injected sample proportion'); ax.set_ylabel('F1-score')
ax.legend(fontsize=8)

ax = axes[1]
for label, results_dict in [('LSTM-VAE with MAD thresholding', temporal_results),
                             ('Transformer-VAE with MAD thresholding', trans_results)]:
    values = [next(iter(results_dict['persistent'][pct].values()))['f1'] for pct in [0.02, 0.05, 0.10]]
    ax.plot(['2%', '5%', '10%'], values, marker='o', label=label)
ax.set_title('Temporal (persistent)'); ax.set_ylim(0, 1)
ax.set_xlabel('Injected sample proportion')
ax.legend(fontsize=8)

plt.suptitle('Detection Performance Across Attack Prevalence')
plt.tight_layout()
plt.savefig('intensity_robustness.pdf')
plt.savefig('intensity_robustness.png', dpi=150)
print("Saved intensity_robustness.pdf and .png to the Colab working directory")
