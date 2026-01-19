import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_absolute_error

# -----------------------------
# AYARLAR 
# -----------------------------
FILENAME = "Smart_Bin.csv"
TARGET_COL = "FL_B"     # "FL_A" veya "FL_B"
TEST_SIZE = 0.20
RANDOM_STATE = 42
TOP_N = 10
MIN_COUNT = 5           # Pivot hucrelerinde guven icin min veri adedi

# -----------------------------
# 1) VERIYI YUKLE
# -----------------------------
df = pd.read_csv(FILENAME)
df.dropna(inplace=True)

# Kolon kontrolu (hoca icin temiz)
needed_cols = ["Container Type", "Recyclable fraction", "VS", TARGET_COL]
missing = [c for c in needed_cols if c not in df.columns]
if missing:
    raise ValueError(f"Eksik kolon(lar): {missing}. CSV kolon adlarini kontrol et.")

# -----------------------------
# 2) PIVOT (IS ANALIZI)
# -----------------------------
pivot_mean = df.pivot_table(
    index="Container Type",
    columns="Recyclable fraction",
    values=TARGET_COL,
    aggfunc="mean"
)

pivot_count = df.pivot_table(
    index="Container Type",
    columns="Recyclable fraction",
    values=TARGET_COL,
    aggfunc="count"
)

# Az veri olan hucreleri maskele (yaniltmasin)
pivot_mean_filtered = pivot_mean.where(pivot_count >= MIN_COUNT)

# Top-N kombinasyon (hoca net gorsun)
top_combos = (
    pivot_mean.stack()
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={0: f"mean_{TARGET_COL}"})
).head(TOP_N)

print("\n--- PIVOT OZET ---")
print(f"Hedef: {TARGET_COL}")
print(f"Pivot min_count filtresi: {MIN_COUNT}")
print("\nIlk 10 kombinasyon (ortalama en yuksek):")
print(top_combos.to_string(index=False))

# -----------------------------
# 3) ML: 3 MODEL DENE + EN IYIYI SEC
# -----------------------------
cat_cols = ["Container Type", "Recyclable fraction"]
num_cols = ["VS"]

X = df[cat_cols + num_cols]
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

preprocess_basic = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

preprocess_knn = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ]
)

models = {
    "RandomForest": Pipeline(steps=[
        ("prep", preprocess_basic),
        ("model", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE))
    ]),
    "KNN": Pipeline(steps=[
        ("prep", preprocess_knn),
        ("model", KNeighborsRegressor(n_neighbors=7))
    ]),
    "LinearRegression": Pipeline(steps=[
        ("prep", preprocess_basic),
        ("model", LinearRegression())
    ]),
}

results = []
preds = {}

for name, pipe in models.items():
    # "Makineye ogretme" burada oluyor: fit()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    preds[name] = y_pred
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    results.append((name, r2, mae))

results_df = pd.DataFrame(results, columns=["Algorithm", "R2", "MAE"]).sort_values(
    by=["R2", "MAE"], ascending=[False, True]
)

best_name = results_df.iloc[0]["Algorithm"]
best_pipe = models[best_name]
best_pred = preds[best_name]
best_r2 = float(results_df.iloc[0]["R2"])
best_mae = float(results_df.iloc[0]["MAE"])

print("\n--- MODEL SONUCLARI (3 algoritma) ---")
print(results_df.to_string(index=False))
print(f"\nEN IYI MODEL: {best_name} | R2={best_r2:.4f} | MAE={best_mae:.4f}")

# Ornek: ilk 10 tahmin 
print("\n--- ORNEK TAHMINLER (Ilk 10) ---")
sample_df = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": best_pred[:10],
    "Error": y_test.values[:10] - best_pred[:10]
})
print(sample_df.to_string(index=False))

# -----------------------------
# 4) 3 GRAFIK (en mantikli set)
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# (1) Pivot Heatmap (is analizi)
im = axes[0].imshow(pivot_mean_filtered.values, aspect="auto")
axes[0].set_title(f"Pivot Heatmap (Mean {TARGET_COL}, min_count>={MIN_COUNT})")
axes[0].set_xlabel("Recyclable fraction")
axes[0].set_ylabel("Container Type")
axes[0].set_xticks(np.arange(len(pivot_mean_filtered.columns)))
axes[0].set_xticklabels(pivot_mean_filtered.columns, rotation=45, ha="right")
axes[0].set_yticks(np.arange(len(pivot_mean_filtered.index)))
axes[0].set_yticklabels(pivot_mean_filtered.index)
plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

# (2) Model karsilastirma (R2)
axes[1].bar(results_df["Algorithm"], results_df["R2"])
axes[1].set_title("Model Karsilastirma (R2)")
axes[1].set_xlabel("Algorithm")
axes[1].set_ylabel("R2")
axes[1].tick_params(axis="x", rotation=45)
axes[1].grid(axis="y", linestyle="--", alpha=0.4)

# (3) Best model "neden" grafigi:
# - RandomForest ise: feature importance
# - LinearRegression ise: coef (etki buyuklugu)
# - KNN ise: (feature importance yok) -> actual vs predicted
model_step = best_pipe.named_steps["model"]
prep_step = best_pipe.named_steps["prep"]

if hasattr(model_step, "feature_importances_"):
    # OneHot isimlerini al (kategorik + sayisal)
    ohe = prep_step.named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(cat_cols))
    feat_names = cat_names + num_cols

    importances = model_step.feature_importances_
    idx = np.argsort(importances)[::-1][:15]

    axes[2].bar([feat_names[i] for i in idx], importances[idx])
    axes[2].set_title(f"Best Model Neden? (Importance) - {best_name}")
    axes[2].tick_params(axis="x", rotation=75)

elif hasattr(model_step, "coef_"):
    ohe = prep_step.named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(cat_cols))
    feat_names = cat_names + num_cols

    coefs = np.abs(model_step.coef_)
    idx = np.argsort(coefs)[::-1][:15]

    axes[2].bar([feat_names[i] for i in idx], coefs[idx])
    axes[2].set_title(f"Best Model Neden? (|coef|) - {best_name}")
    axes[2].tick_params(axis="x", rotation=75)

else:
    # KNN gibi modellerde "neden" grafigi yerine kanit grafigi
    axes[2].scatter(y_test, best_pred, alpha=0.6, edgecolor="k", s=35)
    min_val = min(y_test.min(), np.min(best_pred))
    max_val = max(y_test.max(), np.max(best_pred))
    axes[2].plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    axes[2].set_title(f"Best Model: {best_name} (Actual vs Pred)\nR2={best_r2:.3f}, MAE={best_mae:.3f}")
    axes[2].set_xlabel(f"Actual {TARGET_COL}")
    axes[2].set_ylabel(f"Predicted {TARGET_COL}")
    axes[2].grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()
