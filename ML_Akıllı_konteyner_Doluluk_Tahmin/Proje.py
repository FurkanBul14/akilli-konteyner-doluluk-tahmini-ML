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

# ---------------------------------------------------------
# 1) VERIYI YUKLEME
# ---------------------------------------------------------
filename = "Smart_Bin.csv"
df = pd.read_csv(filename)

# Eksik verileri temizle
df.dropna(inplace=True)

# Hedef degisken secimi (FL_B veya FL_A)
target_col = "FL_B"

# ---------------------------------------------------------
# 2) OZELLIKLER (FEATURES)
# ---------------------------------------------------------
# Kategorik degiskenler
cat_cols = ["Container Type", "Recyclable fraction"]

# Sayisal degiskenler
num_cols = ["VS"]

# Bagimsiz degiskenler (X) ve hedef degisken (y)
X = df[cat_cols + num_cols]
y = df[target_col]

# Veriyi egitim ve test olarak ayirma
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 3) PIVOT ANALIZI (IS ANALIZI)
# ---------------------------------------------------------
# Konteyner tipi ve atik turune gore ortalama doluluk
pivot_mean = df.pivot_table(
    index="Container Type",
    columns="Recyclable fraction",
    values=target_col,
    aggfunc="mean"
)

# Her kombinasyon icin kac veri oldugunu gosterir
pivot_count = df.pivot_table(
    index="Container Type",
    columns="Recyclable fraction",
    values=target_col,
    aggfunc="count"
)

print("\n--- PIVOT ORTALAMA ---")
print(pivot_mean)

print("\n--- PIVOT ADET ---")
print(pivot_count)

# En yuksek ortalama doluluga sahip kombinasyonlari sirala
top_combos = (
    pivot_mean.stack()
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={0: f"mean_{target_col}"})
)

print("\n--- EN YUKSEK ORTALAMAYA SAHIP ILK 10 KOMBINASYON ---")
print(top_combos.head(10).to_string(index=False))

# ---------------------------------------------------------
# 4) MAKINE OGRENMESI MODELLERI
# ---------------------------------------------------------
# Kategorik veriler icin OneHot, sayisal veriler icin dogrudan gecis
preprocess_basic = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# KNN modeli icin sayisal verileri olceklendirme
preprocess_knn = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ]
)

# Kullanilacak 3 farkli regresyon modeli
models = {
    "RandomForest": Pipeline(steps=[
        ("prep", preprocess_basic),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42))
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

# Modelleri egit ve test et
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    preds[name] = y_pred

    # Performans metrikleri
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    results.append((name, r2, mae))

# Sonuclari R2 yuksek, MAE dusuk olacak sekilde sirala
results_df = pd.DataFrame(
    results, columns=["Algorithm", "R2", "MAE"]
).sort_values(by=["R2", "MAE"], ascending=[False, True])

print("\n--- MODEL PERFORMANSLARI ---")
print(results_df.to_string(index=False))

# En iyi modeli sec
best_name = results_df.iloc[0]["Algorithm"]
best_pred = preds[best_name]
best_r2 = float(results_df.iloc[0]["R2"])
best_mae = float(results_df.iloc[0]["MAE"])

print(f"\nEN IYI MODEL: {best_name} | R2={best_r2:.4f} | MAE={best_mae:.4f}")

# ---------------------------------------------------------
# 5) GORSELLESTIRME (3 ANA GRAFIK)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# (1) Pivot Heatmap - Ortalama doluluk
im = axes[0].imshow(pivot_mean.values, aspect="auto")
axes[0].set_title(f"Pivot Heatmap - Ortalama {target_col}")
axes[0].set_xlabel("Recyclable fraction")
axes[0].set_ylabel("Container Type")

axes[0].set_xticks(np.arange(len(pivot_mean.columns)))
axes[0].set_xticklabels(pivot_mean.columns, rotation=45, ha="right")
axes[0].set_yticks(np.arange(len(pivot_mean.index)))
axes[0].set_yticklabels(pivot_mean.index)

plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

# (2) Modellerin R2 degerlerine gore karsilastirilmasi
axes[1].bar(results_df["Algorithm"], results_df["R2"])
axes[1].set_title("Model Karsilastirma (R2)")
axes[1].set_xlabel("Algoritma")
axes[1].set_ylabel("R2")
axes[1].tick_params(axis="x", rotation=45)
axes[1].grid(axis="y", linestyle="--", alpha=0.4)

# (3) En iyi model icin gercek ve tahmin degerlerinin karsilastirilmasi
axes[2].scatter(y_test, best_pred, alpha=0.6, edgecolor="k", s=35)

min_val = min(y_test.min(), np.min(best_pred))
max_val = max(y_test.max(), np.max(best_pred))
axes[2].plot([min_val, max_val], [min_val, max_val], "r--", lw=2)

axes[2].set_title(
    f"En Iyi Model: {best_name}\nR2={best_r2:.3f}, MAE={best_mae:.3f}"
)
axes[2].set_xlabel(f"Gercek {target_col}")
axes[2].set_ylabel(f"Tahmin {target_col}")
axes[2].grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()
