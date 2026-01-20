import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Veri yükleme
df = pd.read_csv("Smart_Bin.csv")
df = df.dropna(subset=["Container Type", "Recyclable fraction", "FL_A", "FL_B", "VS"]).copy()

# ============= PİVOT ANALİZ =============
pivot = df.pivot_table(index="Container Type", columns="Recyclable fraction", 
                       values="FL_B", aggfunc="mean").round(2)
print("\n--- Ortalama Doluluk Seviyesi (FL_B) ---")
print(pivot)

fastest = pivot.stack().sort_values(ascending=False)
print(f"\n EN YUKSEK ORTALAMA DOLULUK: {fastest.index[0][0]} + {fastest.index[0][1]} (FL_B={fastest.iloc[0]:.2f})")


# ============= MAKİNE ÖĞRENMESİ =============
X = df[["Container Type", "Recyclable fraction", "FL_A", "VS"]]
y = df["FL_B"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Binary classification (median üstü/altı)
median = y_train.median()
y_train_bin = (y_train > median).astype(int)
y_test_bin = (y_test > median).astype(int)

# Model eğitimi 
pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), 
     ["Container Type", "Recyclable fraction"]),
    ("num", StandardScaler(), ["FL_A", "VS"])
])

models = {
    "LogReg": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "RF": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
}

results = {}
for name, clf in models.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train_bin)
    pred = pipe.predict(X_test)
    
    acc = accuracy_score(y_test_bin, pred)
    f1 = f1_score(y_test_bin, pred)
    results[name] = {"Accuracy": acc, "F1": f1}
    print(f"\n{name}: Acc={acc*100:.1f}% | F1={f1*100:.1f}%")

best = max(results, key=lambda x: results[x]["F1"])
print(f"\n KAZANAN: {best} (F1={results[best]['F1']*100:.1f}%)")

# ============= GRAFİKLER =============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Grafik 1: Pivot Heatmap
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=axes[0], cbar_kws={'label': 'FL_B'})
axes[0].set_title("Doluluk Seviyesi: Konteyner x Atık Türü", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Atık Türü")
axes[0].set_ylabel("Konteyner Türü")

# Grafik 2: Model Karşılaştırma
model_names = list(results.keys())
acc_vals = [results[m]["Accuracy"]*100 for m in model_names]
f1_vals = [results[m]["F1"]*100 for m in model_names]

x = range(len(model_names))
width = 0.35
axes[1].bar([i - width/2 for i in x], acc_vals, width, label='Accuracy', color='skyblue')
axes[1].bar([i + width/2 for i in x], f1_vals, width, label='F1 Score', color='coral')
axes[1].set_xlabel("Model")
axes[1].set_ylabel("Performans (%)")
axes[1].set_title("Model Performans Karşılaştırması", fontsize=12, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(model_names)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()