"""
Agente 1 — Análisis Exploratorio de Datos (EDA)
Compatible con cualquier CSV/XLSX en español o inglés.
"""

import io, base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paleta en español para gráficas
plt.rcParams["font.family"] = "DejaVu Sans"


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    enc = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return enc


def run(df: pd.DataFrame) -> dict:
    results = {}
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # ── Estadísticas básicas ───────────────────────────────────────────────
    describe_raw = df[num_cols].describe().round(4) if num_cols else pd.DataFrame()

    # Renombrar índices del describe al español
    rename_idx = {
        "count": "conteo",
        "mean":  "media",
        "std":   "desv. estándar",
        "min":   "mínimo",
        "25%":   "percentil 25%",
        "50%":   "mediana",
        "75%":   "percentil 75%",
        "max":   "máximo",
    }
    describe_es = describe_raw.rename(index=rename_idx)

    results["estadisticas"] = {
        "shape":      {"filas": df.shape[0], "columnas": df.shape[1]},
        "dtypes":     df.dtypes.astype(str).to_dict(),
        "nulos":      df.isnull().sum().to_dict(),
        "nulos_pct":  (df.isnull().mean() * 100).round(2).to_dict(),
        "describe":   describe_es.to_dict() if not describe_es.empty else {},
        "mediana":    df[num_cols].median().round(4).to_dict() if num_cols else {},
        "num_cols":   num_cols,
        "cat_cols":   cat_cols,
    }

    # ── Visualización 1: histogramas de todas las columnas numéricas ───────
    if num_cols:
        n = len(num_cols)
        cols_per_row = min(3, n)
        rows = (n + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(
            rows, cols_per_row, figsize=(5 * cols_per_row, 3.5 * rows)
        )
        # Normalizar axes a lista plana
        if n == 1:
            axes = [axes]
        elif rows == 1:
            axes = list(axes)
        else:
            axes = [ax for fila in axes for ax in fila]

        colores = ["#6C63FF","#FF6B6B","#4ECDC4","#F7DC6F",
                   "#A29BFE","#FD79A8","#00B894","#FDCB6E"]
        for i, col in enumerate(num_cols):
            axes[i].hist(df[col].dropna(), bins=25,
                         color=colores[i % len(colores)], edgecolor="white")
            axes[i].set_title(col, fontsize=10, fontweight="bold")
            axes[i].set_ylabel("Frecuencia")
            axes[i].set_xlabel("Valor")
            axes[i].grid(axis="y", alpha=0.3)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Distribución de Variables Numéricas",
                     fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()
        results["histograma_b64"] = _b64(fig)
    else:
        results["histograma_b64"] = None

    # ── Visualización 2: matriz de correlación ─────────────────────────────
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        h = max(4, len(num_cols) * 0.75)
        fig2, ax2 = plt.subplots(figsize=(max(6, h), h))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.5, ax=ax2, annot_kws={"size": 8})
        ax2.set_title("Matriz de Correlación", fontsize=13, fontweight="bold")
        fig2.tight_layout()
        results["correlacion_b64"] = _b64(fig2)
    else:
        results["correlacion_b64"] = None

    # ── Top-5 valores por columna categórica (máx 6 columnas) ─────────────
    cat_summary = {}
    for col in cat_cols[:6]:
        vc = df[col].value_counts().head(5)
        cat_summary[col] = vc.to_dict()
    results["cat_summary"] = cat_summary

    return results
