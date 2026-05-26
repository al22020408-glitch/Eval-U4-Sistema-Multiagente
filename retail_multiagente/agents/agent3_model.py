"""
Agente 3 — Modelo Predictivo
Detecta automáticamente la variable objetivo y la columna de agrupación.
Soporta nombres de columna en español e inglés (con o sin acentos).
"""

import io, base64, unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model    import LinearRegression
from sklearn.tree            import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_squared_error

plt.rcParams["font.family"] = "DejaVu Sans"


# ── Utilidades ─────────────────────────────────────────────────────────────────

def _norm(texto: str) -> str:
    """Quita acentos y pasa a minúsculas."""
    return "".join(
        c for c in unicodedata.normalize("NFD", texto.lower())
        if unicodedata.category(c) != "Mn"
    )


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    enc = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return enc


# ── Listas de palabras clave bilingües ─────────────────────────────────────────

# Columnas que podrían ser el TARGET (variable a predecir)
KW_TARGET = [
    # Español
    "ingreso", "ingresos", "venta", "ventas", "total", "totales",
    "monto", "montos", "precio", "precios", "ganancia", "ganancias",
    "facturacion", "factura", "importe", "importes", "recaudacion",
    "beneficio", "beneficios", "utilidad", "utilidades", "valor",
    # Inglés
    "revenue", "revenues", "sale", "sales", "amount", "amounts",
    "price", "prices", "profit", "profits", "total", "income",
    "earnings", "billing", "turnover",
]

# Columnas que podrían ser el GRUPO para agrupar predicciones
KW_GRUPO = [
    # Español
    "sku", "producto", "productos", "articulo", "articulos",
    "categoria", "categorias", "tipo", "tipos", "clase", "clases",
    "nombre", "nombres", "descripcion", "referencia", "codigo",
    "departamento", "seccion", "familia",
    # Inglés
    "product", "products", "item", "items", "category", "categories",
    "type", "types", "class", "classes", "name", "names",
    "description", "reference", "code", "department", "section",
]

# Columnas que son IDs o índices (excluir de features)
KW_ID = [
    "id", "folio", "numero", "num", "indice", "index",
    "order", "orden", "pedido", "factura_num",
]


# ── Detección automática ───────────────────────────────────────────────────────

def _detectar_target(df: pd.DataFrame) -> str:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        raise ValueError("El dataset no contiene columnas numéricas para predecir.")

    # 1. Buscar por palabras clave (nombre normalizado)
    for col in num_cols:
        col_n = _norm(col)
        if any(kw in col_n for kw in KW_TARGET):
            return col

    # 2. Excluir columnas que parecen IDs (cardinalidad ≈ número de filas)
    candidatas = [
        c for c in num_cols
        if not any(kw in _norm(c) for kw in KW_ID)
        and df[c].nunique() > 1
    ]
    if not candidatas:
        candidatas = num_cols

    # 3. Columna con mayor coeficiente de variación (más "interesante")
    cv = (df[candidatas].std() / (df[candidatas].mean().abs() + 1e-9))
    return cv.idxmax()


def _detectar_grupo(df: pd.DataFrame, target: str) -> str | None:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != target]
    if not cat_cols:
        return None

    # 1. Buscar por palabras clave
    for col in cat_cols:
        col_n = _norm(col)
        if any(kw in col_n for kw in KW_GRUPO):
            return col

    # 2. La categórica con mayor cardinalidad (más grupos = más informativa)
    return max(cat_cols, key=lambda c: df[c].nunique())


# ── Agente principal ───────────────────────────────────────────────────────────

def run(df_clean: pd.DataFrame) -> dict:
    results = {}

    # ── Detectar target ────────────────────────────────────────────────────
    target = _detectar_target(df_clean)
    results["target_detectado"] = target

    # ── Construir features ─────────────────────────────────────────────────
    # Numéricas + columnas codificadas (_cod), sin el target ni IDs
    num_feats = df_clean.select_dtypes(include="number").columns.tolist()
    cod_feats = [c for c in df_clean.columns if c.endswith("_cod")]

    feature_cols = list(dict.fromkeys([
        c for c in num_feats + cod_feats
        if c != target
        and df_clean[c].nunique() > 1
        and not any(kw in _norm(c) for kw in KW_ID)
    ]))

    if not feature_cols:
        return {"error": "No hay features suficientes para entrenar el modelo."}

    results["features_usadas"] = feature_cols

    X = df_clean[feature_cols].fillna(0)
    y = df_clean[target]

    # ── División entrenamiento / prueba ────────────────────────────────────
    test_size = 0.2 if len(X) >= 20 else 0.3
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # ── Modelo 1: Regresión Lineal ─────────────────────────────────────────
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    rmse_lr = float(np.sqrt(mean_squared_error(y_test, lr.predict(X_test))))

    # ── Modelo 2: Árbol de Decisión ────────────────────────────────────────
    profundidad = max(3, min(8, len(feature_cols)))
    dt = DecisionTreeRegressor(max_depth=profundidad, random_state=42)
    dt.fit(X_train, y_train)
    rmse_dt = float(np.sqrt(mean_squared_error(y_test, dt.predict(X_test))))

    results["metricas"] = {
        "Regresión Lineal":  {"RMSE": round(rmse_lr, 4)},
        "Árbol de Decisión": {"RMSE": round(rmse_dt, 4)},
    }

    # ── Selección automática del mejor modelo ──────────────────────────────
    if rmse_lr <= rmse_dt:
        mejor_nombre  = "Regresión Lineal"
        mejor         = lr
        y_pred_test   = lr.predict(X_test)
    else:
        mejor_nombre  = "Árbol de Decisión"
        mejor         = dt
        y_pred_test   = dt.predict(X_test)

    results["mejor_modelo"] = mejor_nombre
    results["mejor_rmse"]   = round(min(rmse_lr, rmse_dt), 4)

    # ── Predicciones agrupadas ─────────────────────────────────────────────
    y_all = mejor.predict(X)
    pred_df = df_clean[[target]].copy()
    pred_df["predicho"] = y_all.round(4)

    grupo_col = _detectar_grupo(df_clean, target)
    results["columna_grupo"] = grupo_col

    if grupo_col and grupo_col in df_clean.columns:
        pred_df["_grupo"] = df_clean[grupo_col].values
        resumen = (
            pred_df.groupby("_grupo")
            .agg(
                total_real    =(target,     "sum"),
                total_pred    =("predicho", "sum"),
                promedio_real =(target,     "mean"),
                promedio_pred =("predicho", "mean"),
                registros     =(target,     "count"),
            )
            .sort_values("total_pred", ascending=False)
            .head(20)
            .round(4)
            .reset_index()
            .rename(columns={
                "_grupo":        grupo_col,
                "total_real":    "Total Real",
                "total_pred":    "Total Predicho",
                "promedio_real": "Promedio Real",
                "promedio_pred": "Promedio Predicho",
                "registros":     "Registros",
            })
        )
    else:
        resumen = pred_df.head(20).reset_index().rename(columns={
            "index":    "Índice",
            target:     "Valor Real",
            "predicho": "Valor Predicho",
        })

    results["predicciones_agrupadas"] = resumen.to_dict(orient="records")

    # ── Gráficas ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Scatter Real vs Predicho
    muestra = min(100, len(y_test))
    idx     = np.random.choice(len(y_test), muestra, replace=False)
    yt = np.array(y_test)[idx]
    yp = y_pred_test[idx]

    axes[0].scatter(yt, yp, alpha=0.6, color="#6C63FF",
                    edgecolors="white", s=50)
    mn, mx = float(np.array(y_test).min()), float(np.array(y_test).max())
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Ideal")
    axes[0].set_xlabel(f"{target} (real)")
    axes[0].set_ylabel(f"{target} (predicho)")
    axes[0].set_title(f"Real vs Predicho\n({mejor_nombre})", fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    # Barras comparativa RMSE
    nombres_barras = ["Regresión\nLineal", "Árbol de\nDecisión"]
    rmses          = [rmse_lr, rmse_dt]
    colores_barras = ["#4ECDC4" if r == min(rmses) else "#FF6B6B" for r in rmses]
    barras = axes[1].bar(nombres_barras, rmses,
                         color=colores_barras, edgecolor="white", width=0.4)
    offset = max(rmses) * 0.015
    for bar, val in zip(barras, rmses):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{val:.4f}", ha="center", fontweight="bold", fontsize=10,
        )
    axes[1].set_title("Comparativa de RMSE\n(menor es mejor)", fontweight="bold")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    results["grafica_b64"] = _b64(fig)

    return results
