"""
Agente 2 — Limpieza de Datos
Compatible con cualquier CSV/XLSX en español o inglés.
"""

import unicodedata
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def _normalizar(texto: str) -> str:
    """Quita acentos y pasa a minúsculas para comparaciones robustas."""
    return "".join(
        c for c in unicodedata.normalize("NFD", texto.lower())
        if unicodedata.category(c) != "Mn"
    )


def _es_fecha(serie: pd.Series) -> bool:
    """
    Devuelve True si la columna parece contener fechas/horas.
    Detecta formatos comunes: 2024-01-31, 31/01/2024, 2024/6/1 16:05:00, etc.
    """
    muestra = serie.dropna().head(30).astype(str)
    if len(muestra) == 0:
        return False
    patron = (
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"   # ISO y variantes
        r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}" # DD/MM/YYYY y variantes
        r"|\d{4}-\d{2}-\d{2}T"            # ISO 8601
    )
    hits = muestra.str.contains(patron, regex=True).sum()
    return hits >= len(muestra) * 0.5


def run(df: pd.DataFrame) -> dict:
    df_clean = df.copy()
    reporte  = {}

    # ── Estado inicial ─────────────────────────────────────────────────────
    reporte["filas_inicial"] = len(df_clean)
    reporte["nulos_inicial"] = df_clean.isnull().sum().to_dict()

    # ── 1. Eliminar duplicados ─────────────────────────────────────────────
    n_dup = int(df_clean.duplicated().sum())
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    reporte["duplicados_eliminados"] = n_dup
    reporte["filas_tras_dedup"]      = len(df_clean)

    # ── 2. Clasificar columnas ─────────────────────────────────────────────
    num_cols = df_clean.select_dtypes(include="number").columns.tolist()
    cat_cols = df_clean.select_dtypes(include=["object", "category"]).columns.tolist()

    # Detectar columnas con fechas para omitirlas de la codificación
    date_like = [c for c in cat_cols if _es_fecha(df_clean[c])]
    cols_a_codificar = [c for c in cat_cols if c not in date_like]

    # ── 3. Imputación de nulos ─────────────────────────────────────────────
    imputaciones = {}

    for col in num_cols:
        n = int(df_clean[col].isnull().sum())
        if n:
            val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(val)
            imputaciones[col] = {
                "nulos": n,
                "estrategia": f"mediana = {val:.4f}",
            }

    for col in cat_cols:
        n = int(df_clean[col].isnull().sum())
        if n:
            val = (
                df_clean[col].mode()[0]
                if not df_clean[col].mode().empty
                else "desconocido"
            )
            df_clean[col] = df_clean[col].fillna(val)
            imputaciones[col] = {
                "nulos": n,
                "estrategia": f"moda = '{val}'",
            }

    reporte["imputaciones"] = imputaciones
    reporte["nulos_final"]  = df_clean.isnull().sum().to_dict()

    # ── 4. Codificación de variables categóricas ───────────────────────────
    encoders     = {}
    encoded_info = {}

    for col in cols_a_codificar:
        le = LabelEncoder()
        df_clean[col + "_cod"] = le.fit_transform(df_clean[col].astype(str))
        encoders[col] = le
        encoded_info[col] = {
            "columna_nueva": col + "_cod",
            "clases":        list(le.classes_),
        }

    reporte["codificacion"]         = encoded_info
    reporte["columnas_fecha_omit"]  = date_like

    return {
        "df_clean": df_clean,
        "encoders": encoders,
        "reporte":  reporte,
    }
