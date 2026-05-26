"""
Agente Orquestador
Coordina la ejecución secuencial de los tres agentes.
Compatible con cualquier CSV/XLSX en español o inglés.
"""

import io
import pandas as pd
from agents import agent1_eda, agent2_cleaning, agent3_model


# Separadores a probar en orden de frecuencia
SEPARADORES = [",", ";", "\t", "|"]


def cargar_dataframe(archivo) -> pd.DataFrame:
    nombre = archivo.name.lower()

    if nombre.endswith(".csv"):
        for sep in SEPARADORES:
            try:
                archivo.seek(0)
                df = pd.read_csv(archivo, sep=sep, encoding="utf-8")
                if df.shape[1] > 1:
                    return df
            except UnicodeDecodeError:
                pass
            except Exception:
                continue

        # Segundo intento con encoding latin-1 (común en archivos del INEGI / SAT)
        for sep in SEPARADORES:
            try:
                archivo.seek(0)
                df = pd.read_csv(archivo, sep=sep, encoding="latin-1")
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue

        archivo.seek(0)
        return pd.read_csv(archivo)  # fallback pandas

    elif nombre.endswith((".xlsx", ".xls")):
        return pd.read_excel(archivo)

    else:
        raise ValueError(
            f"Formato no soportado: '{archivo.name}'.  \n"
            "Por favor usa un archivo con extensión **.csv** o **.xlsx**."
        )


def ejecutar_pipeline(archivo) -> dict:
    resultado = {
        "df_raw":     None,
        "eda":        None,
        "limpieza":   None,
        "prediccion": None,
        "error":      None,
        "avisos":     [],
    }

    # ── Carga ──────────────────────────────────────────────────────────────
    try:
        df_raw = cargar_dataframe(archivo)
        if df_raw.empty:
            resultado["error"] = "El archivo está vacío. Por favor sube un archivo con datos."
            return resultado
        if df_raw.shape[1] < 2:
            resultado["error"] = (
                "El archivo necesita al menos **2 columnas** para ejecutar el análisis."
            )
            return resultado
        resultado["df_raw"] = df_raw
    except Exception as e:
        resultado["error"] = f"**Error al cargar el archivo:** {e}"
        return resultado

    # ── Agente 1: EDA ──────────────────────────────────────────────────────
    try:
        resultado["eda"] = agent1_eda.run(df_raw)
    except Exception as e:
        resultado["error"] = f"**Error en Agente 1 (EDA):** {e}"
        return resultado

    # ── Agente 2: Limpieza ─────────────────────────────────────────────────
    try:
        res2 = agent2_cleaning.run(df_raw)
        resultado["limpieza"] = res2
        df_clean = res2["df_clean"]
    except Exception as e:
        resultado["error"] = f"**Error en Agente 2 (Limpieza):** {e}"
        return resultado

    # ── Agente 3: Predicción ───────────────────────────────────────────────
    try:
        resultado["prediccion"] = agent3_model.run(df_clean)
    except Exception as e:
        resultado["avisos"].append(f"**Aviso en Agente 3 (Predicción):** {e}")
        resultado["prediccion"] = {"error": str(e)}

    return resultado
