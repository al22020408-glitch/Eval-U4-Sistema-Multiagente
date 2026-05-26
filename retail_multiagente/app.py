"""
app.py — Interfaz Web Streamlit
Sistema Multiagente para Análisis y Predicción de Datos
100 % en español · Compatible con cualquier CSV/XLSX
"""

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Sistema Multiagente · Análisis de Datos",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main { background-color:#F8F9FE; }
.metric-card {
    background:white; border-radius:12px; padding:16px 20px;
    box-shadow:0 2px 8px rgba(0,0,0,.07); text-align:center;
    border-left:5px solid #6C63FF;
}
.metric-card h4 { margin:0 0 4px; color:#666; font-size:.82rem; text-transform:uppercase; letter-spacing:.04em; }
.metric-card p  { margin:0; font-size:1.5rem; font-weight:700; color:#1a1a2e; }
.agent-badge {
    display:inline-block; padding:5px 14px; border-radius:20px;
    font-size:.85rem; font-weight:600; margin-bottom:8px;
}
.badge-eda     { background:#EDE9FE; color:#6C63FF; }
.badge-clean   { background:#D1FAE5; color:#059669; }
.badge-predict { background:#FEF3C7; color:#D97706; }
.badge-orch    { background:#DBEAFE; color:#2563EB; }
.winner-box {
    background:linear-gradient(135deg,#6C63FF,#a78bfa);
    color:white; border-radius:12px; padding:14px 22px;
    font-size:1.05rem; font-weight:600; text-align:center;
    margin-bottom:12px;
}
.etiqueta {
    background:#F0F4FF; border:1px solid #c7d2fe; border-radius:8px;
    padding:5px 11px; font-size:.82rem; color:#4338ca;
    display:inline-block; margin:2px;
}
.etiqueta-cat {
    background:#F0FFF4; border:1px solid #86efac; border-radius:8px;
    padding:5px 11px; font-size:.82rem; color:#15803d;
    display:inline-block; margin:2px;
}
</style>
""", unsafe_allow_html=True)


def mostrar_imagen(b64: str, pie: str = ""):
    st.markdown(
        f'<img src="data:image/png;base64,{b64}" '
        f'style="width:100%;border-radius:10px;margin-bottom:4px;">',
        unsafe_allow_html=True,
    )
    if pie:
        st.caption(pie)


# ── BARRA LATERAL ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Sistema Multiagente")
    st.markdown(
        "**TecNM Campus Apatzingán**  \n"
        "Ingeniería en Sistemas Computacionales  \n"
        "Inteligencia Artificial · 2026"
    )
    st.divider()
    st.markdown("#### Flujo del Pipeline")
    for badge, etiqueta in [
        ("badge-orch",    "🔀 Orquestador"),
        ("badge-eda",     "📊 Agente 1 · EDA"),
        ("badge-clean",   "🧹 Agente 2 · Limpieza"),
        ("badge-predict", "🔮 Agente 3 · Predicción"),
    ]:
        st.markdown(
            f"<div class='agent-badge {badge}'>{etiqueta}</div>",
            unsafe_allow_html=True,
        )
    st.divider()
    st.markdown(
        "**Formatos soportados:** `.csv` · `.xlsx`  \n"
        "**Separadores CSV:** `,` `;` `TAB` `|`  \n"
        "**Codificaciones:** UTF-8 · Latin-1"
    )
    st.divider()
    st.caption("Dr. Omar Jehovani López Orozco · Mayo 2026")


# ── ENCABEZADO ────────────────────────────────────────────────────────────────
st.title("🤖 Sistema Multiagente — Análisis y Predicción de Datos")
st.markdown(
    "Carga **cualquier archivo CSV o XLSX** en español o inglés. "
    "El sistema detecta automáticamente las columnas, limpia los datos "
    "y entrena un modelo predictivo sin configuración manual."
)

# ── CARGA DE ARCHIVO ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📂 Cargar Dataset")

col_up, col_pista = st.columns([2, 1])
with col_up:
    archivo = st.file_uploader(
        "Selecciona un archivo CSV o XLSX",
        type=["csv", "xlsx", "xls"],
        help="Soporta separadores , ; TAB | y codificaciones UTF-8 y Latin-1 (ANSI).",
    )
with col_pista:
    st.info(
        "**El sistema acepta cualquier estructura.**  \n\n"
        "Detecta automáticamente:  \n"
        "• Variable objetivo (numérica)  \n"
        "• Variables categóricas  \n"
        "• Columnas con fechas  \n"
        "• Separador del CSV  \n"
        "• Codificación del archivo"
    )

# ── PIPELINE PRINCIPAL ────────────────────────────────────────────────────────
if archivo is not None:

    try:
        from orchestrator import ejecutar_pipeline
    except ImportError as e:
        st.error(f"No se pudo importar el orquestador: {e}")
        st.stop()

    with st.spinner("⚙️ Ejecutando pipeline… por favor espera."):
        res = ejecutar_pipeline(archivo)

    if res.get("error"):
        st.error(f"🚨 {res['error']}")
        st.stop()

    for aviso in res.get("avisos", []):
        st.warning(f"⚠️ {aviso}")

    st.success("✅ Pipeline ejecutado correctamente.")

    df_raw:   pd.DataFrame = res["df_raw"]
    eda:      dict         = res["eda"]
    limpieza: dict         = res["limpieza"]
    pred:     dict         = res["prediccion"]
    stats                  = eda["estadisticas"]
    num_cols               = stats["num_cols"]
    cat_cols               = stats["cat_cols"]

    # ── Métricas globales ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📌 Resumen del Dataset")

    total_nulos = sum(stats["nulos"].values())
    k1, k2, k3, k4, k5 = st.columns(5)
    for col_st, titulo, valor in [
        (k1, "Filas totales",       f"{df_raw.shape[0]:,}"),
        (k2, "Columnas",            str(df_raw.shape[1])),
        (k3, "Columnas numéricas",  str(len(num_cols))),
        (k4, "Columnas de texto",   str(len(cat_cols))),
        (k5, "Valores nulos",       f"{total_nulos:,}"),
    ]:
        col_st.markdown(
            f"<div class='metric-card'><h4>{titulo}</h4><p>{valor}</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### 🗂️ Vista previa (primeras 8 filas)")
    st.dataframe(df_raw.head(8), use_container_width=True)

    with st.expander("🔍 Ver columnas detectadas por tipo"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Columnas numéricas**")
            for c in num_cols:
                st.markdown(f"<span class='etiqueta'>🔢 {c}</span>", unsafe_allow_html=True)
        with c2:
            st.markdown("**Columnas de texto / categóricas**")
            for c in cat_cols:
                st.markdown(f"<span class='etiqueta-cat'>🏷️ {c}</span>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # AGENTE 1 · EDA
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown(
        "<div class='agent-badge badge-eda' style='font-size:1rem;padding:7px 18px;'>"
        "📊 Agente 1 — Análisis Exploratorio de Datos (EDA)</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Ver resultados del Agente 1", expanded=True):

        st.markdown("##### 📋 Estadísticas Descriptivas")
        if stats["describe"]:
            st.dataframe(
                pd.DataFrame(stats["describe"]).T,
                use_container_width=True,
            )
        else:
            st.info("No se encontraron columnas numéricas para describir.")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### 📐 Medianas")
            if stats["mediana"]:
                st.dataframe(
                    pd.DataFrame(
                        list(stats["mediana"].items()),
                        columns=["Columna", "Mediana"],
                    ),
                    use_container_width=True, hide_index=True,
                )
        with col_b:
            st.markdown("##### 🔍 Valores nulos por columna")
            st.dataframe(
                pd.DataFrame({
                    "Columna": list(stats["nulos"].keys()),
                    "Nulos":   list(stats["nulos"].values()),
                    "% Nulos": [f"{v:.1f}%" for v in stats["nulos_pct"].values()],
                }),
                use_container_width=True, hide_index=True,
            )

        st.markdown("##### 📈 Visualizaciones")
        g1, g2 = st.columns(2)
        with g1:
            if eda.get("histograma_b64"):
                mostrar_imagen(
                    eda["histograma_b64"],
                    "Histogramas de distribución de variables numéricas",
                )
            else:
                st.info("Sin columnas numéricas para generar histogramas.")
        with g2:
            if eda.get("correlacion_b64"):
                mostrar_imagen(
                    eda["correlacion_b64"],
                    "Mapa de calor: correlación entre variables numéricas",
                )
            else:
                st.info("Se necesitan al menos 2 columnas numéricas para la correlación.")

        if eda.get("cat_summary"):
            st.markdown("##### 🏷️ Valores más frecuentes por columna de texto")
            columnas_cat = list(eda["cat_summary"].items())
            # Mostrar de a 2 columnas por fila
            for i in range(0, len(columnas_cat), 2):
                fila = st.columns(2)
                for j, (col, vc) in enumerate(columnas_cat[i:i+2]):
                    with fila[j]:
                        st.markdown(f"**`{col}`**")
                        st.dataframe(
                            pd.DataFrame(
                                list(vc.items()), columns=["Valor", "Frecuencia"]
                            ),
                            use_container_width=True, hide_index=True,
                        )

    # ══════════════════════════════════════════════════════════════════════
    # AGENTE 2 · LIMPIEZA
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown(
        "<div class='agent-badge badge-clean' style='font-size:1rem;padding:7px 18px;'>"
        "🧹 Agente 2 — Limpieza de Datos</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Ver resultados del Agente 2", expanded=True):
        rep = limpieza["reporte"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Filas originales",      rep["filas_inicial"])
        m2.metric("Duplicados eliminados", rep["duplicados_eliminados"])
        m3.metric("Filas tras limpieza",   rep["filas_tras_dedup"])

        st.markdown("##### 💉 Imputaciones realizadas")
        if rep["imputaciones"]:
            st.dataframe(
                pd.DataFrame([
                    {
                        "Columna":            col,
                        "Valores nulos":      v["nulos"],
                        "Estrategia aplicada":v["estrategia"],
                    }
                    for col, v in rep["imputaciones"].items()
                ]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.success("✅ No se encontraron valores nulos. El dataset está limpio.")

        if rep.get("columnas_fecha_omit"):
            st.caption(
                "📅 Columnas con fechas detectadas (omitidas de la codificación): "
                + ", ".join(f"`{c}`" for c in rep["columnas_fecha_omit"])
            )

        st.markdown("##### 🔡 Codificación de columnas categóricas")
        if rep["codificacion"]:
            for col, info in rep["codificacion"].items():
                clases   = info["clases"]
                muestra  = ", ".join(f'"{v}"' for v in clases[:6])
                extra    = f" … ({len(clases)} clases en total)" if len(clases) > 6 else f" ({len(clases)} clases)"
                st.markdown(
                    f"**`{col}`** → columna `{info['columna_nueva']}` | {muestra}{extra}"
                )
        else:
            st.info("No se encontraron columnas categóricas para codificar.")

        st.markdown("##### 🗂️ Dataset limpio (primeras 8 filas)")
        st.dataframe(limpieza["df_clean"].head(8), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # AGENTE 3 · PREDICCIÓN
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown(
        "<div class='agent-badge badge-predict' style='font-size:1rem;padding:7px 18px;'>"
        "🔮 Agente 3 — Modelo Predictivo</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Ver resultados del Agente 3", expanded=True):

        if "error" in pred and len(pred) == 1:
            st.error(f"🚨 {pred['error']}")
        else:
            # Variable objetivo y features detectadas
            st.markdown(
                f"🎯 **Variable objetivo detectada:** "
                f"<span class='etiqueta' style='font-size:.9rem;'>{pred.get('target_detectado','?')}</span>",
                unsafe_allow_html=True,
            )
            feats = pred.get("features_usadas", [])
            st.markdown(
                "📥 **Variables de entrada (features):** " +
                " ".join(f"<span class='etiqueta'>{f}</span>" for f in feats),
                unsafe_allow_html=True,
            )

            st.markdown("---")

            # Modelo ganador
            st.markdown(
                f"<div class='winner-box'>🏆 Modelo seleccionado: "
                f"{pred['mejor_modelo']} &nbsp;|&nbsp; "
                f"RMSE: {pred['mejor_rmse']}</div>",
                unsafe_allow_html=True,
            )

            # Tabla comparativa
            st.markdown("##### 📊 Comparativa de modelos")
            st.dataframe(
                pd.DataFrame([
                    {
                        "Modelo":       k,
                        "RMSE":         v["RMSE"],
                        "Seleccionado": "✅ Ganador" if k == pred["mejor_modelo"] else "—",
                    }
                    for k, v in pred["metricas"].items()
                ]),
                use_container_width=True, hide_index=True,
            )

            # Gráficas
            st.markdown("##### 📈 Visualizaciones del modelo")
            if pred.get("grafica_b64"):
                mostrar_imagen(
                    pred["grafica_b64"],
                    "Izquierda: Valores reales vs predichos · Derecha: Comparativa de RMSE",
                )

            # Predicciones agrupadas
            grupo = pred.get("columna_grupo")
            titulo_pred = (
                f"##### 🏷️ Predicciones por `{grupo}` (top 20)"
                if grupo else "##### 🏷️ Predicciones individuales (muestra)"
            )
            st.markdown(titulo_pred)

            df_pred = pd.DataFrame(pred.get("predicciones_agrupadas", []))
            if not df_pred.empty:
                st.dataframe(df_pred, use_container_width=True, hide_index=True)
            else:
                st.info("No se generaron predicciones agrupadas.")

    # ── Pie de página ───────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "🤖 Sistema Multiagente · TecNM Campus Apatzingán · "
        "Inteligencia Artificial 2026 · Dr. Omar Jehovani López Orozco"
    )

# ── ESTADO VACÍO ──────────────────────────────────────────────────────────────
else:
    st.markdown("---")
    st.info(
        "👆 **Sube cualquier archivo** `.csv` o `.xlsx` para iniciar el análisis.  \n\n"
        "El sistema funciona con **columnas en español o inglés**, con o sin acentos, "
        "y detecta automáticamente separadores y codificaciones de archivo."
    )
    st.markdown("""
    ```
    Archivo CSV/XLSX  (cualquier estructura, en español o inglés)
          ↓
    [Orquestador]
      • Detecta separador: , ; TAB |
      • Detecta codificación: UTF-8 o Latin-1 (ANSI)
          ↓
    [Agente 1 · EDA]
      • Estadísticas descriptivas en español
      • Histograma por cada variable numérica
      • Matriz de correlación
      • Top-5 valores por columna de texto
          ↓
    [Agente 2 · Limpieza]
      • Elimina duplicados
      • Imputa nulos: mediana (numéricas) | moda (texto)
      • Detecta y omite columnas con fechas
      • Codifica categóricas con LabelEncoder → columna_cod
          ↓
    [Agente 3 · Predicción]
      • Detecta la variable objetivo por palabras clave en español/inglés
        (ingreso, ventas, total, monto, precio, revenue, sales, amount…)
      • Entrena Regresión Lineal y Árbol de Decisión
      • Compara con RMSE → selecciona el mejor automáticamente
      • Predicciones agrupadas por la columna más representativa
          ↓
    Resultados en pantalla, 100 % en español
    ```
    """)
