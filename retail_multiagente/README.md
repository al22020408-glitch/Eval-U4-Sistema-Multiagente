# 🛍️ Sistema Multiagente para Análisis y Predicción de Ventas en Retail

> **TecNM Campus Apatzingán · Ingeniería en Sistemas Computacionales**  
> Unidad IV · Inteligencia Artificial · Mayo 2026  
> Dr. Omar Jehovani López Orozco

---

## 📐 Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    INTERFAZ WEB (Streamlit)                  │
│                          app.py                             │
└────────────────────────┬────────────────────────────────────┘
                         │  archivo .csv / .xlsx
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  AGENTE ORQUESTADOR                         │
│                    orchestrator.py                          │
│   Coordina la ejecución secuencial de los 3 agentes         │
└───────┬─────────────────┬────────────────────┬─────────────┘
        │                 │                    │
        ▼                 ▼                    ▼
┌──────────────┐  ┌──────────────┐   ┌──────────────────────┐
│  AGENTE 1    │  │  AGENTE 2    │   │      AGENTE 3        │
│     EDA      │→ │  Limpieza    │→  │  Modelo Predictivo   │
│ agent1_eda   │  │agent2_clean  │   │   agent3_model       │
│              │  │              │   │                      │
│ • Estadístic.│  │ • Imputación │   │ • Regresión Lineal   │
│ • Histograma │  │ • Duplicados │   │ • Árbol de Decisión  │
│ • Correlación│  │ • Codificac. │   │ • Comparar RMSE      │
│              │  │              │   │ • Pred. por SKU      │
└──────────────┘  └──────────────┘   └──────────────────────┘
```

---

## 📁 Estructura del Repositorio

```
retail_multiagente/
│
├── app.py                        # Interfaz web (Streamlit)
├── orchestrator.py               # Agente Orquestador
│
├── agents/
│   ├── __init__.py
│   ├── agent1_eda.py             # Agente 1 - Análisis Exploratorio
│   ├── agent2_cleaning.py        # Agente 2 - Limpieza de Datos
│   └── agent3_model.py           # Agente 3 - Modelo Predictivo
│
├── data/
│   └── women_clothing_ecommerce_sales.csv   # Dataset de ejemplo
│
├── requirements.txt              # Dependencias Python
└── README.md                     # Este archivo
```

---

## ⚙️ Instalación y Ejecución Local

### Prerrequisitos

- Python **3.10** o superior
- `pip` actualizado

### Paso 1 — Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/retail_multiagente.git
cd retail_multiagente
```

### Paso 2 — Crear entorno virtual (recomendado)

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Paso 3 — Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Paso 4 — Ejecutar la interfaz web

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en:

```
http://localhost:8501
```

### Paso 5 — Usar la aplicación

1. En la interfaz, haz clic en **"Browse files"** (o arrastra tu archivo).
2. Carga el dataset de ejemplo ubicado en `data/women_clothing_ecommerce_sales.csv`  
   (o cualquier archivo `.csv` / `.xlsx` con columnas de ventas de retail).
3. El pipeline se ejecuta automáticamente y muestra los resultados de los tres agentes.

---

## 🤖 Descripción de los Agentes

### Agente Orquestador (`orchestrator.py`)
Recibe el archivo cargado por el usuario, invoca los tres agentes en orden secuencial y consolida todos los resultados para devolverlos a la interfaz web.

### Agente 1 — EDA (`agents/agent1_eda.py`)
| Función | Detalle |
|---------|---------|
| Estadísticas descriptivas | Media, mediana, desviación estándar por columna numérica |
| Conteo de nulos | Número y porcentaje de valores faltantes |
| Histograma | Distribución de la columna `revenue` |
| Matriz de correlación | Heatmap de correlaciones entre variables numéricas |
| Distribución por color | Agrupación de ventas por categoría de color |

### Agente 2 — Limpieza (`agents/agent2_cleaning.py`)
| Función | Detalle |
|---------|---------|
| Eliminación de duplicados | `drop_duplicates()` exactos |
| Imputación de nulos | Moda para categóricas · Mediana para numéricas |
| Codificación | `LabelEncoder` sobre todas las columnas `object` (excepto `order_date`) |

### Agente 3 — Predicción (`agents/agent3_model.py`)
| Función | Detalle |
|---------|---------|
| Modelos | Regresión Lineal · Árbol de Decisión (`max_depth=5`) |
| Split | 80 % train · 20 % test |
| Métrica | RMSE — se selecciona automáticamente el modelo con menor RMSE |
| Predicciones | Revenue estimado agrupado por SKU (Top 15) |
| Visualización | Scatter Real vs Predicho · Barplot comparativo RMSE |

---

## 📦 Dependencias

| Librería | Uso |
|----------|-----|
| `streamlit` | Interfaz web |
| `pandas` | Manipulación de datos |
| `numpy` | Cálculos numéricos |
| `matplotlib` / `seaborn` | Visualizaciones |
| `scikit-learn` | Modelos ML, codificación, métricas |
| `openpyxl` / `xlrd` | Lectura de archivos Excel |

---

## 👥 Equipo

| Nombre | Rol |
|--------|-----|
| Integrante 1 | Agente 1 (EDA) + Interfaz |
| Integrante 2 | Agente 2 (Limpieza) + Orquestador |
| Integrante 3 | Agente 3 (Predicción) + README |

---

## 📚 Referencias

1. Google. (2024). *Agent Development Kit (ADK) Documentation*. https://google.github.io/adk-docs  
2. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.  
3. Stanford University. (2023). *CS229: Machine Learning Course Notes*. https://cs229.stanford.edu  
