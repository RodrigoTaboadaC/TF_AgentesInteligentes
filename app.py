#PREDICTOR DE RENDIMIENTO ACADÉMICO (GPA)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Agente Inteligente — Predicción de GPA",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos (CSS) 
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-top: 0px;
    }
    .agent-box, .agent-box p {
        background-color: #EFF6FF;
        border-left: 5px solid #2563EB;
        padding: 14px 18px;
        border-radius: 8px;
        font-size: 0.95rem;
        color: #1E293B !important;
    }
    .result-box, .result-box p, .result-box h1, .result-box h3 {
        background-color: #F0FDF4;
        border-left: 5px solid #16A34A;
        padding: 18px 22px;
        border-radius: 10px;
        color: #1E293B !important;
    }
    .warning-box, .warning-box p {
        background-color: #FFFBEB;
        border-left: 5px solid #D97706;
        padding: 14px 18px;
        border-radius: 8px;
        font-size: 0.9rem;
        color: #78350F !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# DICCIONARIOS DE TRADUCCIÓN
MAPA_GENERO = {0: "Masculino", 1: "Femenino"}
MAPA_ETNIA = {0: "Caucásico", 1: "Afroamericano", 2: "Asiático", 3: "Otro"}
MAPA_EDUCACION_PADRES = {
    0: "Ninguna",
    1: "Secundaria completa",
    2: "Educación superior incompleta",
    3: "Licenciatura / Bachillerato",
    4: "Posgrado",
}
MAPA_APOYO_PARENTAL = {
    0: "Ninguno",
    1: "Bajo",
    2: "Moderado",
    3: "Alto",
    4: "Muy alto",
}
MAPA_SI_NO = {0: "No", 1: "Sí"}

NOMBRES_VARIABLES_ES = {
    "Age": "Edad",
    "Gender": "Género",
    "Ethnicity": "Etnia",
    "ParentalEducation": "Educación de los padres",
    "StudyTimeWeekly": "Horas de estudio semanales",
    "Absences": "Ausencias",
    "Tutoring": "Tutoría",
    "ParentalSupport": "Apoyo parental",
    "Extracurricular": "Actividades extracurriculares",
    "Sports": "Deportes",
    "Music": "Música",
    "Volunteering": "Voluntariado",
}

# CARGA DE DATOS Y ENTRENAMIENTO DEL AGENTE 
@st.cache_resource
def entrenar_agente():
    """
    Entrena el agente inteligente (Random Forest Regressor) que actúa como
    motor de predicción del GPA estudiantil.
    """
    df = pd.read_csv("Student_performance_data.csv")

    # Se elimina la columna "StudentID" y "GradeClass" del conjunto de datos,
    # Porque no son relevantes para la predicción del GPA y podrían introducir ruido en el modelo.
    X = df.drop(columns=["StudentID", "GPA", "GradeClass"])
    y = df["GPA"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    metricas = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }

    importancias = pd.Series(modelo.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )

    return modelo, X.columns.tolist(), metricas, importancias, df


modelo_rf, columnas_modelo, metricas, importancias, df_original = entrenar_agente()

# ENCABEZADO
col_logo, col_titulo = st.columns([0.07, 0.93])
with col_logo:
    st.markdown("## 🎓")
with col_titulo:
    st.markdown('<p class="main-header">Agente Inteligente — Predictor de Rendimiento Académico (GPA)</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Trabajo Final · Curso Agentes Inteligentes · USIL 2026-1</p>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="agent-box">
    🤖 <b>¿Qué hace este agente?</b> Este agente inteligente <b>percibe</b> las características
    académicas, demográficas y de comportamiento de un estudiante, y <b>actúa</b> generando una
    predicción de su <b>GPA (Promedio Ponderado)</b> mediante un modelo de <b>Random Forest Regressor</b>
    entrenado sobre datos reales de desempeño estudiantil.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# NAVEGACIÓN POR PESTAÑAS
tab_prediccion, tab_modelo, tab_datos = st.tabs(
    ["🔮 Predicción del GPA", "📊 Desempeño del Agente", "📁 Explorar Dataset"]
)

# TAB 1 — PREDICCIÓN
with tab_prediccion:
    st.subheader("📝 Ingresa las características del estudiante")
    st.caption(
        "El agente procesará esta información (sensores) y generará una predicción del GPA (actuador)."
    )

    with st.form("formulario_prediccion_gpa"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**👤 Datos demográficos**")
            edad = st.slider("Edad", min_value=15, max_value=18, value=16)
            genero_es = st.selectbox("Género", options=list(MAPA_GENERO.values()))
            etnia_es = st.selectbox("Etnia", options=list(MAPA_ETNIA.values()))
            educacion_padres_es = st.selectbox(
                "Nivel educativo de los padres", options=list(MAPA_EDUCACION_PADRES.values())
            )

        with col2:
            st.markdown("**📚 Hábitos académicos**")
            horas_estudio = st.slider(
                "Horas de estudio semanales", min_value=0.0, max_value=20.0, value=10.0, step=0.5
            )
            ausencias = st.slider("Número de ausencias (al año)", min_value=0, max_value=29, value=5)
            tutoria_es = st.selectbox("¿Recibe tutoría académica?", options=list(MAPA_SI_NO.values()))
            apoyo_parental_es = st.selectbox(
                "Nivel de apoyo parental", options=list(MAPA_APOYO_PARENTAL.values())
            )

        with col3:
            st.markdown("**🏀 Actividades extracurriculares**")
            extracurricular_es = st.selectbox(
                "¿Participa en actividades extracurriculares?", options=list(MAPA_SI_NO.values())
            )
            deportes_es = st.selectbox("¿Practica algún deporte?", options=list(MAPA_SI_NO.values()))
            musica_es = st.selectbox("¿Participa en actividades musicales?", options=list(MAPA_SI_NO.values()))
            voluntariado_es = st.selectbox("¿Realiza voluntariado?", options=list(MAPA_SI_NO.values()))

        enviado = st.form_submit_button("🚀 Predecir GPA", use_container_width=True)

    if enviado:
        # Traducir las selecciones en español de vuelta a los códigos del dataset
        inv_genero = {v: k for k, v in MAPA_GENERO.items()}
        inv_etnia = {v: k for k, v in MAPA_ETNIA.items()}
        inv_educacion = {v: k for k, v in MAPA_EDUCACION_PADRES.items()}
        inv_apoyo = {v: k for k, v in MAPA_APOYO_PARENTAL.items()}
        inv_sino = {v: k for k, v in MAPA_SI_NO.items()}

        entrada = pd.DataFrame([{
            "Age": edad,
            "Gender": inv_genero[genero_es],
            "Ethnicity": inv_etnia[etnia_es],
            "ParentalEducation": inv_educacion[educacion_padres_es],
            "StudyTimeWeekly": horas_estudio,
            "Absences": ausencias,
            "Tutoring": inv_sino[tutoria_es],
            "ParentalSupport": inv_apoyo[apoyo_parental_es],
            "Extracurricular": inv_sino[extracurricular_es],
            "Sports": inv_sino[deportes_es],
            "Music": inv_sino[musica_es],
            "Volunteering": inv_sino[voluntariado_es],
        }])
        entrada = entrada.reindex(columns=columnas_modelo, fill_value=0)

        gpa_predicho = float(modelo_rf.predict(entrada)[0])
        gpa_predicho = max(0.0, min(4.0, gpa_predicho))  # el GPA real está acotado [0, 4]

        st.divider()
        st.subheader("💡 Resultado de la inferencia")

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.markdown(
                f"""
                <div class="result-box">
                <h3 style="margin-top:0;">🎯 GPA Estimado</h3>
                <h1 style="color:#16A34A; font-size:3rem; margin:0;">{gpa_predicho:.2f}</h1>
                <p style="color:#475569;">en escala de 0.0 a 4.0</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_b:
            if gpa_predicho >= 3.5:
                nivel, color, mensaje = "Excelente", "🟢", "El estudiante muestra un desempeño académico sobresaliente."
            elif gpa_predicho >= 3.0:
                nivel, color, mensaje = "Bueno", "🟢", "El estudiante presenta un desempeño académico sólido."
            elif gpa_predicho >= 2.0:
                nivel, color, mensaje = "Regular", "🟡", "El estudiante se encuentra en un nivel intermedio; podría beneficiarse de acompañamiento adicional."
            else:
                nivel, color, mensaje = "En riesgo académico", "🔴", "El estudiante presenta indicadores de riesgo académico y podría requerir intervención temprana (tutoría, refuerzo o seguimiento)."

            st.markdown(
                f"""
                **{color} Categoría: {nivel}**

                {mensaje}

                **¿Qué significa esta predicción?** El agente, basándose en patrones aprendidos de
                cientos de estudiantes con características similares, estima que este perfil
                obtendría un GPA de **{gpa_predicho:.2f}**. La variable con mayor influencia en el modelo
                es el número de **ausencias**, por lo que reducir el ausentismo suele ser la
                palanca más efectiva para mejorar el rendimiento académico estimado.
                """
            )

        st.markdown(
            """
            <div class="warning-box">
            ⚠️ <b>Nota de transparencia:</b> Esta predicción es una estimación estadística basada en
            datos históricos y no debe usarse como única base para decisiones académicas sobre un
            estudiante real.
            </div>
            """,
            unsafe_allow_html=True,
        )

# TAB 2 — DESEMPEÑO DEL AGENTE (métricas + feature importance)
with tab_modelo:
    st.subheader("📊 ¿Qué tan confiable es el agente?")
    st.caption("Métricas calculadas sobre el conjunto de prueba (20% de los datos, no usado en entrenamiento).")

    col1, col2, col3 = st.columns(3)
    col1.metric("R² (Coef. de determinación)", f"{metricas['R2']:.4f}")
    col2.metric("RMSE", f"{metricas['RMSE']:.4f}")
    col3.metric("MAE", f"{metricas['MAE']:.4f}")

    st.markdown(
        """
        <div class="warning-box">
        ⚠️ <b>Importante sobre estos resultados:</b> el R² es muy alto principalmente porque la
        variable <b>Ausencias</b> tiene una correlación de <b>-0.92</b> con el GPA en este dataset:
        a más ausencias, el GPA cae de forma casi lineal. Esto significa que gran parte del poder
        predictivo del agente depende de esta única variable, y no únicamente de relaciones
        complejas entre los 12 factores analizados.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("🔍 ¿En qué variables se basa el agente para predecir?")
    st.caption("Importancia de variables (feature importance) calculada por el Random Forest.")

    importancias_es = importancias.rename(index=NOMBRES_VARIABLES_ES).reset_index()
    importancias_es.columns = ["Variable", "Importancia"]

    fig_importancia = px.bar(
        importancias_es.sort_values("Importancia"),
        x="Importancia",
        y="Variable",
        orientation="h",
        title="Importancia de cada variable en el modelo Random Forest",
        color="Importancia",
        color_continuous_scale="Blues",
    )
    fig_importancia.update_layout(showlegend=False)
    st.plotly_chart(fig_importancia, use_container_width=True)

    st.caption(
        "💡 Como se observa, **Ausencias** domina ampliamente la importancia del modelo, seguida "
        "habitualmente por **horas de estudio semanales** y **apoyo parental**, mientras que variables "
        "demográficas como edad o etnia aportan muy poco al poder predictivo."
    )

# TAB 3 — EXPLORACIÓN DEL DATASET
with tab_datos:
    st.subheader("📁 Dataset utilizado: Student Performance Data")
    st.caption("Conjunto de datos sobre el cual el agente fue entrenado.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de estudiantes", f"{len(df_original):,}")
    col2.metric("GPA promedio", f"{df_original['GPA'].mean():.2f}")
    col3.metric("Ausencias promedio", f"{df_original['Absences'].mean():.1f}")

    col4, col5 = st.columns(2)
    with col4:
        fig_hist = px.histogram(
            df_original, x="GPA", nbins=25,
            title="Distribución del GPA en el dataset",
            labels={"GPA": "GPA"},
            color_discrete_sequence=["#2563EB"],
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    with col5:
        fig_scatter = px.scatter(
            df_original, x="Absences", y="GPA",
            title="Relación entre ausencias y GPA",
            labels={"Absences": "Ausencias", "GPA": "GPA"},
            opacity=0.5,
            trendline="ols",
            color_discrete_sequence=["#16A34A"],
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with st.expander("🔍 Ver muestra de los datos originales"):
        st.dataframe(df_original.head(20), use_container_width=True)

st.divider()
st.caption(
    "Agente Inteligente — Predicción de GPA | Trabajo Final | Curso Agentes Inteligentes | USIL 2026-1"
)
