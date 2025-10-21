import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import zipfile
from datetime import datetime

# Importar TensorFlow con manejo de errores
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ---------------------------
# CONFIGURACIÓN
# ---------------------------
DB_PATH = "predictions.db"
MODEL_PATH = "model/keras_model.h5"
LABELS_PATH = "model/labels.txt"
IMAGE_SIZE = (224, 224)

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Sistema de Reconocimiento Facial",
    layout="wide",
    page_icon="🎯",
    initial_sidebar_state="expanded"
)

# Estilos CSS mejorados
st.markdown("""
<style>
    /* Sidebar personalizado */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Fondo principal con gradiente sutil */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Títulos personalizados */
    h1 {
        color: #1e3c72;
        font-weight: 800;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h2, h3 {
        color: #2c3e50;
        font-weight: 700;
    }
    
    /* Tarjetas con sombra */
    .css-1r6slb0 {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Botones mejorados */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Métricas personalizadas */
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: bold;
        color: #667eea;
    }
    
    /* Footer personalizado */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        padding: 15px;
        font-size: 14px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 999;
    }
    
    /* Tarjetas de información */
    .info-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-left: 5px solid #667eea;
    }
    
    /* Etiquetas grandes */
    .big-label {
        font-size: 36px !important;
        font-weight: 800;
        color: #667eea;
        text-align: center;
        padding: 20px;
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Progress bar personalizado */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tabs mejorados */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: white;
        border-radius: 10px;
        padding: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    /* Expander personalizado */
    .streamlit-expanderHeader {
        background-color: white;
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# DATABASE
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT UNIQUE,
            name TEXT,
            email TEXT,
            role TEXT,
            threshold REAL DEFAULT 0.5,
            notes TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            source TEXT,
            filename TEXT,
            label TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

def insert_prediction(timestamp, source, filename, label, confidence):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO predictions (timestamp, source, filename, label, confidence) VALUES (?,?,?,?,?)',
              (timestamp, source, filename, label, float(confidence)))
    conn.commit()
    conn.close()

# ---------------------------
# CARGA DE MODELO
# ---------------------------
@st.cache_resource
def load_tm_model(model_path, labels_path):
    """Carga el modelo de Teachable Machine con manejo correcto de versiones"""
    model = keras.models.load_model(
        model_path, 
        compile=False,
        custom_objects=None
    )
    
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f.readlines()]
    else:
        labels = [f"Clase_{i}" for i in range(model.output_shape[-1])]
    
    return model, labels

def preprocess_image(image: Image.Image, target_size=IMAGE_SIZE):
    """Preprocesa la imagen para Teachable Machine"""
    img = image.convert('RGB')
    img = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)
    img_array = np.asarray(img, dtype=np.float32)
    normalized = (img_array / 127.5) - 1
    return np.expand_dims(normalized, axis=0)

# ---------------------------
# FUNCIÓN DE PREDICCIÓN
# ---------------------------
def predict_and_display(image, source_name, filename):
    """Función reutilizable para hacer predicciones y mostrar resultados"""
    try:
        with st.spinner("🔄 Analizando imagen con IA..."):
            # Preprocesar imagen
            processed = preprocess_image(image, IMAGE_SIZE)
            
            # Hacer predicción
            predictions = model.predict(processed, verbose=0)[0]
            
            # Obtener resultado principal
            top_idx = int(np.argmax(predictions))
            top_label = labels[top_idx]
            top_conf = float(predictions[top_idx])
            
            # Contenedor principal con diseño mejorado
            st.markdown("<div class='info-card'>", unsafe_allow_html=True)
            
            # Mostrar resultado principal
            st.markdown("### 🎯 Identificación Detectada")
            st.markdown(f"<div class='big-label'>👤 {top_label}</div>", unsafe_allow_html=True)
            
            # Barra de confianza con color dinámico
            st.progress(top_conf)
            
            # Color según confianza
            if top_conf >= 0.8:
                conf_color = "🟢"
            elif top_conf >= 0.5:
                conf_color = "🟡"
            else:
                conf_color = "🔴"
            
            st.markdown(f"### {conf_color} Nivel de Confianza: **{top_conf:.1%}**")
            
            # Buscar información de la persona
            conn = sqlite3.connect(DB_PATH)
            df_people = pd.read_sql_query('SELECT * FROM people', conn)
            conn.close()
            
            threshold = 0.5
            person_info = df_people[df_people['label'] == top_label]
            
            if not person_info.empty:
                person = person_info.iloc[0]
                threshold = float(person['threshold'])
                
                st.markdown("---")
                st.markdown("#### 📋 Información de la Persona")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"""
                    **👤 Nombre:** {person['name']}  
                    **💼 Rol:** {person['role']}  
                    """)
                with col_b:
                    st.markdown(f"""
                    **📧 Email:** {person['email']}  
                    **📊 Umbral:** {threshold:.0%}
                    """)
                
                if top_conf >= threshold:
                    st.success(f"✅ **Persona reconocida correctamente** (Confianza: {top_conf:.1%} ≥ {threshold:.0%})")
                else:
                    st.warning(f"⚠️ **Confianza por debajo del umbral** ({top_conf:.1%} < {threshold:.0%})")
                
                if person['notes']:
                    st.info(f"📝 **Notas:** {person['notes']}")
            else:
                if top_conf >= threshold:
                    st.success(f"✅ Reconocido como: **{top_label}** (Confianza: {top_conf:.1%})")
                else:
                    st.warning(f"⚠️ Confianza baja: {top_conf:.1%} (Umbral: {threshold:.0%})")
                st.info("ℹ️ Esta persona no está registrada. Ve a **'👥 Administración'** para añadirla.")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Mostrar top 3 predicciones en expander
            with st.expander("📊 Ver Top 3 Predicciones Detalladas"):
                top3_indices = np.argsort(predictions)[-3:][::-1]
                
                for i, idx in enumerate(top3_indices, 1):
                    label_name = labels[idx]
                    conf = predictions[idx]
                    
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col1:
                        st.markdown(f"**#{i}**")
                    with col2:
                        st.markdown(f"**{label_name}**")
                    with col3:
                        st.markdown(f"**{conf:.1%}**")
                    st.progress(float(conf))
                    st.markdown("")
            
            # Botón para guardar con diseño mejorado
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("💾 Guardar Predicción en Base de Datos", 
                           type="primary", 
                           use_container_width=True,
                           key=f"save_{source_name}_{datetime.now().timestamp()}"):
                    timestamp = datetime.now().isoformat()
                    
                    insert_prediction(
                        timestamp=timestamp,
                        source=source_name,
                        filename=filename,
                        label=top_label,
                        confidence=top_conf
                    )
                    
                    st.success("✅ ¡Predicción guardada exitosamente en la base de datos!")
                    st.balloons()
                    
    except Exception as e:
        st.error(f"❌ Error al procesar la imagen: {str(e)}")
        with st.expander("🔍 Ver detalles del error"):
            st.code(str(e), language="text")

# ---------------------------
# INICIALIZACIÓN
# ---------------------------
init_db()

# Cargar modelo
model = None
labels = []

if not TF_AVAILABLE:
    st.sidebar.error("❌ TensorFlow no está instalado")
    st.sidebar.code("pip install tensorflow==2.12.0", language="bash")
elif not os.path.exists(MODEL_PATH):
    st.sidebar.warning("⚠️ No se encontró el archivo del modelo")
    st.sidebar.info(f"📁 Ruta esperada: `{MODEL_PATH}`")
else:
    try:
        with st.spinner("🔄 Cargando modelo de IA..."):
            model, labels = load_tm_model(MODEL_PATH, LABELS_PATH)
        st.sidebar.success("✅ Modelo cargado correctamente")
        st.sidebar.metric("📊 Clases Disponibles", len(labels))
        
        with st.sidebar.expander("🔍 Ver clases del modelo"):
            for i, label in enumerate(labels):
                st.write(f"**{i+1}.** {label}")
    except Exception as e:
        st.sidebar.error(f"⚠️ Error al cargar el modelo")
        with st.sidebar.expander("🔍 Ver detalles"):
            st.code(str(e), language="text")
        st.sidebar.info("💡 Solución:")
        st.sidebar.code("pip uninstall tensorflow\npip install tensorflow==2.12.0", language="bash")

# ---------------------------
# HEADER PRINCIPAL
# ---------------------------
st.markdown("<h1>🎯 Sistema de Reconocimiento Facial Inteligente</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 18px; margin-bottom: 30px;'>Tecnología de Machine Learning para identificación de personas en tiempo real</p>", unsafe_allow_html=True)

# ---------------------------
# NAVEGACIÓN
# ---------------------------
page = st.sidebar.radio(
    "📂 Menú de Navegación", 
    ["🎥 Reconocimiento en Vivo", "👥 Administración de Personas", "📊 Dashboard Analítico", "📥 Exportación de Datos"],
    label_visibility="visible"
)

st.sidebar.markdown("---")

# ---------------------------
# PÁGINA: RECONOCIMIENTO EN VIVO
# ---------------------------
if page == "🎥 Reconocimiento en Vivo":
    st.header("🎥 Captura y Reconocimiento en Tiempo Real")
    
    if model is None:
        st.error("❌ El modelo no está cargado. Revisa la barra lateral para más información.")
        st.info("📝 **Pasos para solucionar:**")
        st.markdown("""
        1. Verifica que exista la carpeta `model/` con:
           - `keras_model.h5`
           - `labels.txt`
        2. Instala TensorFlow: `pip install tensorflow==2.12.0`
        3. Reinicia la aplicación
        """)
    else:
        # Tabs para diferentes fuentes
        tab1, tab2 = st.tabs(["📸 Cámara en Vivo", "📁 Subir Imagen"])
        
        # TAB 1: CÁMARA
        with tab1:
            st.markdown("### 📸 Captura desde la Cámara")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                with st.expander("ℹ️ Instrucciones de Uso", expanded=False):
                    st.markdown("""
                    1. **Autoriza** el acceso a tu cámara
                    2. **Posiciónate** frente a la cámara
                    3. **Asegura** buena iluminación
                    4. **Haz clic** en "Take Photo"
                    5. **Revisa** el resultado a la derecha
                    """)
                
                camera_photo = st.camera_input("🎥 Activa tu cámara y captura")
                
                if camera_photo is not None:
                    image = Image.open(camera_photo)
                    st.image(image, caption="✅ Imagen capturada correctamente", use_container_width=True)
            
            with col2:
                if camera_photo is None:
                    st.info("👈 Captura una foto con la cámara para comenzar el análisis")
                    st.image("https://via.placeholder.com/400x300/667eea/ffffff?text=Esperando+Captura", use_container_width=True)
                else:
                    image = Image.open(camera_photo)
                    filename = f"cam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    predict_and_display(image, "camera_live", filename)
        
        # TAB 2: SUBIR ARCHIVO
        with tab2:
            st.markdown("### 📁 Cargar Imagen desde Archivo")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                with st.expander("ℹ️ Formatos Aceptados", expanded=False):
                    st.markdown("""
                    - ✅ **JPG / JPEG**
                    - ✅ **PNG**
                    - 📏 Tamaño recomendado: Mínimo 224x224 px
                    - 💡 Mejor iluminación = Mejor precisión
                    """)
                
                uploaded_file = st.file_uploader(
                    "Arrastra o selecciona una imagen", 
                    type=['jpg', 'jpeg', 'png'],
                    help="Sube una foto clara del rostro"
                )
                
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"✅ Archivo cargado: {uploaded_file.name}", use_container_width=True)
            
            with col2:
                if uploaded_file is None:
                    st.info("👈 Sube una imagen para iniciar el reconocimiento")
                    st.image("https://via.placeholder.com/400x300/764ba2/ffffff?text=Subir+Imagen", use_container_width=True)
                else:
                    image = Image.open(uploaded_file)
                    predict_and_display(image, "uploaded_file", uploaded_file.name)

# ---------------------------
# PÁGINA: ADMINISTRACIÓN
# ---------------------------
elif page == "👥 Administración de Personas":
    st.header("👥 Gestión de Personas Registradas")
    
    conn = sqlite3.connect(DB_PATH)
    df_people = pd.read_sql_query('SELECT * FROM people', conn)
    conn.close()

    # Métricas con diseño mejorado
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👤 Total Personas", len(df_people), delta=None)
    with col2:
        st.metric("🏷️ Etiquetas Únicas", df_people['label'].nunique() if not df_people.empty else 0)
    with col3:
        avg_th = df_people['threshold'].mean() if not df_people.empty else 0.5
        st.metric("📊 Umbral Promedio", f"{avg_th:.2f}")
    with col4:
        active = len(df_people[df_people['threshold'] >= 0.5]) if not df_people.empty else 0
        st.metric("✅ Activos", active)

    st.markdown("---")
    
    # Tabs para organizar mejor
    tab1, tab2, tab3 = st.tabs(["📋 Lista de Personas", "➕ Registrar Nueva", "🗑️ Eliminar Persona"])
    
    # TAB 1: LISTADO
    with tab1:
        st.subheader("📋 Personas Registradas en el Sistema")
        if df_people.empty:
            st.info("📭 No hay personas registradas. Ve a la pestaña **'➕ Registrar Nueva'** para añadir.")
        else:
            # Formatear dataframe
            df_display = df_people.copy()
            df_display['threshold'] = df_display['threshold'].apply(lambda x: f"{x:.0%}")
            
            st.dataframe(
                df_display,
                use_container_width=True,
                height=400,
                column_config={
                    "id": "ID",
                    "label": st.column_config.TextColumn("Etiqueta", width="medium"),
                    "name": st.column_config.TextColumn("Nombre", width="large"),
                    "email": "Email",
                    "role": "Rol",
                    "threshold": "Umbral",
                    "notes": "Notas"
                }
            )
    
    # TAB 2: CREAR/EDITAR
    with tab2:
        st.subheader("➕ Registrar Nueva Persona")
        
        with st.form("person_form", clear_on_submit=True):
            st.markdown("**📝 Información Básica:**")
            col1, col2 = st.columns(2)
            
            with col1:
                label = st.text_input(
                    "🏷️ Etiqueta del Modelo *", 
                    placeholder="Ej: 0 Joel Pesantez",
                    help="Debe coincidir EXACTAMENTE con una clase de Teachable Machine"
                )
                name = st.text_input(
                    "👤 Nombre Completo *",
                    placeholder="Ej: Joel Pesantez"
                )
                email = st.text_input(
                    "📧 Correo Electrónico",
                    placeholder="ejemplo@email.com"
                )
            
            with col2:
                role = st.text_input(
                    "💼 Rol / Cargo",
                    placeholder="Ej: Estudiante"
                )
                threshold = st.slider(
                    "📊 Umbral de Confianza", 
                    0.0, 1.0, 0.5, 0.05,
                    help="Confianza mínima requerida (50% recomendado)"
                )
                notes = st.text_area(
                    "📝 Notas Adicionales",
                    placeholder="Información relevante..."
                )
            
            st.markdown("---")
            submitted = st.form_submit_button("💾 Guardar Persona", type="primary", use_container_width=True)
            
            if submitted:
                if not label or not name:
                    st.error("❌ Los campos **Etiqueta** y **Nombre** son obligatorios")
                else:
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    try:
                        c.execute('''INSERT OR REPLACE INTO people 
                                     (label, name, email, role, threshold, notes) 
                                     VALUES (?,?,?,?,?,?)''',
                                  (label, name, email, role, threshold, notes))
                        conn.commit()
                        st.success(f"✅ **{name}** ha sido guardado correctamente en el sistema")
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error al guardar: {e}")
                    finally:
                        conn.close()
    
    # TAB 3: ELIMINAR
    with tab3:
        st.subheader("🗑️ Eliminar Persona del Sistema")
        
        if df_people.empty:
            st.info("📭 No hay personas registradas para eliminar")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                del_label = st.selectbox(
                    "Selecciona la persona a eliminar",
                    ["-- Seleccionar --"] + list(df_people['label']),
                    help="Esta acción no se puede deshacer"
                )
            
            if del_label != "-- Seleccionar --":
                person = df_people[df_people['label'] == del_label].iloc[0]
                
                st.warning(f"⚠️ **Atención:** Vas a eliminar a **{person['name']}** (Etiqueta: {del_label})")
                
                with st.expander("📋 Ver información completa"):
                    st.json({
                        "Nombre": person['name'],
                        "Email": person['email'],
                        "Rol": person['role'],
                        "Umbral": f"{person['threshold']:.0%}",
                        "Notas": person['notes']
                    })
                
                col_a, col_b, col_c = st.columns([1, 1, 1])
                with col_b:
                    if st.button("🗑️ Confirmar Eliminación", type="secondary", use_container_width=True):
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        c.execute('DELETE FROM people WHERE label=?', (del_label,))
                        conn.commit()
                        conn.close()
                        st.success(f"✅ **{person['name']}** ha sido eliminado del sistema")
                        st.rerun()

# ---------------------------
# PÁGINA: ANALÍTICA
# ---------------------------
elif page == "📊 Dashboard Analítico":
    st.header("📊 Panel de Analítica y Estadísticas")
    
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT * FROM predictions', conn, parse_dates=['timestamp'])
    conn.close()

    if df.empty:
        st.info("📭 No hay predicciones registradas. Ve a **'🎥 Reconocimiento en Vivo'** para empezar a recopilar datos.")
        st.image("https://via.placeholder.com/800x400/667eea/ffffff?text=Sin+Datos+Disponibles", use_container_width=True)
    else:
        # Métricas generales con diseño mejorado
        st.markdown("### 📈 Métricas Generales del Sistema")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🔢 Total Predicciones", len(df))
        with col2:
            st.metric("🏷️ Etiquetas Únicas", df['label'].nunique())
        with col3:
            st.metric("📊 Confianza Promedio", f"{df['confidence'].mean():.1%}")
        with col4:
            st.metric("🎯 Confianza Máxima", f"{df['confidence'].max():.1%}")
        with col5:
            st.metric("📉 Confianza Mínima", f"{df['confidence'].min():.1%}")

        st.markdown("---")
        
        # Mostrar datos completos
        with st.expander("📋 Ver Tabla Completa de Predicciones"):
            df_display = df.copy()
            df_display['confidence'] = df_display['confidence'].apply(lambda x: f"{x:.1%}")
            df_display['timestamp'] = pd.to_datetime(df_display['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(df_display, use_container_width=True, height=300)

        st.markdown("---")
        st.markdown("### 📊 Visualizaciones Estadísticas")

        # Configurar estilo de matplotlib
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # GRÁFICA 1 y 2
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 1️⃣ Distribución de Reconocimientos por Persona")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            counts = df['label'].value_counts()
            colors = plt.cm.Spectral(np.linspace(0, 1, len(counts)))
            counts.plot(kind='barh', ax=ax1, color=colors)
            ax1.set_xlabel('Cantidad de Reconocimientos', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Persona', fontsize=12, fontweight='bold')
            ax1.set_title('Frecuencia de Detecciones', fontsize=14, fontweight='bold', pad=20)
            ax1.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig1)

        with col2:
            st.markdown("#### 2️⃣ Confianza Promedio por Persona")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            avg_conf = df.groupby('label')['confidence'].mean().sort_values()
            colors2 = ['#ff6b6b' if x < 0.5 else '#ffd93d' if x < 0.8 else '#6bcf7f' for x in avg_conf]
            avg_conf.plot(kind='barh', ax=ax2, color=colors2)
            ax2.set_xlabel('Nivel de Confianza Promedio', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Persona', fontsize=12, fontweight='bold')
            ax2.set_title('Precisión de Reconocimiento', fontsize=14, fontweight='bold', pad=20)
            ax2.axvline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Umbral Mínimo (50%)')
            ax2.axvline(0.8, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Alta Confianza (80%)')
            ax2.legend(loc='lower right')
            ax2.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)

        # GRÁFICA 3 y 4
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("#### 3️⃣ Histograma de Distribución de Confianza")
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            n, bins, patches = ax3.hist(df['confidence'], bins=25, edgecolor='black', alpha=0.8)
            
            # Colorear barras según el rango
            for i, patch in enumerate(patches):
                if bins[i] < 0.5:
                    patch.set_facecolor('#ff6b6b')
                elif bins[i] < 0.8:
                    patch.set_facecolor('#ffd93d')
                else:
                    patch.set_facecolor('#6bcf7f')
            
            ax3.set_xlabel('Nivel de Confianza', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
            ax3.set_title('Distribución de Precisión del Modelo', fontsize=14, fontweight='bold', pad=20)
            ax3.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Umbral 50%')
            ax3.axvline(df['confidence'].mean(), color='blue', linestyle='-', linewidth=2, label=f'Promedio ({df["confidence"].mean():.1%})')
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig3)

        with col4:
            st.markdown("#### 4️⃣ Timeline de Reconocimientos")
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            series = df.groupby('date').size()
            
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            series.plot(ax=ax4, marker='o', color='#667eea', linewidth=3, markersize=8)
            ax4.fill_between(series.index, series.values, alpha=0.3, color='#667eea')
            ax4.set_ylabel('Número de Reconocimientos', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Fecha', fontsize=12, fontweight='bold')
            ax4.set_title('Evolución Temporal de Detecciones', fontsize=14, fontweight='bold', pad=20)
            ax4.grid(True, alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig4)

        # GRÁFICA 5
        st.markdown("#### 5️⃣ Patrón de Reconocimientos por Hora del Día")
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        hour_counts = df.groupby('hour').size().reindex(range(24), fill_value=0)
        
        fig5, ax5 = plt.subplots(figsize=(14, 5))
        colors5 = plt.cm.viridis(hour_counts / hour_counts.max())
        bars = ax5.bar(hour_counts.index, hour_counts.values, color=colors5, edgecolor='black', linewidth=1.5)
        
        # Agregar valores en las barras
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax5.set_ylabel('Cantidad de Reconocimientos', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Hora del Día (24h)', fontsize=12, fontweight='bold')
        ax5.set_title('Actividad por Hora - Identificación de Patrones Temporales', fontsize=14, fontweight='bold', pad=20)
        ax5.set_xticks(range(24))
        ax5.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right')
        ax5.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig5)

        # Análisis adicional
        st.markdown("---")
        st.markdown("### 📈 Análisis Detallado")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown("#### 🏆 Persona Más Detectada")
            top_person = df['label'].value_counts().idxmax()
            top_count = df['label'].value_counts().max()
            st.success(f"**{top_person}**")
            st.metric("Detecciones", top_count)
        
        with col_b:
            st.markdown("#### ⭐ Mayor Confianza Promedio")
            best_conf = df.groupby('label')['confidence'].mean().idxmax()
            best_conf_val = df.groupby('label')['confidence'].mean().max()
            st.success(f"**{best_conf}**")
            st.metric("Confianza", f"{best_conf_val:.1%}")
        
        with col_c:
            st.markdown("#### 🕐 Hora Más Activa")
            peak_hour = hour_counts.idxmax()
            peak_count = hour_counts.max()
            st.success(f"**{peak_hour:02d}:00**")
            st.metric("Reconocimientos", peak_count)

        # Exportar gráficas
        st.markdown("---")
        st.markdown("### 📦 Exportación de Visualizaciones")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📦 Descargar Todas las Gráficas (ZIP)", type="primary", use_container_width=True):
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, 'w') as zf:
                    for i, fig in enumerate([fig1, fig2, fig3, fig4, fig5], start=1):
                        img_bytes = io.BytesIO()
                        fig.savefig(img_bytes, format='png', dpi=300, bbox_inches='tight')
                        img_bytes.seek(0)
                        zf.writestr(f'grafica_{i}_reconocimiento.png', img_bytes.read())
                
                zip_buf.seek(0)
                st.download_button(
                    label="⬇️ Descargar ZIP con Gráficas",
                    data=zip_buf,
                    file_name=f"analisis_reconocimiento_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )

# ---------------------------
# PÁGINA: EXPORTAR
# ---------------------------
elif page == "📥 Exportación de Datos":
    st.header("📥 Exportación y Generación de Reportes")
    
    conn = sqlite3.connect(DB_PATH)
    df_pred = pd.read_sql_query('SELECT * FROM predictions', conn)
    df_people = pd.read_sql_query('SELECT * FROM people', conn)
    conn.close()

    st.markdown("### 📊 Datos Disponibles para Exportar")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Base de Predicciones")
        if df_pred.empty:
            st.info("📭 Sin datos de predicciones")
        else:
            st.metric("Total de Registros", len(df_pred))
            st.metric("Días con Actividad", f"{df_pred['timestamp'].nunique()} días")
            
            # Vista previa
            with st.expander("👁️ Vista Previa de Datos"):
                st.dataframe(df_pred.head(10), use_container_width=True)
            
            # Formatear datos
            df_export = df_pred.copy()
            df_export['confidence'] = df_export['confidence'].apply(lambda x: f"{x:.2%}")
            df_export['timestamp'] = pd.to_datetime(df_export['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Descargar Predicciones (CSV)",
                data=csv,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
    
    with col2:
        st.markdown("#### 👥 Base de Personas")
        if df_people.empty:
            st.info("📭 Sin datos de personas")
        else:
            st.metric("Total de Personas", len(df_people))
            st.metric("Etiquetas Registradas", df_people['label'].nunique())
            
            # Vista previa
            with st.expander("👁️ Vista Previa de Datos"):
                st.dataframe(df_people, use_container_width=True)
            
            # Formatear datos
            df_people_export = df_people.copy()
            df_people_export['threshold'] = df_people_export['threshold'].apply(lambda x: f"{x:.0%}")
            
            csv_p = df_people_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Descargar Personas (CSV)",
                data=csv_p,
                file_name=f"people_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )

    st.markdown("---")

# ---------------------------
# FOOTER PERSONALIZADO
# ---------------------------
st.markdown("---")
st.markdown("""
<div class='footer'>
    <strong>🎓 Sistema de Reconocimiento Facial Inteligente</strong><br>
    Elaborado por: <strong>Joel Pesantez</strong> y <strong>Maura Calle</strong> | 
    Powered by Teachable Machine & Streamlit | 
    © 2024
</div>
""", unsafe_allow_html=True)

# Espacio para el footer
st.markdown("<br><br>", unsafe_allow_html=True)