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
MODEL_PATH = os.path.join("model", "keras_model.h5")
LABELS_PATH = os.path.join("model", "labels.txt")
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
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    h1 {
        color: #1e3c72;
        font-weight: 800;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    h2, h3 {
        color: #2c3e50;
        font-weight: 700;
    }
    
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
    
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: bold;
        color: #667eea;
    }
    
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
</style>
""", unsafe_allow_html=True)

# ---------------------------
# DATABASE
# ---------------------------
def init_db():
    """Inicializa la base de datos con las tablas necesarias"""
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
    """Inserta una predicción en la base de datos"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('INSERT INTO predictions (timestamp, source, filename, label, confidence) VALUES (?,?,?,?,?)',
                  (timestamp, source, filename, label, float(confidence)))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error al guardar en DB: {e}")
        return False

# ---------------------------
# CARGA DE MODELO
# ---------------------------
@st.cache_resource
def load_tm_model(model_path, labels_path):
    """Carga el modelo de Teachable Machine"""
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
    """Realiza la predicción y muestra los resultados"""
    try:
        with st.spinner("🔄 Analizando imagen con IA..."):
            processed = preprocess_image(image, IMAGE_SIZE)
            predictions = model.predict(processed, verbose=0)[0]
            top_idx = int(np.argmax(predictions))
            top_label = labels[top_idx]
            top_conf = float(predictions[top_idx])

            # Mostrar resultado principal
            st.markdown(f"### 👤 {top_label}")
            st.metric("Confianza", f"{top_conf:.1%}")

            # Obtener umbral de la persona
            try:
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute('SELECT threshold, name FROM people WHERE label = ?', (top_label,))
                row = c.fetchone()
                conn.close()
                
                threshold = float(row[0]) if row else 0.5
                person_name = row[1] if row and row[1] else top_label
                
                # Indicador visual de umbral
                if top_conf >= threshold:
                    st.success(f"✅ Identificado: **{person_name}** (confianza {top_conf:.1%} ≥ umbral {threshold:.0%})")
                else:
                    st.warning(f"⚠️ Confianza por debajo del umbral ({top_conf:.1%} < {threshold:.0%})")
                    
                    # Sugerencia si el umbral es muy alto
                    if threshold >= 0.9:
                        st.error(f"""
                        🚨 **Umbral demasiado alto ({threshold:.0%})**
                        
                        Este umbral es prácticamente imposible de alcanzar. Se recomienda:
                        - Ir a **'👥 Administración'** → **'✏️ Editar Persona'**
                        - Ajustar el umbral de **{person_name}** a **65%**
                        """)
            except:
                threshold = 0.5

            # Mostrar top 3 predicciones
            st.markdown("#### Top 3 predicciones")
            top3 = np.argsort(predictions)[-3:][::-1]
            for i, idx in enumerate(top3, 1):
                conf_value = float(predictions[idx])
                conf_pct = conf_value * 100
                st.progress(conf_value, text=f"{i}. {labels[idx]} - {conf_pct:.1f}%")

            # GUARDADO AUTOMÁTICO si pasa el umbral
            st.markdown("---")
            
            # Checkbox para habilitar guardado automático
            auto_save = st.checkbox("💾 Guardar automáticamente en BD", value=True, key=f"auto_{filename}")
            
            if auto_save and top_conf >= threshold:
                # Verificar si ya se guardó esta predicción
                save_key = f"{source_name}_{filename}_{top_label}_{top_conf:.4f}"
                
                if f'saved_{save_key}' not in st.session_state:
                    success = insert_prediction(
                        timestamp=datetime.now().isoformat(),
                        source=source_name,
                        filename=filename,
                        label=top_label,
                        confidence=top_conf
                    )
                    
                    if success:
                        st.session_state[f'saved_{save_key}'] = True
                        st.success("✅ Predicción guardada automáticamente en BD")
                    else:
                        st.error("❌ Error al guardar")
                else:
                    st.info("ℹ️ Esta predicción ya fue guardada")
            elif auto_save:
                st.info(f"ℹ️ No se guarda: confianza {top_conf:.1%} < umbral {threshold:.0%}")

    except Exception as e:
        st.error(f"❌ Error en predicción: {str(e)}")
        import traceback
        with st.expander("🔍 Ver detalles del error"):
            st.code(traceback.format_exc())

# ---------------------------
# INICIALIZACIÓN
# ---------------------------
# Inicializar session_state para mensajes
if 'last_save' not in st.session_state:
    st.session_state.last_save = None

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
        st.sidebar.error(f"⚠️ Error al cargar el modelo: {str(e)}")

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
        st.error("❌ El modelo no está cargado. Revisa la barra lateral.")
    else:
        tab1, tab2 = st.tabs(["📸 Cámara en Vivo", "📁 Subir Imagen"])
        
        with tab1:
            st.markdown("### 📸 Captura desde la Cámara")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                camera_photo = st.camera_input("🎥 Activa tu cámara y captura")
                if camera_photo is not None:
                    image = Image.open(camera_photo)
                    st.image(image, caption="✅ Imagen capturada", use_container_width=True)
            
            with col2:
                if camera_photo is None:
                    st.info("👈 Captura una foto para comenzar")
                else:
                    image = Image.open(camera_photo)
                    filename = f"cam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    predict_and_display(image, "camera_live", filename)
        
        with tab2:
            st.markdown("### 📁 Cargar Imagen desde Archivo")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                uploaded_file = st.file_uploader(
                    "Selecciona una imagen", 
                    type=['jpg', 'jpeg', 'png']
                )
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"✅ {uploaded_file.name}", use_container_width=True)
            
            with col2:
                if uploaded_file is None:
                    st.info("👈 Sube una imagen")
                else:
                    image = Image.open(uploaded_file)
                    predict_and_display(image, "uploaded_file", uploaded_file.name)

# ---------------------------
# PÁGINA: ADMINISTRACIÓN
# ---------------------------
elif page == "👥 Administración de Personas":
    st.header("👥 Gestión de Personas Registradas")
    
    # Leer datos con manejo de errores
    try:
        conn = sqlite3.connect(DB_PATH)
        df_people = pd.read_sql_query('SELECT * FROM people', conn)
        conn.close()
    except Exception as e:
        st.error(f"Error al leer base de datos: {e}")
        df_people = pd.DataFrame()

    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👤 Total Personas", len(df_people))
    with col2:
        st.metric("🏷️ Etiquetas Únicas", df_people['label'].nunique() if not df_people.empty else 0)
    with col3:
        avg_th = df_people['threshold'].mean() if not df_people.empty else 0.5
        st.metric("📊 Umbral Promedio", f"{avg_th:.2f}")
    with col4:
        active = len(df_people[df_people['threshold'] >= 0.5]) if not df_people.empty else 0
        st.metric("✅ Activos", active)

    st.markdown("---")
    
    # AGREGAR TAB DE EDICIÓN
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Lista de Personas", "➕ Registrar Nueva", "✏️ Editar Persona", "🗑️ Eliminar Persona"])
    
    with tab1:
        st.subheader("📋 Personas Registradas")
        if df_people.empty:
            st.info("📭 No hay personas registradas")
        else:
            # Identificar umbrales problemáticos
            high_threshold = df_people[df_people['threshold'] >= 0.9]
            if not high_threshold.empty:
                st.warning(f"⚠️ {len(high_threshold)} persona(s) tienen umbral ≥90%. Esto impedirá el guardado automático.")
                
                with st.expander("🔧 Ajustar Umbrales Problemáticos"):
                    st.write("**Personas con umbrales muy altos:**")
                    for idx, row in high_threshold.iterrows():
                        col_a, col_b, col_c = st.columns([2, 1, 1])
                        with col_a:
                            st.write(f"**{row['name']}** ({row['label']})")
                        with col_b:
                            st.metric("Umbral", f"{row['threshold']:.0%}")
                        with col_c:
                            if st.button(f"Ajustar a 65%", key=f"fix_{row['label']}"):
                                try:
                                    conn = sqlite3.connect(DB_PATH)
                                    c = conn.cursor()
                                    c.execute('UPDATE people SET threshold=? WHERE label=?', (0.65, row['label']))
                                    conn.commit()
                                    conn.close()
                                    st.success(f"✅ Ajustado a 65%")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
            
            # Mostrar explicación de la Etiqueta del Modelo
           
    
    with tab2:
        st.subheader("➕ Registrar Nueva Persona")
        
        
        
        # Mostrar clases disponibles en el modelo
        if model is not None and labels:
            with st.expander("🔍 Ver clases disponibles en el modelo"):
                st.write("**Etiquetas que el modelo puede reconocer:**")
                for idx, label in enumerate(labels):
                    is_registered = label in df_people['label'].values if not df_people.empty else False
                    status = "✅ Ya registrada" if is_registered else "⚠️ No registrada"
                    st.write(f"{idx + 1}. `{label}` - {status}")
        
        with st.form("person_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                label = st.text_input("🏷️ Etiqueta del Modelo *", placeholder="Ej: 0 Joel Pesantez",
                                     help="Debe coincidir EXACTAMENTE con el nombre de la clase en Teachable Machine")
                name = st.text_input("👤 Nombre Completo *", placeholder="Ej: Joel Pesantez")
                email = st.text_input("📧 Email", placeholder="ejemplo@email.com")
            
            with col2:
                role = st.text_input("💼 Rol", placeholder="Ej: Estudiante")
                threshold = st.slider("📊 Umbral de Confianza", 0.0, 1.0, 0.65, 0.05, 
                                     help="⚠️ Valores muy altos (>90%) impedirán el guardado automático")
                notes = st.text_area("📝 Notas", placeholder="Información relevante...")
            
            # Advertencia visual según el umbral
            if threshold >= 0.9:
                st.warning(f"⚠️ Umbral muy alto ({threshold:.0%}). Es posible que nunca se guarden predicciones.")
            elif threshold >= 0.8:
                st.info(f"ℹ️ Umbral alto ({threshold:.0%}). Solo se guardarán detecciones muy precisas.")
            else:
                st.success(f"✅ Umbral óptimo ({threshold:.0%}).")
            
            submitted = st.form_submit_button("💾 Guardar Persona", type="primary", use_container_width=True)
            
            if submitted:
                if not label or not name:
                    st.error("⚠️ La etiqueta y el nombre son obligatorios")
                else:
                    try:
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        c.execute('''INSERT OR REPLACE INTO people 
                                    (label, name, email, role, threshold, notes) 
                                    VALUES (?,?,?,?,?,?)''',
                                (label, name, email, role, threshold, notes))
                        conn.commit()
                        conn.close()
                        st.success(f"✅ {name} guardado correctamente con umbral {threshold:.0%}")
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
    
    # TAB 3: EDITAR PERSONA
    with tab3:
        st.subheader("✏️ Editar Persona")
        
        if df_people.empty:
            st.info("📭 No hay personas para editar")
        else:
            edit_label = st.selectbox(
                "Selecciona la persona a editar",
                ["-- Seleccionar --"] + list(df_people['label']),
                key="edit_select"
            )
            
            if edit_label != "-- Seleccionar --":
                # Obtener datos actuales
                person = df_people[df_people['label'] == edit_label].iloc[0]
                
                st.markdown("---")
                st.markdown(f"### Editando: **{person['name']}**")
                st.info(f"**Etiqueta actual:** `{person['label']}`")
                
                with st.form("edit_person_form", clear_on_submit=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        new_label = st.text_input("🏷️ Etiqueta del Modelo *", value=person['label'],
                                                 help="⚠️ Solo cambiar si también cambias el nombre en Teachable Machine")
                        new_name = st.text_input("👤 Nombre Completo *", value=person['name'] if pd.notna(person['name']) else "")
                        new_email = st.text_input("📧 Email", value=person['email'] if pd.notna(person['email']) else "")
                    
                    with col2:
                        new_role = st.text_input("💼 Rol", value=person['role'] if pd.notna(person['role']) else "")
                        new_threshold = st.slider("📊 Umbral de Confianza", 0.0, 1.0, 
                                                 float(person['threshold']) if pd.notna(person['threshold']) else 0.65, 
                                                 0.05,
                                                 help="⚠️ Valores muy altos (>90%) impedirán el guardado automático")
                        new_notes = st.text_area("📝 Notas", value=person['notes'] if pd.notna(person['notes']) else "")
                    
                    # Advertencia visual según el umbral
                    if new_threshold >= 0.9:
                        st.warning(f"⚠️ Umbral muy alto ({new_threshold:.0%}). Es posible que nunca se guarden predicciones.")
                    elif new_threshold >= 0.8:
                        st.info(f"ℹ️ Umbral alto ({new_threshold:.0%}). Solo se guardarán detecciones muy precisas.")
                    else:
                        st.success(f"✅ Umbral óptimo ({new_threshold:.0%}).")
                    
                    # Advertencia si cambió la etiqueta
                    if new_label != edit_label:
                        st.warning("""
                        ⚠️ **Estás cambiando la etiqueta del modelo**
                        
                        **Consecuencias:**
                        - Las nuevas predicciones usarán la nueva etiqueta
                        - Las predicciones antiguas mantendrán la etiqueta vieja
                        - Asegúrate de que la nueva etiqueta coincida con Teachable Machine
                        """)
                    
                    col_btn1, col_btn2 = st.columns(2)
                    
                    with col_btn1:
                        update_btn = st.form_submit_button("💾 Guardar Cambios", type="primary", use_container_width=True)
                    
                    with col_btn2:
                        cancel_btn = st.form_submit_button("❌ Cancelar", use_container_width=True)
                    
                    if update_btn:
                        if not new_label or not new_name:
                            st.error("⚠️ La etiqueta y el nombre son obligatorios")
                        else:
                            try:
                                conn = sqlite3.connect(DB_PATH)
                                c = conn.cursor()
                                
                                # Si cambió la etiqueta, verificar que no exista otra persona con esa etiqueta
                                if new_label != edit_label:
                                    c.execute('SELECT COUNT(*) FROM people WHERE label=?', (new_label,))
                                    if c.fetchone()[0] > 0:
                                        st.error(f"❌ Ya existe una persona con la etiqueta '{new_label}'")
                                        conn.close()
                                    else:
                                        # Actualizar incluyendo la etiqueta
                                        c.execute('''UPDATE people 
                                                    SET label=?, name=?, email=?, role=?, threshold=?, notes=?
                                                    WHERE label=?''',
                                                (new_label, new_name, new_email, new_role, new_threshold, new_notes, edit_label))
                                        conn.commit()
                                        conn.close()
                                        st.success(f"✅ {new_name} actualizado correctamente")
                                        st.balloons()
                                        st.rerun()
                                else:
                                    # Actualizar sin cambiar la etiqueta
                                    c.execute('''UPDATE people 
                                                SET name=?, email=?, role=?, threshold=?, notes=?
                                                WHERE label=?''',
                                            (new_name, new_email, new_role, new_threshold, new_notes, edit_label))
                                    conn.commit()
                                    conn.close()
                                    st.success(f"✅ {new_name} actualizado correctamente")
                                    st.balloons()
                                    st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error al actualizar: {e}")
                    
                    if cancel_btn:
                        st.rerun()
    
    # TAB 4: ELIMINAR
    with tab4:
        st.subheader("🗑️ Eliminar Persona")
        
        if df_people.empty:
            st.info("📭 No hay personas para eliminar")
        else:
            del_label = st.selectbox(
                "Selecciona la persona",
                ["-- Seleccionar --"] + list(df_people['label']),
                key="delete_select"
            )
            
            if del_label != "-- Seleccionar --":
                person = df_people[df_people['label'] == del_label].iloc[0]
                
                st.markdown("---")
                
                # Mostrar información de la persona
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.write(f"**Nombre:** {person['name']}")
                    st.write(f"**Etiqueta:** `{person['label']}`")
                    st.write(f"**Email:** {person['email'] if pd.notna(person['email']) else 'N/A'}")
                with col_info2:
                    st.write(f"**Rol:** {person['role'] if pd.notna(person['role']) else 'N/A'}")
                    st.write(f"**Umbral:** {person['threshold']:.0%}")
                
                st.warning(f"⚠️ **¿Estás seguro de eliminar a {person['name']}?**")
                st.error("Esta acción no se puede deshacer. Solo se eliminará de la base de datos, no del modelo.")
                
                col_del1, col_del2 = st.columns(2)
                with col_del1:
                    if st.button("🗑️ Confirmar Eliminación", type="secondary", use_container_width=True):
                        try:
                            conn = sqlite3.connect(DB_PATH)
                            c = conn.cursor()
                            c.execute('DELETE FROM people WHERE label=?', (del_label,))
                            conn.commit()
                            conn.close()
                            st.success(f"✅ {person['name']} eliminado correctamente")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                
                with col_del2:
                    if st.button("❌ Cancelar", use_container_width=True):
                        st.rerun()

# ---------------------------
# PÁGINA: ANALÍTICA
# ---------------------------
elif page == "📊 Dashboard Analítico":
    st.header("📊 Panel de Analítica y Estadísticas")
    
    # Leer datos
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query('SELECT * FROM predictions', conn)
        conn.close()
        
        st.info(f"📊 Registros cargados: **{len(df)}**")
        
    except Exception as e:
        st.error(f"❌ Error al leer BD: {e}")
        df = pd.DataFrame()

    if df.empty:
        st.info("📭 No hay predicciones registradas")
    else:
        # Conversión segura de datos
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
        
        # Eliminar filas con datos inválidos
        df = df.dropna(subset=['timestamp', 'confidence'])
        
        if df.empty:
            st.warning("⚠️ Los datos contienen errores de formato")
        else:
            df['date'] = df['timestamp'].dt.date
            df['hour'] = df['timestamp'].dt.hour

            # Métricas principales
            st.markdown("### 📈 Métricas Generales")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("🔢 Total", len(df))
            with col2:
                st.metric("🏷️ Personas", df['label'].nunique())
            with col3:
                st.metric("📊 Promedio", f"{df['confidence'].mean():.1%}")
            with col4:
                st.metric("🎯 Máxima", f"{df['confidence'].max():.1%}")
            with col5:
                st.metric("📉 Mínima", f"{df['confidence'].min():.1%}")

            st.markdown("---")
            
            # Gráficas
            st.markdown("### 📊 Visualizaciones Estadísticas")
            plt.style.use('seaborn-v0_8-darkgrid')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 1️⃣ Distribución por Persona")
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
                plt.close()

            with col2:
                st.markdown("#### 2️⃣ Confianza Promedio")
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
                plt.close()
            
            # Segunda fila de gráficas
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("#### 3️⃣ Distribución de Confianza")
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                n, bins, patches = ax3.hist(df['confidence'], bins=25, edgecolor='black', alpha=0.8)
                for i, patch in enumerate(patches):
                    if bins[i] < 0.5:
                        patch.set_facecolor('#ff6b6b')
                    elif bins[i] < 0.8:
                        patch.set_facecolor('#ffd93d')
                    else:
                        patch.set_facecolor('#6bcf7f')
                ax3.set_xlabel('Nivel de Confianza', fontsize=12, fontweight='bold')
                ax3.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
                ax3.set_title('Histograma de Precisión', fontsize=14, fontweight='bold', pad=20)
                ax3.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Umbral 50%')
                ax3.axvline(df['confidence'].mean(), color='blue', linestyle='-', linewidth=2, 
                           label=f'Promedio ({df["confidence"].mean():.1%})')
                ax3.legend()
                ax3.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close()

            with col4:
                st.markdown("#### 4️⃣ Timeline de Detecciones")
                series = df.groupby('date').size()
                fig4, ax4 = plt.subplots(figsize=(10, 6))
                series.plot(ax=ax4, marker='o', color='#667eea', linewidth=3, markersize=8)
                ax4.fill_between(series.index, series.values, alpha=0.3, color='#667eea')
                ax4.set_ylabel('Número de Reconocimientos', fontsize=12, fontweight='bold')
                ax4.set_xlabel('Fecha', fontsize=12, fontweight='bold')
                ax4.set_title('Evolución Temporal', fontsize=14, fontweight='bold', pad=20)
                ax4.grid(True, alpha=0.3)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig4)
                plt.close()
            
            # Gráfica 5 - Full width
            st.markdown("#### 5️⃣ Patrón por Hora del Día")
            hour_counts = df.groupby('hour').size().reindex(range(24), fill_value=0)
            fig5, ax5 = plt.subplots(figsize=(14, 5))
            colors5 = plt.cm.viridis(hour_counts / hour_counts.max() if hour_counts.max() > 0 else hour_counts)
            bars = ax5.bar(hour_counts.index, hour_counts.values, color=colors5, edgecolor='black', linewidth=1.5)
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax5.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontweight='bold', fontsize=10)
            ax5.set_ylabel('Cantidad de Reconocimientos', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Hora del Día (24h)', fontsize=12, fontweight='bold')
            ax5.set_title('Actividad por Hora - Patrones Temporales', fontsize=14, fontweight='bold', pad=20)
            ax5.set_xticks(range(24))
            ax5.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right')
            ax5.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig5)
            plt.close()
            
            # Análisis detallado
            st.markdown("---")
            st.markdown("### 📈 Análisis Detallado")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown("#### 🏆 Más Detectada")
                top_person = df['label'].value_counts().idxmax()
                top_count = df['label'].value_counts().max()
                st.success(f"**{top_person}**")
                st.metric("Detecciones", top_count)
            
            with col_b:
                st.markdown("#### ⭐ Mayor Confianza")
                best_conf = df.groupby('label')['confidence'].mean().idxmax()
                best_conf_val = df.groupby('label')['confidence'].mean().max()
                st.success(f"**{best_conf}**")
                st.metric("Confianza", f"{best_conf_val:.1%}")
            
            with col_c:
                st.markdown("#### 🕐 Hora Pico")
                peak_hour = hour_counts.idxmax()
                peak_count = hour_counts.max()
                st.success(f"**{peak_hour:02d}:00**")
                st.metric("Reconocimientos", peak_count)
            
            # Exportar gráficas
            st.markdown("---")
            st.markdown("### 📦 Exportar Visualizaciones")
            
            if st.button("📦 Generar ZIP con Gráficas", type="primary", use_container_width=True):
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, 'w') as zf:
                    for i, fig in enumerate([fig1, fig2, fig3, fig4, fig5], start=1):
                        img_bytes = io.BytesIO()
                        fig.savefig(img_bytes, format='png', dpi=300, bbox_inches='tight')
                        img_bytes.seek(0)
                        zf.writestr(f'grafica_{i}_reconocimiento.png', img_bytes.read())
                
                zip_buf.seek(0)
                st.download_button(
                    label="⬇️ Descargar ZIP",
                    data=zip_buf,
                    file_name=f"graficas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            
            # Tabla de datos
            with st.expander("📋 Ver Tabla Completa"):
                df_display = df.copy()
                df_display['confidence'] = df_display['confidence'].apply(lambda x: f"{x:.1%}")
                df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                st.dataframe(df_display, use_container_width=True)

# ---------------------------
# PÁGINA: EXPORTAR
# ---------------------------
elif page == "📥 Exportación de Datos":
    st.header("📥 Exportación de Datos")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        df_pred = pd.read_sql_query('SELECT * FROM predictions', conn)
        df_people = pd.read_sql_query('SELECT * FROM people', conn)
        conn.close()
    except Exception as e:
        st.error(f"Error: {e}")
        df_pred = pd.DataFrame()
        df_people = pd.DataFrame()

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Predicciones")
        if not df_pred.empty:
            st.metric("Total", len(df_pred))
            csv = df_pred.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Descargar CSV",
                data=csv,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("Sin datos")
    
    with col2:
        st.markdown("#### 👥 Personas")
        if not df_people.empty:
            st.metric("Total", len(df_people))
            csv_p = df_people.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Descargar CSV",
                data=csv_p,
                file_name=f"people_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("Sin datos")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.markdown("""
<div class='footer'>
    <strong>🎓 Sistema de Reconocimiento Facial</strong><br>
    Joel Pesantez & Maura Calle | Powered by Teachable Machine & Streamlit | © 2024
</div>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
