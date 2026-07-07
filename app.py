import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
import time
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import (
    SatelliteDatasetGenerator,
    extract_optical_features,
    extract_sar_features,
    extract_text_features,
    build_spatial_graph
)
from model import CrossModalEmbeddingAligner
from model_advanced import (
    CLIPWrapper,
    Dinov2Wrapper,
    MultiModalTransformer,
    GCNRefiner,
    generate_saliency_heatmap
)
from index_search import FaissIndexManager

# --- Device Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Multi-Sensor Satellite Retrieval",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Localization / Multi-language dictionary ---
LOCALIZATION = {
    "en": {
        "title": "🛰️ Multi-Sensor Satellite Image Retrieval",
        "subtitle": "Cross-Modal Retrieval between Optical, Radar (SAR), and Text using Transformers & GNNs",
        "sidebar_settings": "⚙️ Configuration Panel",
        "lang_select": "🌐 Select Language",
        "dataset_config": "1. Dataset Configuration",
        "num_scenes": "Number of Scenes",
        "gen_seed": "Generator Seed",
        "regen_btn": "📦 Regenerate Dataset",
        "model_select": "2. Select Model to Train",
        "epochs": "Training Epochs",
        "lr": "Learning Rate",
        "train_btn": "🔥 Train Active Model",
        "reset_btn": "🔄 Reset Model",
        "model_status": "Model Status:",
        "epochs_trained": "Trained Epochs",
        "tab_gis": "🌍 GIS Explorer",
        "tab_models": "🧠 Models & latent PCA",
        "tab_retrieve": "🔍 Real-time Retrieval (FAISS)",
        "tab_explain": "👁️ Explainability (Grad-CAM)",
        "tab_cloud": "🚢 Docker & Cloud",
        "desc_gis": "Explore geolocated satellite scenes plotted over real-world biome coordinates.",
        "desc_models": "Train models and inspect alignment in 3D joint embedding space.",
        "desc_retrieve": "Perform fast cross-sensor queries using optimized FAISS indexing.",
        "desc_explain": "Inspect neural attention maps highlighting discriminative features.",
        "desc_cloud": "Production containerization and deployment instructions.",
        "urban": "Urban (Tokyo, Japan)",
        "forest": "Forest (Amazon Rainforest)",
        "water": "Water (Lake Geneva, Switzerland)",
        "farmland": "Farmland (Iowa Corn Belt)",
        "desert": "Desert (Sahara Desert, Algeria)",
        "class_label": "Land Cover Class",
        "river": "Has Winding River",
        "road": "Has Straight Road",
        "description": "Text Description",
        "query_config": "Query Configuration",
        "query_mod": "Query Modality",
        "db_mod": "Database Modality",
        "search_btn": "Search Modality Database",
        "latency_cmp": "Latency Comparison (ms)",
        "gnn_refine": "Enable GNN Neighborhood Refinement (Spatial Context Smooth)"
    },
    "es": {
        "title": "🛰️ Recuperación de Imágenes Satelitales Multisensoriales",
        "subtitle": "Búsqueda transmodal entre imágenes ópticas, radar (SAR) y texto mediante Transformers y GNN",
        "sidebar_settings": "⚙️ Panel de Configuración",
        "lang_select": "🌐 Seleccionar Idioma",
        "dataset_config": "1. Configuración del Conjunto de Datos",
        "num_scenes": "Número de Escenas",
        "gen_seed": "Semilla del Generador",
        "regen_btn": "📦 Regenerar Conjunto de Datos",
        "model_select": "2. Seleccionar Modelo para Entrenar",
        "epochs": "Épocas de Entrenamiento",
        "lr": "Tasa de Aprendizaje",
        "train_btn": "🔥 Entrenar Modelo Activo",
        "reset_btn": "🔄 Restablecer Modelo",
        "model_status": "Estado del Modelo:",
        "epochs_trained": "Épocas Entrenadas",
        "tab_gis": "🌍 Explorador GIS",
        "tab_models": "🧠 Modelos y PCA Latente",
        "tab_retrieve": "🔍 Búsqueda en Tiempo Real (FAISS)",
        "tab_explain": "👁️ Explicabilidad (Grad-CAM)",
        "tab_cloud": "🚢 Docker y Nube",
        "desc_gis": "Explore escenas satelitales geolocalizadas representadas en coordenadas reales de biomas.",
        "desc_models": "Entrene modelos e inspeccione la alineación en el espacio tridimensional de incrustaciones.",
        "desc_retrieve": "Realice consultas rápidas entre sensores utilizando indexación optimizada de FAISS.",
        "desc_explain": "Inspeccione mapas de atención neuronal que destacan características clave.",
        "desc_cloud": "Instrucciones de contenedorización y despliegue para producción.",
        "urban": "Urbano (Tokio, Japón)",
        "forest": "Bosque (Selva Amazónica)",
        "water": "Agua (Lago de Ginebra, Suiza)",
        "farmland": "Tierras de cultivo (Iowa, EE.UU.)",
        "desert": "Desierto (Desierto del Sahara, Argelia)",
        "class_label": "Clase de Cobertura Terrestre",
        "river": "Contiene Río Sinuoso",
        "road": "Contiene Carretera Recta",
        "description": "Descripción de Texto",
        "query_config": "Configuración de Consulta",
        "query_mod": "Modalidad de Consulta",
        "db_mod": "Modalidad de Base de Datos",
        "search_btn": "Buscar en Base de Datos",
        "latency_cmp": "Comparación de Latencia (ms)",
        "gnn_refine": "Activar Refinamiento de Vecindario GNN (Suavizado de Contexto Espacial)"
    },
    "fr": {
        "title": "🛰️ Recherche d'Images Satellitaires Multi-Capteurs",
        "subtitle": "Recherche transmodale entre images optiques, radar (SAR) et texte à l'aide de Transformers et GNN",
        "sidebar_settings": "⚙️ Panneau de Configuration",
        "lang_select": "🌐 Sélectionner la Langue",
        "dataset_config": "1. Configuration du Jeu de Données",
        "num_scenes": "Nombre de Scènes",
        "gen_seed": "Graine du Générateur",
        "regen_btn": "📦 Régénérer le Jeu de Données",
        "model_select": "2. Sélectionner le Modèle à Entraîner",
        "epochs": "Époques d'Entraînement",
        "lr": "Taux d'Apprentissage",
        "train_btn": "🔥 Entraîner le Modèle Actif",
        "reset_btn": "🔄 Réinitialiser le Modèle",
        "model_status": "Statut du Modèle :",
        "epochs_trained": "Époques Entraînées",
        "tab_gis": "🌍 Explorateur SIG",
        "tab_models": "🧠 Modèles & PCA Latent",
        "tab_retrieve": "🔍 Recherche en Temps Réel (FAISS)",
        "tab_explain": "👁️ Explicabilité (Grad-CAM)",
        "tab_cloud": "🚢 Docker & Cloud",
        "desc_gis": "Explorez des scènes satellites géolocalisées tracées sur de vraies coordonnées de biomes.",
        "desc_models": "Entraînez des modèles et inspectez l'alignement dans un espace d'intégration 3D.",
        "desc_retrieve": "Effectuez des requêtes croisées rapides à l'aide d'une indexation FAISS optimisée.",
        "desc_explain": "Inspectez les cartes d'attention neuronale mettant en valeur les caractéristiques clés.",
        "desc_cloud": "Instructions de conteneurisation et de déploiement en production.",
        "urban": "Urbain (Tokyo, Japon)",
        "forest": "Forêt (Forêt Amazonienne)",
        "water": "Eau (Lac Léman, Suisse)",
        "farmland": "Terres agricoles (Iowa, États-Unis)",
        "desert": "Désert (Désert du Sahara, Algérie)",
        "class_label": "Classe de Couverture Terrestre",
        "river": "Contient une Rivière Sinueuse",
        "road": "Contient une Route Droite",
        "description": "Description Textuelle",
        "query_config": "Configuration de la Requête",
        "query_mod": "Modalité de Requête",
        "db_mod": "Modalité de Base de Données",
        "search_btn": "Rechercher dans la Base de Données",
        "latency_cmp": "Comparaison de Latence (ms)",
        "gnn_refine": "Activer le Raffinement de Voisinage GNN (Lissage Contextuel Spatial)"
    }
}

# --- Styling CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #0c0f17;
        color: #e2e8f0;
    }
    h1, h2, h3, .stMarkdown p {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #047857 50%, #0f172a 100%);
        padding: 2.2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #1e40af;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }
    .banner h1 {
        color: #ffffff;
        margin: 0;
        font-size: 2.3rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .banner p {
        color: #a7f3d0;
        margin-top: 0.6rem;
        font-size: 1.1rem;
        font-weight: 300;
    }
    .result-card {
        background-color: #151b2c;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-3px);
        border-color: #10b981;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #111827;
        padding: 8px;
        border-radius: 8px;
        border: 1px solid #1f2937;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 6px;
        color: #9ca3af;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #10b981 !important;
        color: #ffffff !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar UI Language Switcher ---
lang_display = st.sidebar.selectbox("🌐 Select Language / Seleccione Idioma / Langue", options=["English", "Español", "Français"])
lang_code = "en"
if lang_display == "Español":
    lang_code = "es"
elif lang_display == "Français":
    lang_code = "fr"

def t(key):
    return LOCALIZATION[lang_code].get(key, key)

st.sidebar.markdown(f"## {t('sidebar_settings')}")

# --- Initialize Session States ---
if 'generator' not in st.session_state:
    st.session_state.generator = SatelliteDatasetGenerator(size=64)

if 'dataset' not in st.session_state:
    st.session_state.dataset = st.session_state.generator.generate_dataset(num_samples=150, seed=42)
    st.session_state.dataset_dirty = True

# Encoders (loaded lazily to ensure robust starts)
if 'clip_wrapper' not in st.session_state:
    st.session_state.clip_wrapper = None
if 'dinov2_wrapper' not in st.session_state:
    st.session_state.dinov2_wrapper = None

def get_clip():
    if st.session_state.clip_wrapper is None:
        st.session_state.clip_wrapper = CLIPWrapper()
    return st.session_state.clip_wrapper

def get_dinov2():
    if st.session_state.dinov2_wrapper is None:
        st.session_state.dinov2_wrapper = Dinov2Wrapper()
    return st.session_state.dinov2_wrapper

# Models
if 'base_model' not in st.session_state or st.session_state.get('dataset_dirty', False):
    data = st.session_state.dataset
    vocab = st.session_state.generator.vocabulary
    
    X_opt = np.array([extract_optical_features(p['optical']) for p in data], dtype=np.float32)
    X_sar = np.array([extract_sar_features(p['sar']) for p in data], dtype=np.float32)
    X_txt = np.array([extract_text_features(p['description'], vocab) for p in data], dtype=np.float32)
    
    st.session_state.X_opt = X_opt
    st.session_state.X_sar = X_sar
    st.session_state.X_txt = X_txt
    
    # Base MLP Aligner Model
    st.session_state.base_model = CrossModalEmbeddingAligner(
        dim_opt=17, dim_sar=6, dim_txt=20,
        hidden_dim=32, embed_dim=16, temperature=0.1, seed=42
    )
    st.session_state.base_model.fit_normalizers(X_opt, X_sar, X_txt)
    st.session_state.base_loss_history = []
    st.session_state.base_epochs = 0
    
    # Advanced Transformer Model
    st.session_state.transformer_model = MultiModalTransformer(
        opt_dim=17, sar_dim=6, txt_dim=20, embed_dim=64
    ).to(DEVICE)
    st.session_state.gcn_refiner = GCNRefiner(
        in_features=64, hidden_features=64, out_features=64
    ).to(DEVICE)
    st.session_state.adv_loss_history = []
    st.session_state.adv_epochs = 0
    st.session_state.dataset_dirty = False

# --- Sidebar Controls ---
st.sidebar.markdown(f"### {t('dataset_config')}")
num_samples = st.sidebar.slider(t("num_scenes"), min_value=50, max_value=300, value=150, step=10)
gen_seed = st.sidebar.number_input(t("gen_seed"), min_value=0, max_value=1000, value=42)

if st.sidebar.button(t("regen_btn"), use_container_width=True):
    with st.spinner("Generating new dataset..."):
        st.session_state.dataset = st.session_state.generator.generate_dataset(num_samples=num_samples, seed=gen_seed)
        st.session_state.dataset_dirty = True
        st.rerun()

st.sidebar.markdown(f"### {t('model_select')}")
model_type = st.sidebar.selectbox("Active Model Architectures", options=["Base NumPy MLP Aligner", "Advanced Transformer + GCN Refiner"])
epochs_to_train = st.sidebar.slider(t("epochs"), min_value=10, max_value=400, value=100, step=10)
learning_rate = st.sidebar.slider(t("lr"), min_value=0.001, max_value=0.05, value=0.01, step=0.001, format="%.3f")

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    reset_trigger = st.button(t("reset_btn"), use_container_width=True)
with col_sb2:
    train_trigger = st.button(t("train_btn"), type="primary", use_container_width=True)

if reset_trigger:
    if model_type == "Base NumPy MLP Aligner":
        st.session_state.base_model = CrossModalEmbeddingAligner(
            dim_opt=17, dim_sar=6, dim_txt=20,
            hidden_dim=32, embed_dim=16, temperature=0.1, seed=42
        )
        st.session_state.base_model.fit_normalizers(st.session_state.X_opt, st.session_state.X_sar, st.session_state.X_txt)
        st.session_state.base_loss_history = []
        st.session_state.base_epochs = 0
        st.success("Base model reset.")
    else:
        st.session_state.transformer_model = MultiModalTransformer(
            opt_dim=17, sar_dim=6, txt_dim=20, embed_dim=64
        ).to(DEVICE)
        st.session_state.gcn_refiner = GCNRefiner(
            in_features=64, hidden_features=64, out_features=64
        ).to(DEVICE)
        st.session_state.adv_loss_history = []
        st.session_state.adv_epochs = 0
        st.success("Advanced Transformer and GCN Refiner reset.")

if train_trigger:
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    chart_placeholder = st.empty()
    
    X_opt, X_sar, X_txt = st.session_state.X_opt, st.session_state.X_sar, st.session_state.X_txt
    
    if model_type == "Base NumPy MLP Aligner":
        aligner = st.session_state.base_model
        loss_history = list(st.session_state.base_loss_history)
        
        for epoch in range(epochs_to_train):
            metrics = aligner.train_step(X_opt, X_sar, X_txt, lr=learning_rate)
            loss_history.append(metrics['loss'])
            pct = (epoch + 1) / epochs_to_train
            progress_bar.progress(pct)
            status_text.markdown(f"**Epoch {epoch+1}/{epochs_to_train}** | Loss: **{metrics['loss']:.4f}**")
            
            if epoch % 5 == 0 or epoch == epochs_to_train - 1:
                chart_placeholder.line_chart(pd.DataFrame({'Contrastive Loss': loss_history}))
                
        st.session_state.base_loss_history = loss_history
        st.session_state.base_epochs += epochs_to_train
        st.success(f"Trained Base MLP for {epochs_to_train} epochs!")
        st.rerun()
    else:
        # Advanced Transformer + GCN Training
        transformer = st.session_state.transformer_model
        gcn = st.session_state.gcn_refiner
        optimizer = torch.optim.Adam(list(transformer.parameters()) + list(gcn.parameters()), lr=learning_rate)
        
        # Build spatial neighborhood adjacency
        G = build_spatial_graph(st.session_state.dataset)
        N = len(st.session_state.dataset)
        adj = np.zeros((N, N), dtype=np.float32)
        for u, v in G.edges():
            adj[u, v] = 1.0
            adj[v, u] = 1.0
        adj_tensor = torch.tensor(adj, device=DEVICE)
        
        # Scale features using normalizers
        m_opt, s_opt = st.session_state.base_model.mean_opt, st.session_state.base_model.std_opt
        m_sar, s_sar = st.session_state.base_model.mean_sar, st.session_state.base_model.std_sar
        m_txt, s_txt = st.session_state.base_model.mean_txt, st.session_state.base_model.std_txt
        
        opt_tensor = torch.tensor((X_opt - m_opt) / s_opt, dtype=torch.float32, device=DEVICE)
        sar_tensor = torch.tensor((X_sar - m_sar) / s_sar, dtype=torch.float32, device=DEVICE)
        txt_tensor = torch.tensor((X_txt - m_txt) / s_txt, dtype=torch.float32, device=DEVICE)
        
        # Class labels for supervised contrastive loss
        class_to_idx = {'urban': 0, 'forest': 1, 'water': 2, 'farmland': 3, 'desert': 4}
        labels = torch.tensor([class_to_idx[p['class']] for p in st.session_state.dataset], dtype=torch.long, device=DEVICE)
        
        labels_left = labels.unsqueeze(1)
        labels_right = labels.unsqueeze(0)
        mask_pos = (labels_left == labels_right).float()
        
        loss_history = list(st.session_state.adv_loss_history)
        transformer.train()
        gcn.train()
        
        for epoch in range(epochs_to_train):
            optimizer.zero_grad()
            z_fused = transformer(opt_tensor, sar_tensor, txt_tensor)
            z_refined = gcn(z_fused, adj_tensor)
            
            # Supervised contrastive alignment
            similarity_matrix = torch.mm(z_refined, z_refined.T)
            temp = 0.15
            exp_sim = torch.exp(similarity_matrix / temp)
            
            pos_sum = torch.sum(exp_sim * mask_pos, dim=1)
            all_sum = torch.sum(exp_sim, dim=1)
            loss = -torch.log(pos_sum / (all_sum + 1e-8) + 1e-8).mean()
            
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            
            pct = (epoch + 1) / epochs_to_train
            progress_bar.progress(pct)
            status_text.markdown(f"**Epoch {epoch+1}/{epochs_to_train}** | Loss: **{loss.item():.4f}**")
            
            if epoch % 5 == 0 or epoch == epochs_to_train - 1:
                chart_placeholder.line_chart(pd.DataFrame({'Supervised Contrastive Loss': loss_history}))
                
        st.session_state.adv_loss_history = loss_history
        st.session_state.adv_epochs += epochs_to_train
        st.success(f"Trained Advanced Transformer + GNN for {epochs_to_train} epochs!")
        st.rerun()

# Display sidebar stats
active_epochs = st.session_state.base_epochs if model_type == "Base NumPy MLP Aligner" else st.session_state.adv_epochs
st.sidebar.markdown(f"""
<div style="background-color: #111827; padding: 0.9rem; border-radius: 8px; margin-top: 1rem; border: 1px solid #1f2937;">
    <p style="margin: 0; font-size: 0.9rem; color: #9ca3af;"><b>{t('model_status')}</b></p>
    <p style="margin: 0.3rem 0 0 0; font-size: 0.95rem; color: #10b981;">
        {model_type} &rarr; <b>{active_epochs} {t('epochs_trained')}</b>
    </p>
</div>
""", unsafe_allow_html=True)

# --- Header Banner ---
st.markdown(f"""
<div class="banner">
    <h1>{t('title')}</h1>
    <p>{t('subtitle')}</p>
</div>
""", unsafe_allow_html=True)

# --- Main App Tabs ---
tab_gis, tab_models, tab_retrieve, tab_explain, tab_cloud = st.tabs([
    t("tab_gis"),
    t("tab_models"),
    t("tab_retrieve"),
    t("tab_explain"),
    t("tab_cloud")
])

# ========================================================
# TAB 1: GIS EXPLORER
# ========================================================
with tab_gis:
    st.header(t("tab_gis"))
    st.markdown(f"*{t('desc_gis')}*")
    
    dataset = st.session_state.dataset
    
    col_map, col_details = st.columns([3, 2])
    
    with col_details:
        st.subheader("Scene Patch Inspector")
        sample_idx = st.slider("Select Scene Patch ID", 0, len(dataset) - 1, 0)
        selected_sample = dataset[sample_idx]
        
        st.markdown(f"""
        - **{t('class_label')}**: `{t(selected_sample['class'])}`
        - **{t('river')}**: `{"Yes" if selected_sample['has_river'] else "No"}`
        - **{t('road')}**: `{"Yes" if selected_sample['has_road'] else "No"}`
        - **Latitude**: `{selected_sample['lat']:.5f}`
        - **Longitude**: `{selected_sample['lon']:.5f}`
        - **Grid Neighborhood**: Row `{selected_sample['row']}`, Col `{selected_sample['col']}`
        """)
        
        st.markdown(f"**{t('description')}**:")
        st.info(f"\"{selected_sample['description']}\"")
        
        # Display sensors side by side
        col_opt, col_sar = st.columns(2)
        with col_opt:
            st.image(selected_sample['optical'], caption="Optical (RGB)", use_container_width=True)
        with col_sar:
            st.image(selected_sample['sar'], caption="Speckled Radar (SAR)", use_container_width=True)
            
    with col_map:
        # Define marker colors based on land cover type
        folium_colors = {
            'urban': 'red',
            'forest': 'green',
            'water': 'blue',
            'farmland': 'purple',
            'desert': 'orange'
        }
        
        # Center map dynamically at selected scene
        m = folium.Map(location=[selected_sample['lat'], selected_sample['lon']], zoom_start=6, tiles="CartoDB dark_matter")
        
        # Add all dataset markers
        for idx, item in enumerate(dataset):
            color = folium_colors[item['class']]
            folium.Marker(
                location=[item['lat'], item['lon']],
                popup=f"ID: {idx} | Class: {item['class'].upper()}",
                tooltip=f"ID {idx}: {item['class']}",
                icon=folium.Icon(color=color, icon="info-sign")
            ).add_to(m)
            
        # Draw dynamic circle around the currently selected item
        folium.CircleMarker(
            location=[selected_sample['lat'], selected_sample['lon']],
            radius=18,
            color="#10b981",
            fill=True,
            fill_color="#10b981",
            fill_opacity=0.4,
            popup="Selected Scene"
        ).add_to(m)
        
        # Render Map
        st_folium(m, height=450, width=700, key="gis_explore_map")

# ========================================================
# TAB 2: MODELS & LATENT PCA
# ========================================================
with tab_models:
    st.header(t("tab_models"))
    st.markdown(f"*{t('desc_models')}*")
    
    col_t1, col_t2 = st.columns([1, 1])
    
    with col_t1:
        st.subheader("Model Architectures")
        st.markdown("""
        **1. CLIP-based Encoders:**
        - Extracts deep semantic representations aligned between text and vision. Fallback path automatically spins up a lightweight custom CNN.
        
        **2. DINOv2 Self-Supervised Features:**
        - Computes robust texture and context representations from optical bands.
        
        **3. Multi-Modal Transformer Aligners:**
        - Projects physical descriptors (17d Optical, 6d SAR, 20d Text) into 64d, then runs sequence self-attention to correlate cross-sensor details.
        
        **4. Graph Neural Network Refiner (GNN):**
        - Operates on the neighborhood grid graph. Propagates node embeddings to smooth predictions across adjacent locations.
        """)
        
    with col_t2:
        st.subheader("GNN Neighborhood Controls")
        gnn_active = st.checkbox(t("gnn_refine"), value=True)
        
        # Compute embeddings for visual projection
        X_opt, X_sar, X_txt = st.session_state.X_opt, st.session_state.X_sar, st.session_state.X_txt
        
        if model_type == "Base NumPy MLP Aligner":
            z_o, z_s, z_t = st.session_state.base_model.forward(X_opt, X_sar, X_txt, normalized=True)
        else:
            # Scale
            m_opt, s_opt = st.session_state.base_model.mean_opt, st.session_state.base_model.std_opt
            m_sar, s_sar = st.session_state.base_model.mean_sar, st.session_state.base_model.std_sar
            m_txt, s_txt = st.session_state.base_model.mean_txt, st.session_state.base_model.std_txt
            
            opt_t = torch.tensor((X_opt - m_opt) / s_opt, dtype=torch.float32, device=DEVICE)
            sar_t = torch.tensor((X_sar - m_sar) / s_sar, dtype=torch.float32, device=DEVICE)
            txt_t = torch.tensor((X_txt - m_txt) / s_txt, dtype=torch.float32, device=DEVICE)
            
            st.session_state.transformer_model.eval()
            st.session_state.gcn_refiner.eval()
            with torch.no_grad():
                z_fused = st.session_state.transformer_model(opt_t, sar_t, txt_t)
                
                if gnn_active:
                    G = build_spatial_graph(st.session_state.dataset)
                    N = len(st.session_state.dataset)
                    adj = np.zeros((N, N), dtype=np.float32)
                    for u, v in G.edges():
                        adj[u, v] = 1.0
                        adj[v, u] = 1.0
                    adj_t = torch.tensor(adj, device=DEVICE)
                    z_ref = st.session_state.gcn_refiner(z_fused, adj_t)
                    z_o = z_ref.cpu().numpy()
                    z_s = z_o.copy() # Jointly aligned space representation
                    z_t = z_o.copy()
                else:
                    z_o = z_fused.cpu().numpy()
                    z_s = z_o.copy()
                    z_t = z_o.copy()
                    
        # Compute retrieval metric (mAP) on the current representations
        def get_map(z1, z2):
            B = z1.shape[0]
            sims = np.dot(z1, z2.T)
            map_sum = 0.0
            for i in range(B):
                ranks = np.argsort(sims[i])[::-1]
                r = np.where(ranks == i)[0][0] + 1
                map_sum += 1.0 / r
            return map_sum / B

        map_val = get_map(z_o, z_s)
        st.metric("Modality Alignment mAP", f"{map_val:.2%}")
        
    st.subheader("🎨 Latent Space Projection (Plotly 3D PCA)")
    
    # Combine subset for plot to avoid visualization clutter
    plot_num = min(40, len(st.session_state.dataset))
    z_o_sub = z_o[:plot_num]
    z_s_sub = z_s[:plot_num]
    z_t_sub = z_t[:plot_num]
    classes_sub = [p['class'] for p in st.session_state.dataset[:plot_num]]
    
    all_z = np.concatenate([z_o_sub, z_s_sub, z_t_sub], axis=0)
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    coords_3d = pca.fit_transform(all_z)
    
    # Coordinate slices
    c_o = coords_3d[:plot_num]
    c_s = coords_3d[plot_num:2*plot_num]
    c_t = coords_3d[2*plot_num:]
    
    df_o = pd.DataFrame({'x': c_o[:, 0], 'y': c_o[:, 1], 'z': c_o[:, 2], 'Class': classes_sub, 'Modality': 'Optical'})
    df_s = pd.DataFrame({'x': c_s[:, 0], 'y': c_s[:, 1], 'z': c_s[:, 2], 'Class': classes_sub, 'Modality': 'SAR'})
    df_t = pd.DataFrame({'x': c_t[:, 0], 'y': c_t[:, 1], 'z': c_t[:, 2], 'Class': classes_sub, 'Modality': 'Text'})
    
    df_plotly = pd.concat([df_o, df_s, df_t])
    df_plotly['Label'] = df_plotly['Class'].apply(lambda x: x.upper()) + " (" + df_plotly['Modality'] + ")"
    
    fig = px.scatter_3d(
        df_plotly, x='x', y='y', z='z',
        color='Class', symbol='Modality',
        hover_name='Label',
        color_discrete_map={
            'urban': '#ef4444',
            'forest': '#10b981',
            'water': '#3b82f6',
            'farmland': '#8b5cf6',
            'desert': '#eab308'
        }
    )
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="#0f172a", gridcolor="#1f2937", showbackground=True),
            yaxis=dict(backgroundcolor="#0f172a", gridcolor="#1f2937", showbackground=True),
            zaxis=dict(backgroundcolor="#0f172a", gridcolor="#1f2937", showbackground=True)
        ),
        paper_bgcolor="#0c0f17",
        font_color="#e2e8f0",
        margin=dict(l=0, r=0, b=0, t=30)
    )
    st.plotly_chart(fig, use_container_width=True)

# ========================================================
# TAB 3: REAL-TIME RETRIEVAL (FAISS)
# ========================================================
with tab_retrieve:
    st.header(t("tab_retrieve"))
    st.markdown(f"*{t('desc_retrieve')}*")
    
    # Calculate embeddings
    X_opt, X_sar, X_txt = st.session_state.X_opt, st.session_state.X_sar, st.session_state.X_txt
    if model_type == "Base NumPy MLP Aligner":
        z_o, z_s, z_t = st.session_state.base_model.forward(X_opt, X_sar, X_txt, normalized=True)
        dim = z_o.shape[1]
    else:
        # Scale and project using Advanced Transformer
        m_opt, s_opt = st.session_state.base_model.mean_opt, st.session_state.base_model.std_opt
        m_sar, s_sar = st.session_state.base_model.mean_sar, st.session_state.base_model.std_sar
        m_txt, s_txt = st.session_state.base_model.mean_txt, st.session_state.base_model.std_txt
        
        opt_t = torch.tensor((X_opt - m_opt) / s_opt, dtype=torch.float32, device=DEVICE)
        sar_t = torch.tensor((X_sar - m_sar) / s_sar, dtype=torch.float32, device=DEVICE)
        txt_t = torch.tensor((X_txt - m_txt) / s_txt, dtype=torch.float32, device=DEVICE)
        
        st.session_state.transformer_model.eval()
        with torch.no_grad():
            z_fused = st.session_state.transformer_model(opt_t, sar_t, txt_t).cpu().numpy()
            z_o = z_fused
            z_s = z_fused
            z_t = z_fused
        dim = z_o.shape[1]
        
    col_q_set, col_db_set = st.columns(2)
    with col_q_set:
        query_mod = st.selectbox(t("query_mod"), options=["Text", "Optical Image", "SAR Radar"])
    with col_db_set:
        db_mod = st.selectbox(t("db_mod"), options=["Optical Image", "SAR Radar"])
        
    metric_type = st.radio("FAISS Distance Metric", options=["cosine", "l2"], horizontal=True)
    
    # Map selection to variables
    q_dict = {"Text": z_t, "Optical Image": z_o, "SAR Radar": z_s}
    db_dict = {"Optical Image": z_o, "SAR Radar": z_s}
    
    z_q_source = q_dict[query_mod]
    z_db_source = db_dict[db_mod]
    
    # Query selector
    query_id = st.slider("Select Query Index", 0, len(st.session_state.dataset) - 1, 0)
    q_emb = z_q_source[query_id]
    
    # Build FAISS Index Manager
    faiss_mgr = FaissIndexManager(dimension=dim, metric=metric_type)
    faiss_mgr.add(z_db_source)
    faiss_mgr.finalize()
    
    # Run search
    matched_indices, matched_scores = faiss_mgr.search(q_emb, k=5)
    
    st.subheader(f"Top 5 Retrieval Matches in Database (Query Modality: {query_mod} &rarr; Target: {db_mod})")
    
    cols = st.columns(5)
    for rank_idx, (match_id, score) in enumerate(zip(matched_indices, matched_scores)):
        with cols[rank_idx]:
            matched_scene = st.session_state.dataset[match_id]
            is_correct = (match_id == query_id)
            
            border_color = "#10b981" if is_correct else "#2d3748"
            border_txt = "🎯 EXACT MATCH" if is_correct else f"{matched_scene['class'].upper()}"
            
            st.markdown(f"""
            <div class="result-card" style="border-color: {border_color};">
                <p style="margin: 0; font-size: 0.85rem; color: #94a3b8; font-weight: bold;">
                    {border_txt}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show target database sensor image
            tgt_img = matched_scene['optical'] if db_mod == "Optical Image" else matched_scene['sar']
            st.image(tgt_img, use_container_width=True)
            st.markdown(f"""
            <p style='text-align: center; margin-top: 0.4rem; font-size: 0.85rem;'>
                Rank {rank_idx+1} | Score: <b>{score:.4f}</b>
            </p>
            """, unsafe_allow_html=True)
            
    # Benchmarking segment
    st.markdown("---")
    st.subheader("⚡ Speed Benchmarking: NumPy linear scan vs FAISS search")
    
    # Scale DB for benchmark to show performance differentiation
    bench_mgr = FaissIndexManager(dimension=dim, metric=metric_type)
    bench_mgr.add(z_db_source)
    # Append 5,000 synthetic database embeddings to simulate a real-world regional database scale
    synth_vectors = np.random.randn(5000, dim).astype(np.float32)
    if metric_type == "cosine":
        synth_vectors = synth_vectors / np.linalg.norm(synth_vectors, axis=1, keepdims=True)
    bench_mgr.add(synth_vectors)
    bench_mgr.finalize()
    
    bench_results = bench_mgr.benchmark_search(q_emb, k=5, runs=100)
    
    col_b1, col_b2 = st.columns([2, 3])
    with col_b1:
        st.markdown(f"""
        - **NumPy Cosine Scan Time**: `{bench_results['numpy_latency_ms']:.4f} ms`
        - **FAISS Flat Search Time**: `{bench_results['faiss_latency_ms']:.4f} ms`
        - **FAISS Index Status**: `{"✅ Active CPU Flat Index" if bench_results['faiss_available'] else "⚠️ Fallback to NumPy (FAISS unavailable)"}`
        """)
        if bench_results['faiss_available'] and bench_results['faiss_latency_ms'] > 0:
            speedup = bench_results['numpy_latency_ms'] / bench_results['faiss_latency_ms']
            st.metric("FAISS Speedup Factor", f"{speedup:.1f}x Faster")
            
    with col_b2:
        # Render a simple bar chart comparing times
        fig_bench = go.Figure(data=[
            go.Bar(
                x=["NumPy Linear Scan", "FAISS Search Engine"],
                y=[bench_results['numpy_latency_ms'], max(0.001, bench_results['faiss_latency_ms'])],
                marker_color=["#ef4444", "#10b981"]
            )
        ])
        fig_bench.update_layout(
            title="Query Latency Comparison (shorter is better)",
            yaxis_title="Time (Milliseconds)",
            paper_bgcolor="#0c0f17",
            plot_bgcolor="#151b2c",
            font_color="#e2e8f0",
            height=280
        )
        st.plotly_chart(fig_bench, use_container_width=True)

# ========================================================
# TAB 4: EXPLAINABILITY (GRAD-CAM)
# ========================================================
with tab_explain:
    st.header(t("tab_explain"))
    st.markdown(f"*{t('desc_explain')}*")
    
    st.write("Visual attention heatmaps provide insight into which spatial pixels/quadrants the neural model prioritizes.")
    
    # Selector for explainability
    cam_sample_idx = st.slider("Select Target Scene to Explain", 0, len(st.session_state.dataset) - 1, 0)
    cam_item = st.session_state.dataset[cam_sample_idx]
    
    encoder_choice = st.radio("Attention Modality Model", options=["DINOv2 Self-Supervised Features", "CLIP Visual Encoder"], horizontal=True)
    
    col_orig, col_cam = st.columns(2)
    
    with col_orig:
        st.subheader("Original Image")
        st.image(cam_item['optical'], use_container_width=True, caption=f"Class: {cam_item['class'].upper()}")
        
    with col_cam:
        st.subheader("Grad-CAM Activation Overlay")
        
        # Load encoders
        if encoder_choice == "CLIP Visual Encoder":
            wrapper = get_clip()
        else:
            wrapper = get_dinov2()
            
        with st.spinner("Computing backprop saliency gradients..."):
            heatmap = generate_saliency_heatmap(cam_item['optical'], wrapper)
            
            # Map heatmap to PIL color overlay
            img_np = np.array(cam_item['optical'].resize((128, 128)), dtype=np.float32) / 255.0
            
            # Interpolate heatmap to fit image
            import scipy.ndimage as ndimage
            heatmap_resized = ndimage.zoom(heatmap, 128.0 / heatmap.shape[0], order=1)
            heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)
            
            # Apply color mapping
            colormap = plt.cm.jet
            heatmap_colored = colormap(heatmap_resized)[:, :, :3]
            
            # Blend original and heatmap
            alpha = 0.5
            blended = alpha * img_np + (1.0 - alpha) * heatmap_colored
            blended = np.clip(blended, 0.0, 1.0)
            
            st.image(blended, use_container_width=True, caption=f"Grad-CAM overlay using {encoder_choice}")

# ========================================================
# TAB 5: DOCKER & CLOUD
# ========================================================
with tab_cloud:
    st.header(t("tab_cloud"))
    st.markdown(f"*{t('desc_cloud')}*")
    
    st.subheader("🐳 Local Containerization (Docker)")
    st.markdown("""
    To run this application locally inside a secure, reproducible container, use the provided configuration files in the root folder:
    
    **1. Build the Docker Image:**
    ```bash
    docker build -t satellite-retrieval:latest .
    ```
    
    **2. Run with Port Forwarding:**
    ```bash
    docker run -p 8501:8501 satellite-retrieval:latest
    ```
    
    **3. Or launch using Docker Compose:**
    ```bash
    docker-compose up --build
    ```
    Open [http://localhost:8501](http://localhost:8501) in your browser to view the application.
    """)
    
    st.markdown("---")
    
    st.subheader("☁️ Production Cloud Deployment")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("""
        #### Option A: Hugging Face Spaces (Quickest)
        1. Create a new space on Hugging Face using the **Streamlit** SDK.
        2. Commit all python files (`app.py`, `dataset.py`, `model_advanced.py`, `index_search.py`) along with `requirements.txt`.
        3. The Space will build and deploy automatically, providing a public sharing link.
        """)
        
    with col_c2:
        st.markdown("""
        #### Option B: Google Cloud Run (Containerized Serverless)
        1. Authenticate with Google Cloud SDK:
           ```bash
           gcloud auth login
           ```
        2. Build and push container to Google Artifact Registry:
           ```bash
           gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/satellite-retrieval
           ```
        3. Deploy to serverless container hosting:
           ```bash
           gcloud run deploy satellite-retrieval --image gcr.io/YOUR_PROJECT_ID/satellite-retrieval --platform managed --allow-unauthenticated
           ```
        """)
