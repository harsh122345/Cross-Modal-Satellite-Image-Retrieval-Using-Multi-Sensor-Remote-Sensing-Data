import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from dataset import (
    SatelliteDatasetGenerator,
    extract_optical_features,
    extract_sar_features,
    extract_text_features
)
from model import CrossModalEmbeddingAligner

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Cross-Modal Satellite Retrieval",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Dark Theme Styling ---
st.markdown("""
<style>
    /* Main body background and text */
    .stApp {
        background-color: #0d0f14;
        color: #e2e8f0;
    }
    
    /* Title and header formatting */
    h1, h2, h3, .stMarkdown p {
        font-family: 'Inter', sans-serif;
    }
    
    /* Top banner */
    .banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #064e3b 100%);
        padding: 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #1e40af;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .banner h1 {
        color: #f8fafc;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .banner p {
        color: #93c5fd;
        margin-top: 0.5rem;
        font-size: 1.1rem;
        font-weight: 300;
    }
    
    /* Custom Card Style for Retrieval Results */
    .result-card {
        background-color: #1a1e29;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    
    /* Sidebar styling */
    .css-1d391tw, [data-testid="stSidebar"] {
        background-color: #111420 !important;
        border-right: 1px solid #1f2937;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #111420;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid #1f2937;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e3a8a !important;
        color: #ffffff !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'generator' not in st.session_state:
    st.session_state.generator = SatelliteDatasetGenerator(size=64)

if 'dataset' not in st.session_state:
    # Generate default dataset
    st.session_state.dataset = st.session_state.generator.generate_dataset(num_samples=150, seed=42)
    st.session_state.dataset_dirty = True

if 'model' not in st.session_state or st.session_state.get('dataset_dirty', False):
    # Process features
    data = st.session_state.dataset
    vocab = st.session_state.generator.vocabulary
    
    X_opt = np.array([extract_optical_features(p['optical']) for p in data], dtype=np.float32)
    X_sar = np.array([extract_sar_features(p['sar']) for p in data], dtype=np.float32)
    X_txt = np.array([extract_text_features(p['description'], vocab) for p in data], dtype=np.float32)
    
    st.session_state.X_opt = X_opt
    st.session_state.X_sar = X_sar
    st.session_state.X_txt = X_txt
    
    # Initialize and fit model normalizers
    st.session_state.model = CrossModalEmbeddingAligner(
        dim_opt=17, dim_sar=6, dim_txt=20,
        hidden_dim=32, embed_dim=16, temperature=0.1, seed=42
    )
    st.session_state.model.fit_normalizers(X_opt, X_sar, X_txt)
    st.session_state.loss_history = []
    st.session_state.trained_epochs = 0
    st.session_state.dataset_dirty = False

# --- App Header Banner ---
st.markdown("""
<div class="banner">
    <h1>🛰️ Cross-Modal Satellite Image Retrieval</h1>
    <p>Multi-Sensor Alignment between Optical RGB, Synthetic Aperture Radar (SAR), and Text Descriptions</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Simulation Settings")

# Dataset Generator
st.sidebar.subheader("Dataset Configuration")
num_samples = st.sidebar.slider("Number of Scenes", min_value=50, max_value=300, value=150, step=10)
gen_seed = st.sidebar.number_input("Generator Seed", min_value=0, max_value=1000, value=42)

if st.sidebar.button("📦 Regenerate Dataset", use_container_width=True):
    with st.spinner("Generating synthetic remote sensing patches..."):
        st.session_state.dataset = st.session_state.generator.generate_dataset(num_samples=num_samples, seed=gen_seed)
        st.session_state.dataset_dirty = True
        st.rerun()

# Model Parameters
st.sidebar.subheader("Model Configuration")
hidden_dim = st.sidebar.selectbox("Hidden Dimension", options=[16, 32, 64, 128], index=1)
embed_dim = st.sidebar.selectbox("Joint Embedding Dim", options=[8, 16, 32, 64], index=1)
temp = st.sidebar.slider("InfoNCE Temperature", min_value=0.01, max_value=0.5, value=0.1, step=0.01)

# Training Settings
st.sidebar.subheader("Training Configuration")
epochs_to_train = st.sidebar.slider("Training Epochs", min_value=10, max_value=500, value=150, step=10)
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.001, max_value=0.05, value=0.01, step=0.001, format="%.3f")

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    if st.button("🔄 Reset Model", use_container_width=True):
        st.session_state.model = CrossModalEmbeddingAligner(
            dim_opt=17, dim_sar=6, dim_txt=20,
            hidden_dim=hidden_dim, embed_dim=embed_dim, temperature=temp, seed=42
        )
        st.session_state.model.fit_normalizers(st.session_state.X_opt, st.session_state.X_sar, st.session_state.X_txt)
        st.session_state.loss_history = []
        st.session_state.trained_epochs = 0
        st.success("Model reset completed.")

with col_sb2:
    train_trigger = st.button("🔥 Train Model", type="primary", use_container_width=True)

# Display Current Model Info
st.sidebar.markdown(f"""
<div style="background-color: #1a1e29; padding: 0.8rem; border-radius: 6px; margin-top: 1rem; border: 1px solid #1f2937;">
    <p style="margin: 0; font-size: 0.9rem; color: #94a3b8;"><b>Model Status:</b></p>
    <p style="margin: 0.2rem 0 0 0; font-size: 0.9rem; color: #10b981;">
        Trained Epochs: <b>{st.session_state.trained_epochs}</b>
    </p>
    <p style="margin: 0.2rem 0 0 0; font-size: 0.8rem; color: #94a3b8;">
        Dimensions: {st.session_state.model.branch_opt.W1.shape[0]}d (Opt) / 
        {st.session_state.model.branch_sar.W1.shape[0]}d (SAR) / 
        {st.session_state.model.branch_txt.W1.shape[0]}d (Txt) &rarr; <b>{embed_dim}d (Shared)</b>
    </p>
</div>
""", unsafe_allow_html=True)

# --- Handle Inline Training Request ---
if train_trigger:
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    # Simple line plot container
    chart_placeholder = st.empty()
    
    X_opt, X_sar, X_txt = st.session_state.X_opt, st.session_state.X_sar, st.session_state.X_txt
    aligner = st.session_state.model
    
    # Temporarily override temperature in case user changed it in slider
    aligner.temperature = temp
    
    loss_history = list(st.session_state.loss_history)
    
    for epoch in range(epochs_to_train):
        metrics = aligner.train_step(X_opt, X_sar, X_txt, lr=learning_rate)
        loss_history.append(metrics['loss'])
        
        # Update progress and stats
        pct = (epoch + 1) / epochs_to_train
        progress_bar.progress(pct)
        status_text.markdown(f"**Epoch {epoch+1}/{epochs_to_train}** | Loss: **{metrics['loss']:.4f}** (OS: {metrics['loss_os']:.3f}, OT: {metrics['loss_ot']:.3f}, ST: {metrics['loss_st']:.3f})")
        
        # Periodically show loss curve
        if epoch % 5 == 0 or epoch == epochs_to_train - 1:
            df_loss = pd.DataFrame({'Total Contrastive Loss': loss_history})
            chart_placeholder.line_chart(df_loss)
            
    st.session_state.loss_history = loss_history
    st.session_state.trained_epochs += epochs_to_train
    st.success(f"Successfully trained model for {epochs_to_train} epochs!")
    st.rerun()

# --- Main Application Interface (Tabs) ---
tab_explore, tab_train, tab_retrieve = st.tabs([
    "📂 Explore Dataset",
    "🧠 Model Training & Embedding Space",
    "🔎 Retrieval Playground"
])

# ==========================================
# TAB 1: EXPLORE DATASET
# ==========================================
with tab_explore:
    st.header("📂 Multi-Sensor Satellite Dataset")
    st.markdown("""
    This generator simulates paired remote sensing imagery across different physical sensor architectures.
    - **Optical Imagery**: Represents passive solar reflective spectral bands (RGB color space). Shows colors, crop status, building materials.
    - **Synthetic Aperture Radar (SAR)**: Represents active microwave backscatter. Water acts as a specular reflector (looks very dark). Buildings generate double-bounce corner reflections (look extremely bright/white). Vegetation creates volume scattering (looks grainy gray).
    - **Text Descriptions**: Semantic descriptions referencing land cover type and key local elements (e.g. roads, rivers).
    """)
    
    # View statistics
    df_dataset = pd.DataFrame(st.session_state.dataset)
    class_counts = df_dataset['class'].value_counts()
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("Total Paired Samples", len(st.session_state.dataset))
    with col_stat2:
        st.metric("Land Cover Classes", len(class_counts))
    with col_stat3:
        st.metric("Sensor Modalities", "3 (Optical, SAR, Text)")
        
    st.subheader("Interactive Scene Browser")
    sample_idx = st.slider("Select Sample Index", 0, len(st.session_state.dataset)-1, 0)
    
    selected_sample = st.session_state.dataset[sample_idx]
    
    col_img1, col_img2, col_desc = st.columns([1.2, 1.2, 2])
    with col_img1:
        st.image(selected_sample['optical'], caption="Optical (RGB) Band", use_container_width=True)
    with col_img2:
        st.image(selected_sample['sar'], caption="Synthetic Aperture Radar (SAR) Speckled Band", use_container_width=True)
    with col_desc:
        st.markdown(f"""
        ### Scene Metadata
        - **Land Cover Class**: `{selected_sample['class'].upper()}`
        - **Contains Winding River**: `{"Yes" if selected_sample['has_river'] else "No"}`
        - **Contains Straight Road**: `{"Yes" if selected_sample['has_road'] else "No"}`
        
        ### Natural Language Description
        *{selected_sample['description']}*
        
        ### Raw Feature Representations
        We extract hand-crafted descriptors representing physical modalities:
        - **Optical (17 features)**: Mean and Standard Deviation of RGB channels, 3-bin color histograms, and Sobel edge-gradient statistics.
        - **SAR (6 features)**: Mean backscatter, standard deviation, and quadrant standard deviations for structural grain texture.
        - **Text (20 features)**: L2-normalized term counts of satellite-specific vocabulary terms.
        """)
        
    # Feature distribution expander
    with st.expander("📊 View Raw Feature Values"):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.write("Optical Feature Vector", st.session_state.X_opt[sample_idx])
        with col_f2:
            st.write("SAR Feature Vector", st.session_state.X_sar[sample_idx])
        with col_f3:
            st.write("Text Feature Vector (Vocab counts)", st.session_state.X_txt[sample_idx])


# ==========================================
# TAB 2: MODEL TRAINING & LATENT SPACE
# ==========================================
with tab_train:
    st.header("🧠 Joint Multi-Modal Alignment")
    
    st.markdown("""
    Here we align the three modal feature representations in a shared **contrastive embedding space**. 
    Before training, vectors are randomly distributed. After training, paired features are projected close together, allowing cross-sensor retrieval.
    """)
    
    # Metrics computation
    X_opt, X_sar, X_txt = st.session_state.X_opt, st.session_state.X_sar, st.session_state.X_txt
    aligner = st.session_state.model
    
    # Compute embeddings
    z_opt, z_sar, z_txt = aligner.forward(X_opt, X_sar, X_txt, normalized=True)
    
    # Evaluate retrieval accuracy
    def evaluate_retrieval(z_query, z_database):
        B = z_query.shape[0]
        similarities = np.dot(z_query, z_database.T) # B x B
        
        recalls = {1: 0, 5: 0}
        map_sum = 0.0
        
        for i in range(B):
            sims = similarities[i]
            sorted_indices = np.argsort(sims)[::-1] # Rank descending
            
            # Find rank of ground truth matching index (which is i)
            rank = np.where(sorted_indices == i)[0][0] + 1 # 1-based rank
            
            if rank == 1:
                recalls[1] += 1
            if rank <= 5:
                recalls[5] += 1
                
            map_sum += 1.0 / rank
            
        return recalls[1] / B, recalls[5] / B, map_sum / B

    r1_os, r5_os, map_os = evaluate_retrieval(z_opt, z_sar)
    r1_to, r5_to, map_to = evaluate_retrieval(z_txt, z_opt)
    r1_ts, r5_ts, map_ts = evaluate_retrieval(z_txt, z_sar)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(f"""
        <div style="background-color: #1a1e29; padding: 1rem; border-radius: 8px; border: 1px solid #1f2937; text-align: center;">
            <h4 style="color: #3b82f6; margin-top:0;">Optical &rarr; SAR Alignment</h4>
            <p style="font-size: 1.8rem; font-weight: 700; margin: 0.5rem 0;">mAP: {map_os:.1%}</p>
            <p style="color: #94a3b8; margin: 0; font-size:0.9rem;">Recall@1: <b>{r1_os:.1%}</b> | Recall@5: <b>{r5_os:.1%}</b></p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_m2:
        st.markdown(f"""
        <div style="background-color: #1a1e29; padding: 1rem; border-radius: 8px; border: 1px solid #1f2937; text-align: center;">
            <h4 style="color: #10b981; margin-top:0;">Text &rarr; Optical Alignment</h4>
            <p style="font-size: 1.8rem; font-weight: 700; margin: 0.5rem 0;">mAP: {map_to:.1%}</p>
            <p style="color: #94a3b8; margin: 0; font-size:0.9rem;">Recall@1: <b>{r1_to:.1%}</b> | Recall@5: <b>{r5_to:.1%}</b></p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_m3:
        st.markdown(f"""
        <div style="background-color: #1a1e29; padding: 1rem; border-radius: 8px; border: 1px solid #1f2937; text-align: center;">
            <h4 style="color: #f59e0b; margin-top:0;">Text &rarr; SAR Alignment</h4>
            <p style="font-size: 1.8rem; font-weight: 700; margin: 0.5rem 0;">mAP: {map_ts:.1%}</p>
            <p style="color: #94a3b8; margin: 0; font-size:0.9rem;">Recall@1: <b>{r1_ts:.1%}</b> | Recall@5: <b>{r5_ts:.1%}</b></p>
        </div>
        """, unsafe_allow_html=True)

    # Embedding Space Visualizer
    st.subheader("🎨 Latent Space Projection (2D PCA)")
    
    st.markdown("""
    This chart visualizes the aligned embeddings projected onto the first 2 principal components. 
    Paired elements are connected by thin lines. Notice how scenes representing the same land cover type form tight clusters after training.
    """)
    
    # Choose sample size for visualization to keep it clean
    vis_samples = min(30, len(st.session_state.dataset))
    
    # Select first vis_samples
    z_o_vis = z_opt[:vis_samples]
    z_s_vis = z_sar[:vis_samples]
    z_t_vis = z_txt[:vis_samples]
    
    classes_vis = [p['class'] for p in st.session_state.dataset[:vis_samples]]
    
    # Combine all embeddings to compute joint PCA
    all_embeddings = np.concatenate([z_o_vis, z_s_vis, z_t_vis], axis=0)
    
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)
    
    # Split back
    coords_o = embeddings_2d[:vis_samples]
    coords_s = embeddings_2d[vis_samples:2*vis_samples]
    coords_t = embeddings_2d[2*vis_samples:]
    
    # Plot using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor('#0d0f14')
    ax.set_facecolor('#1a1e29')
    
    # Define colors for land classes
    color_map = {
        'urban': '#ef4444',     # Red
        'forest': '#10b981',    # Green
        'water': '#3b82f6',     # Blue
        'farmland': '#8b5cf6',  # Purple
        'desert': '#eab308'     # Yellow
    }
    
    # Draw connections first (so markers stay on top)
    for i in range(vis_samples):
        c = color_map[classes_vis[i]]
        # Line Optical-SAR
        ax.plot([coords_o[i, 0], coords_s[i, 0]], [coords_o[i, 1], coords_s[i, 1]], color=c, alpha=0.25, linestyle=':')
        # Line Optical-Text
        ax.plot([coords_o[i, 0], coords_t[i, 0]], [coords_o[i, 1], coords_t[i, 1]], color=c, alpha=0.25, linestyle=':')
        
    # Draw markers for modalities
    for cl in np.unique(classes_vis):
        idx = [i for i, x in enumerate(classes_vis) if x == cl]
        color = color_map[cl]
        
        # Label once for legend
        ax.scatter(coords_o[idx, 0], coords_o[idx, 1], c=color, marker='o', s=80, label=f'{cl.upper()} (Optical)')
        ax.scatter(coords_s[idx, 0], coords_s[idx, 1], c=color, marker='s', s=80, edgecolors='white', linewidths=0.5, label=f'{cl.upper()} (SAR)')
        ax.scatter(coords_t[idx, 0], coords_t[idx, 1], c=color, marker='*', s=140, label=f'{cl.upper()} (Text)')
        
    ax.set_title("Aligned Embedding Space of Remote Sensing Modalities", color='#f8fafc', fontsize=12, fontweight='bold')
    ax.tick_params(colors='#94a3b8')
    ax.spines['bottom'].color = '#2d3748'
    ax.spines['top'].color = '#2d3748'
    ax.spines['left'].color = '#2d3748'
    ax.spines['right'].color = '#2d3748'
    ax.grid(True, color='#2d3748', linestyle='--', alpha=0.5)
    
    # Legend
    legend = ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", facecolor='#111420', edgecolor='#2d3748')
    for text in legend.get_texts():
        text.set_color('#e2e8f0')
        
    plt.tight_layout()
    st.pyplot(fig)


# ==========================================
# TAB 3: RETRIEVAL PLAYGROUND
# ==========================================
with tab_retrieve:
    st.header("🔎 Interactive Cross-Modal Retrieval Search Engine")
    
    st.markdown("""
    Use one sensor modality to query another sensor modality in the database. Choose your search mode below:
    """)
    
    search_mode = st.radio(
        "Search Query Modality",
        options=["Text to Multi-Sensor (Optical & SAR)", "Optical to SAR", "SAR to Optical"],
        horizontal=True
    )
    
    dataset = st.session_state.dataset
    vocab = st.session_state.generator.vocabulary
    
    # Pre-computed embeddings
    z_opt, z_sar, z_txt = st.session_state.model.forward(
        st.session_state.X_opt, st.session_state.X_sar, st.session_state.X_txt, normalized=True
    )
    
    if search_mode == "Text to Multi-Sensor (Optical & SAR)":
        col_q1, col_q2 = st.columns([2, 1])
        with col_q1:
            custom_query = st.text_input(
                "Enter text search query", 
                value="A winding river passing through a dense forest canopy"
            )
        with col_q2:
            st.markdown("<br>", unsafe_allow_html=True)
            preset_query = st.selectbox(
                "Or choose a preset query",
                options=[
                    "Custom Query...",
                    "A dense green forest canopy with high tree density",
                    "An urban district with multiple buildings and concrete streets",
                    "A body of open water with a calm, deep blue surface",
                    "A grid of agricultural farmland fields with cultivated crops",
                    "An arid desert landscape featuring wind-swept sand dunes"
                ]
            )
            
        query_text = preset_query if preset_query != "Custom Query..." else custom_query
        
        # Extract and project text
        q_feat = extract_text_features(query_text, vocab).reshape(1, -1)
        # Scale
        q_feat_norm = (q_feat - st.session_state.model.mean_txt) / st.session_state.model.std_txt
        u_q = st.session_state.model.branch_txt.forward(q_feat_norm)
        z_q = u_q / np.linalg.norm(u_q, axis=1, keepdims=True)
        
        # Find matches in Optical and SAR database
        sims_opt = np.dot(z_q, z_opt.T)[0]
        sims_sar = np.dot(z_q, z_sar.T)[0]
        
        ranks_opt = np.argsort(sims_opt)[::-1][:5]
        ranks_sar = np.argsort(sims_sar)[::-1][:5]
        
        col_ret1, col_ret2 = st.columns(2)
        with col_ret1:
            st.subheader("Top Retrieved Optical Images")
            for rank_i, idx in enumerate(ranks_opt):
                sim = sims_opt[idx]
                scene = dataset[idx]
                
                col_r_img, col_r_meta = st.columns([1, 2])
                with col_r_img:
                    st.image(scene['optical'], use_container_width=True)
                with col_r_meta:
                    st.markdown(f"""
                    **Rank {rank_i+1}** | Match Similarity: `{sim:.3f}`
                    - Class: `{scene['class'].upper()}`
                    - Ground Truth: *"{scene['description']}"*
                    """)
                    st.markdown("---")
                    
        with col_ret2:
            st.subheader("Top Retrieved SAR Radar Images")
            for rank_i, idx in enumerate(ranks_sar):
                sim = sims_sar[idx]
                scene = dataset[idx]
                
                col_r_img, col_r_meta = st.columns([1, 2])
                with col_r_img:
                    st.image(scene['sar'], use_container_width=True)
                with col_r_meta:
                    st.markdown(f"""
                    **Rank {rank_i+1}** | Match Similarity: `{sim:.3f}`
                    - Class: `{scene['class'].upper()}`
                    - Ground Truth: *"{scene['description']}"*
                    """)
                    st.markdown("---")
                    
    elif search_mode == "Optical to SAR":
        st.subheader("Query SAR sensor database using Optical RGB images")
        
        query_idx = st.slider("Select Query Optical Image Index", 0, len(dataset)-1, 0)
        
        col_q_img, col_q_meta = st.columns([1.5, 3])
        with col_q_img:
            st.image(dataset[query_idx]['optical'], caption="Query Optical Image", use_container_width=True)
        with col_q_meta:
            st.markdown(f"""
            ### Query Scene Details
            - **Land Cover Class**: `{dataset[query_idx]['class'].upper()}`
            - **Ground Truth Text**: *"{dataset[query_idx]['description']}"*
            """)
            
        # Get query embedding
        z_q = z_opt[query_idx].reshape(1, -1)
        
        # Search SAR database
        sims_sar = np.dot(z_q, z_sar.T)[0]
        ranks_sar = np.argsort(sims_sar)[::-1][:5]
        
        st.subheader("Top Matching SAR Radar Patches")
        cols = st.columns(5)
        for i, idx in enumerate(ranks_sar):
            with cols[i]:
                sim = sims_sar[idx]
                scene = dataset[idx]
                is_correct = (idx == query_idx)
                
                border_color = "#10b981" if is_correct else "#2d3748"
                border_text = "🎯 EXACT MATCH" if is_correct else f"Class: {scene['class'].upper()}"
                
                st.markdown(f"""
                <div class="result-card" style="border-color: {border_color};">
                    <p style="margin: 0 0 0.5rem 0; font-size: 0.85rem; color: #94a3b8; font-weight: bold;">
                        {border_text}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.image(scene['sar'], use_container_width=True)
                st.markdown(f"""
                <p style="text-align: center; margin-top: 0.3rem; font-size:0.85rem;">
                    Rank {i+1} | Sim: <b>{sim:.3f}</b>
                </p>
                """, unsafe_allow_html=True)
                
    elif search_mode == "SAR to Optical":
        st.subheader("Query Optical RGB database using SAR radar images")
        
        query_idx = st.slider("Select Query SAR Image Index", 0, len(dataset)-1, 0)
        
        col_q_img, col_q_meta = st.columns([1.5, 3])
        with col_q_img:
            st.image(dataset[query_idx]['sar'], caption="Query SAR Image", use_container_width=True)
        with col_q_meta:
            st.markdown(f"""
            ### Query Scene Details
            - **Land Cover Class**: `{dataset[query_idx]['class'].upper()}`
            - **Ground Truth Text**: *"{dataset[query_idx]['description']}"*
            """)
            
        # Get query embedding
        z_q = z_sar[query_idx].reshape(1, -1)
        
        # Search Optical database
        sims_opt = np.dot(z_q, z_opt.T)[0]
        ranks_opt = np.argsort(sims_opt)[::-1][:5]
        
        st.subheader("Top Matching Optical Patches")
        cols = st.columns(5)
        for i, idx in enumerate(ranks_opt):
            with cols[i]:
                sim = sims_opt[idx]
                scene = dataset[idx]
                is_correct = (idx == query_idx)
                
                border_color = "#10b981" if is_correct else "#2d3748"
                border_text = "🎯 EXACT MATCH" if is_correct else f"Class: {scene['class'].upper()}"
                
                st.markdown(f"""
                <div class="result-card" style="border-color: {border_color};">
                    <p style="margin: 0 0 0.5rem 0; font-size: 0.85rem; color: #94a3b8; font-weight: bold;">
                        {border_text}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.image(scene['optical'], use_container_width=True)
                st.markdown(f"""
                <p style="text-align: center; margin-top: 0.3rem; font-size:0.85rem;">
                    Rank {i+1} | Sim: <b>{sim:.3f}</b>
                </p>
                """, unsafe_allow_html=True)
