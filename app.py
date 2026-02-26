"""
Advanced Video Stabilization Dashboard
Dashboard Streamlit per confrontare Motion Estimation (RAFT Deep Learning vs SIFT Feature-Based) 
e Trajectory Smoothing (Media Mobile vs Kalman Filter).
"""

import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
import time
import json
import base64
from video_stabilization import (
    process_video,
    stabilize_video_moving_average,
    stabilize_video_kalman,
    get_video_info
)
from metrics import (
    generate_metrics_report,
    save_metrics_to_json,
    format_metrics_for_display,
    compute_stability_score
)

# Configurazione pagina
st.set_page_config(
    page_title="Advanced Video Stabilization",
    page_icon="🎥",
    layout="wide"
)


def frames_to_video(frames, output_path, fps=30):
    """
    Converte una lista di frame in un video MP4.
    
    Args:
        frames: Lista di frame (numpy arrays)
        output_path: Percorso del file video di output
        fps: Frame rate del video
    """
    if len(frames) == 0:
        return False
    
    h, w = frames[0].shape[:2]
    
    # Prova diversi codec in ordine di preferenza
    codecs = ['avc1', 'mp4v', 'X264', 'H264']
    
    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            if out.isOpened():
                for frame in frames:
                    out.write(frame)
                out.release()
                return True
        except:
            continue
    
    # Fallback: usa il codec di default
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()
    return True


def plot_trajectory_comparison(trajectory_raw, trajectory_ma, trajectory_kalman):
    """
    Crea un grafico che mostra le traiettorie sovrapposte.
    Evidenzia visivamente il ritardo della Media Mobile.
    
    Args:
        trajectory_raw: Traiettoria RAW da motion estimation (instabile)
        trajectory_ma: Traiettoria smoothata con Media Mobile (latenza)
        trajectory_kalman: Traiettoria smoothata con Kalman (ottimale)
        
    Returns:
        Figure matplotlib
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Trova lunghezza minima per uniformare tutte le traiettorie
    min_len = len(trajectory_raw)
    
    # Taglia tutte le traiettorie alla lunghezza minima
    trajectory_raw = trajectory_raw[:min_len]
    trajectory_ma = trajectory_ma[:min_len]
    trajectory_kalman = trajectory_kalman[:min_len]
    
    frames = np.arange(min_len)
    
    # Plot X
    axes[0].plot(frames, trajectory_raw[:, 0], 'gray', alpha=0.4, linewidth=0.8, label='RAW (Motion Estimation)', linestyle='--')
    axes[0].plot(frames, trajectory_ma[:, 0], 'b-', linewidth=1.8, label='Media Mobile', alpha=0.8)
    axes[0].plot(frames, trajectory_kalman[:, 0], 'g-', linewidth=1.8, label='Kalman Filter', alpha=0.8)
    axes[0].set_ylabel('Traslazione X (px)', fontsize=10)
    axes[0].set_title('Confronto Traiettorie: Asse X', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Plot Y
    axes[1].plot(frames, trajectory_raw[:, 1], 'gray', alpha=0.4, linewidth=0.8, label='RAW (Motion Estimation)', linestyle='--')
    axes[1].plot(frames, trajectory_ma[:, 1], 'b-', linewidth=1.8, label='Media Mobile', alpha=0.8)
    axes[1].plot(frames, trajectory_kalman[:, 1], 'g-', linewidth=1.8, label='Kalman Filter', alpha=0.8)
    axes[1].set_ylabel('Traslazione Y (px)', fontsize=10)
    axes[1].set_title('Confronto Traiettorie: Asse Y', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # Plot Angle
    axes[2].plot(frames, np.degrees(trajectory_raw[:, 2]), 'gray', alpha=0.4, linewidth=0.8, label='RAW (Motion Estimation)', linestyle='--')
    axes[2].plot(frames, np.degrees(trajectory_ma[:, 2]), 'b-', linewidth=1.8, label='Media Mobile', alpha=0.8)
    axes[2].plot(frames, np.degrees(trajectory_kalman[:, 2]), 'g-', linewidth=1.8, label='Kalman Filter', alpha=0.8)
    axes[2].set_ylabel('Rotazione (gradi)', fontsize=10)
    axes[2].set_xlabel('Frame', fontsize=10)
    axes[2].set_title('Confronto Traiettorie: Rotazione', fontsize=12, fontweight='bold')
    axes[2].legend(loc='upper right', fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    # Titolo principale
    st.title("🎥 Modern Video Stabilization: Deep Learning vs Feature-Based")
    st.markdown("""
    ### Dashboard Comparativa Avanzata
    Confronto tra **Deep Learning Optical Flow (RAFT)** e **Feature Matching Classico (SIFT)** 
    per Motion Estimation, con **Trajectory Smoothing** (Kalman vs Media Mobile).
    """)
    
    # Sidebar con controlli
    st.sidebar.header("⚙️ Configurazione Pipeline")
    
    # ===== MODULO 1: MOTION ESTIMATION =====
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Motion Estimation")
    
    motion_method = st.sidebar.selectbox(
        "Algoritmo di Stima del Movimento",
        ["RAFT (Deep Learning Optical Flow) 🚀", 
         "SIFT (Feature Matching Classico)"],
        help="RAFT: Deep Learning denso. SIFT: Feature matching tradizionale."
    )
    
    # Mappa la selezione al parametro
    if 'RAFT' in motion_method:
        method_key = 'raft'
    else:
        method_key = 'sift'
    
    # Inizializza tutte le variabili con valori di default
    # (saranno sovrascritte solo per il metodo selezionato)
    raft_use_small = False
    raft_num_samples = 200
    sift_n_features = 500
    sift_contrast_threshold = 0.04
    sift_edge_threshold = 10
    sift_sigma = 1.6
    sift_ratio_threshold = 0.75
    
    # ===== PARAMETRI SPECIFICI PER ALGORITMO =====
    st.sidebar.markdown("#### ⚙️ Parametri Algoritmo")
    
    if method_key == 'raft':
        # Parametri RAFT Deep Learning
        with st.sidebar.expander("🔧 Configura RAFT (Deep Learning)", expanded=True):
            
            raft_model_size = st.radio(
                "Modello RAFT",
                ["Large (Accurato, ~300MB)", "Small (Veloce, ~100MB)"],
                help="Large: migliore qualità, più lento. Small: più veloce, leggermente meno accurato",
                key="raft_model"
            )
            
            raft_use_small = 'Small' in raft_model_size
            
            raft_num_samples = st.slider(
                "Punti di Campionamento",
                min_value=100,
                max_value=500,
                value=200,
                step=50,
                help="Numero di punti estratti dal flow denso per stimare la trasformazione",
                key="raft_samples"
            )
            
            # Mostra quale device sarà usato
            try:
                import torch
                device = "GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU"
                st.info(f"**Device**: {device}")
                if not torch.cuda.is_available():
                    st.warning("⚠️ GPU non rilevata. RAFT sarà più lento su CPU (~2-5 FPS)")
            except:
                st.error("❌ PyTorch non trovato. Installa con: `pip install torch torchvision`")
        
        st.sidebar.success(f"✅ RAFT: {'Small' if raft_use_small else 'Large'}, {raft_num_samples} samples")
        
    else:  # SIFT
        # Parametri SIFT Feature Matching
        with st.sidebar.expander("🔧 Configura SIFT", expanded=False):
            st.markdown("**Feature Detection**")
            
            sift_n_features = st.slider(
                "Numero Max Features",
                min_value=100,
                max_value=1000,
                value=500,
                step=50,
                help="Numero massimo di keypoints SIFT da estrarre",
                key="sift_features"
            )
            
            sift_contrast_threshold = st.slider(
                "Contrast Threshold",
                min_value=0.01,
                max_value=0.1,
                value=0.04,
                step=0.01,
                format="%.2f",
                help="Soglia contrasto (più alto = meno keypoints ma più robusti)",
                key="sift_contrast"
            )
            
            sift_edge_threshold = st.slider(
                "Edge Threshold",
                min_value=5,
                max_value=20,
                value=10,
                step=1,
                help="Soglia per filtrare edge (più alto = meno edge responses)",
                key="sift_edge"
            )
            
            sift_sigma = st.slider(
                "Sigma (Blur)",
                min_value=1.0,
                max_value=2.0,
                value=1.6,
                step=0.1,
                format="%.1f",
                help="Sigma per Gaussian blur iniziale",
                key="sift_sigma"
            )
            
            st.markdown("**Feature Matching**")
            
            sift_ratio_threshold = st.slider(
                "Lowe's Ratio Test",
                min_value=0.5,
                max_value=0.9,
                value=0.75,
                step=0.05,
                format="%.2f",
                help="Soglia per ratio test (più basso = match più selettivi)",
                key="sift_ratio"
            )
        
        st.sidebar.success(f"✅ SIFT: max {sift_n_features} features, ratio {sift_ratio_threshold}")
    
    # ===== MODULO 2: TRAJECTORY SMOOTHING =====
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Trajectory Smoothing")
    
    # Media Mobile
    st.sidebar.markdown("#### Media Mobile")
    smoothing_radius = st.sidebar.slider(
        "Raggio Finestra (frame)",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
        help="Finestra temporale per smoothing. ⬆️ = più smooth + più latenza"
    )
    
    # Filtro di Kalman
    st.sidebar.markdown("#### Filtro di Kalman (6 Stati)")
    
    # Preset per semplificare
    kalman_preset = st.sidebar.selectbox(
        "Preset Kalman",
        ["Bilanciato", "Aggressivo", "Ultra Smooth (video mossi)", "Reattivo (min lag)", "Custom"],
        help="Configurazioni pre-definite per Q e R"
    )
    
    if kalman_preset == "Bilanciato":
        process_noise = 0.0005
        measurement_noise = 2.0
    elif kalman_preset == "Aggressivo":
        process_noise = 0.00005
        measurement_noise = 5.0
    elif kalman_preset == "Ultra Smooth (video mossi)":
        process_noise = 0.00001  # Q molto basso = modello rigido
        measurement_noise = 15.0  # R molto alto = ignora jitter, ratio 1,500,000!
    elif kalman_preset == "Reattivo (min lag)":
        process_noise = 0.005
        measurement_noise = 0.5
    else:  # Custom
        with st.sidebar.expander("🔧 Configura Kalman Custom", expanded=True):
            st.markdown("""
            **Process Noise (Q)**: Quanto può cambiare il movimento
            - ⬇️ Basso = movimento fluido predicibile
            - ⬆️ Alto = movimento può cambiare rapidamente
            """)
            
            process_noise = st.slider(
                "Process Noise (Q)",
                min_value=0.00001,
                max_value=0.01,
                value=0.0005,
                step=0.00001,
                format="%.5f",
                help="Varianza del modello di movimento",
                key="kalman_q"
            )
            
            st.markdown("""
            **Measurement Noise (R)**: Quanto sono rumorose le misure
            - ⬆️ Alto = ignora jitter, smooth aggressivo
            - ⬇️ Basso = segue misure da vicino
            """)
            
            measurement_noise = st.slider(
                "Measurement Noise (R)",
                min_value=0.1,
                max_value=20.0,  # Aumentato a 20 per smoothing ultra-aggressivo
                value=2.0,
                step=0.1,
                help="Varianza delle misurazioni (più alto = più smooth)",
                key="kalman_r"
            )
            
            # Visualizza rapporto Q/R
            ratio = measurement_noise / process_noise
            st.metric("Rapporto R/Q", f"{ratio:.0f}", 
                     help="Più alto = più smooth. Valore tipico: 1000-10000")
            
            # Warning per oversmoothing
            if process_noise < 0.0001 and measurement_noise > 10.0:
                st.warning("""
                ⚠️ **Attenzione Oversmoothing**!
                - Q molto basso + R molto alto
                - Kalman seguirà troppo lentamente
                - Metriche potrebbero favorire MA
                
                💡 Prova "Aggressivo" o "Bilanciato"
                """)
            elif ratio > 200000:
                st.warning(f"""
                ⚠️ **R/Q troppo alto** ({ratio:,.0f})!
                - Kalman potrebbe perdere vs MA
                """)
    
    st.sidebar.info(f"**Q={process_noise:.5f}, R={measurement_noise:.1f}**")
    
    # ===== MODULO 2.5: RISOLUZIONE VIDEO =====
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎬 Risoluzione Processing")
    
    resolution_mode = st.sidebar.selectbox(
        "Qualità Video",
        ["Alta Qualità (Originale) 🔥", 
         "Bilanciata (720p)", 
         "Performance (640p - veloce)"],
        index=1,  # Default: Bilanciata
        help="Risoluzione per processing. Originale = massima qualità ma più lento"
    )
    
    # Mappa selezione a width per process_video
    if "Originale" in resolution_mode:
        processing_width = None  # Mantieni risoluzione originale
        st.sidebar.info("💎 **Qualità massima** (più lento, specialmente con RAFT)")
    elif "720p" in resolution_mode:
        processing_width = 1280
        st.sidebar.success("⚖️ **Ottimo compromesso** qualità/performance")
    else:
        processing_width = 640
        st.sidebar.success("⚡ **Massima velocità** (qualità ridotta)")
    
    # ===== MODULO 3: COMPENSAZIONE E ZOOM =====
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Adaptive Zoom & Crop")
    
    enable_zoom = st.sidebar.checkbox(
        "Attiva Zoom Adattivo",
        value=True,
        help="Applica zoom per nascondere bordi neri"
    )
    
    zoom_percentage = st.sidebar.slider(
        "Percentuale Zoom",
        min_value=0,
        max_value=20,
        value=10,
        step=1,
        help="% di zoom (5-10% raccomandato)"
    ) if enable_zoom else 0
    
    zoom_factor = 1.0 + (zoom_percentage / 100.0)
    
    # ===== RIEPILOGO CONFIGURAZIONE =====
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Riepilogo Configurazione")
    
    with st.sidebar.expander("📊 Vedi Configurazione Completa", expanded=False):
        st.markdown(f"""
        **Motion Estimation:**
        - Algoritmo: `{motion_method}`
        """)
        
        if method_key == 'raft':
            st.markdown(f"""
            - Modello: `{'Small' if raft_use_small else 'Large'}`
            - Punti Campionamento: `{raft_num_samples}`
            """)
        else:  # sift
            st.markdown(f"""
            - Max Features: `{sift_n_features}`
            - Contrast Threshold: `{sift_contrast_threshold:.2f}`
            - Edge Threshold: `{sift_edge_threshold}`
            - Sigma: `{sift_sigma:.1f}`
            - Ratio Test: `{sift_ratio_threshold:.2f}`
            """)
        
        st.markdown(f"""
        **Trajectory Smoothing:**
        - Media Mobile Radius: `{smoothing_radius} frame`
        - Kalman Q: `{process_noise:.5f}`
        - Kalman R: `{measurement_noise:.1f}`
        - Rapporto R/Q: `{measurement_noise/process_noise:.0f}`
        
        **Risoluzione Processing:**
        - Width: `{'Originale' if processing_width is None else f'{processing_width}px'}`
        
        **Post-Processing:**
        - Zoom: `{zoom_percentage}%` ({'ON' if enable_zoom else 'OFF'})
        """)
    
    # ===== UPLOAD VIDEO =====
    st.sidebar.markdown("---")
    st.sidebar.header("📁 Carica Video")
    uploaded_file = st.sidebar.file_uploader(
        "Seleziona un video",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Carica un video instabile da stabilizzare"
    )
    
    if uploaded_file is not None:
        # Salva il file temporaneamente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Mostra informazioni video
        video_info = get_video_info(video_path)
        if video_info:
            st.sidebar.success(f"""
            **Info Video:**
            - FPS: {video_info['fps']:.2f}
            - Frame: {video_info['frame_count']}
            - Risoluzione: {video_info['width']}x{video_info['height']}
            """)
        
        # Inizializza session state
        if 'processed' not in st.session_state:
            st.session_state.processed = False
        if 'params' not in st.session_state:
            st.session_state.params = {}
        
        # Controlla se i parametri sono cambiati
        current_params = {
            'method': method_key,
            'radius': smoothing_radius,
            'process_noise': process_noise,
            'measurement_noise': measurement_noise,
            'zoom_factor': zoom_factor,
            'video_path': video_path,
            'processing_width': processing_width
        }
        
        # Aggiungi parametri specifici per algoritmo
        if method_key == 'raft':
            current_params.update({
                'raft_use_small': raft_use_small,
                'raft_num_samples': raft_num_samples
            })
        else:  # sift
            current_params.update({
                'sift_n_features': sift_n_features,
                'sift_contrast': sift_contrast_threshold,
                'sift_edge': sift_edge_threshold,
                'sift_sigma': sift_sigma,
                'sift_ratio': sift_ratio_threshold
            })
        
        params_changed = st.session_state.params != current_params
        
        # Bottone di processing
        if st.sidebar.button("🚀 Avvia Stabilizzazione", type="primary") or params_changed:
            st.session_state.params = current_params
            
            with st.spinner(f'🔄 Processing video con {motion_method}...'):
                # Step 1: Processa il video (estrai frame e movimento)
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Prepara parametri per process_video
                if method_key == 'raft':
                    sift_params = None
                    raft_params = {
                        'use_small': raft_use_small,
                        'num_samples': raft_num_samples
                    }
                else:  # sift
                    sift_params = {
                        'n_features': sift_n_features,
                        'contrast_threshold': sift_contrast_threshold,
                        'edge_threshold': sift_edge_threshold,
                        'sigma': sift_sigma,
                        'ratio_threshold': sift_ratio_threshold
                    }
                    raft_params = None
                
                # Inizia timing
                total_start_time = time.time()
                
                status_text.text(f"📹 Stima movimento con {motion_method}...")
                motion_start = time.time()
                frames, transforms = process_video(
                    video_path,
                    width=processing_width,  # Usa risoluzione configurabile
                    method=method_key,
                    sift_params=sift_params,
                    raft_params=raft_params
                )
                motion_time = time.time() - motion_start
                progress_bar.progress(25)
                
                # ✅ CALCOLA TRAJECTORY RAW UNA VOLTA SOLA (prima di qualsiasi smoothing)
                status_text.text("📐 Calcolo traiettoria cumulativa raw...")
                from trajectory_smoothing import compute_trajectory_from_transforms
                trajectory_raw = compute_trajectory_from_transforms(transforms)
                
                # Step 2: Stabilizza con Media Mobile
                status_text.text("📊 Stabilizzazione con Media Mobile...")
                ma_start = time.time()
                frames_ma, _, traj_ma = stabilize_video_moving_average(
                    frames, transforms, radius=smoothing_radius, zoom_factor=zoom_factor
                )
                ma_time = time.time() - ma_start
                progress_bar.progress(50)
                
                # Step 3: Stabilizza con Kalman
                status_text.text("🎯 Stabilizzazione con Filtro di Kalman...")
                kalman_start = time.time()
                frames_kalman, _, traj_kalman = stabilize_video_kalman(
                    frames, transforms, 
                    process_noise=process_noise,
                    measurement_noise=measurement_noise,
                    zoom_factor=zoom_factor
                )
                kalman_time = time.time() - kalman_start
                
                # ✅ UNIFORMA LE LUNGHEZZE (prendi il minimo comune)
                min_len = min(len(trajectory_raw), len(traj_ma), len(traj_kalman))
                trajectory_raw = trajectory_raw[:min_len]
                traj_ma = traj_ma[:min_len]
                traj_kalman = traj_kalman[:min_len]
                
                # ✅ VALIDA CHE ABBIANO LA STESSA SHAPE
                assert trajectory_raw.shape == traj_ma.shape == traj_kalman.shape, \
                    f"Shape mismatch! Raw: {trajectory_raw.shape}, MA: {traj_ma.shape}, Kalman: {traj_kalman.shape}"
                
                total_time = time.time() - total_start_time
                progress_bar.progress(75)
                
                progress_bar.progress(90)
                
                # Step 4: Crea video output
                status_text.text("💾 Creazione video stabilizzati...")
                
                # Salva i video stabilizzati
                fps = video_info['fps'] if video_info else 30
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='_ma.mp4') as tmp_ma:
                    video_ma_path = tmp_ma.name
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='_kalman.mp4') as tmp_kalman:
                    video_kalman_path = tmp_kalman.name
                
                # Crea i video
                frames_to_video(frames_ma, video_ma_path, fps)
                frames_to_video(frames_kalman, video_kalman_path, fps)
                
                
                # Leggi i video come bytes per Streamlit
                with open(video_ma_path, 'rb') as f:
                    video_ma_bytes = f.read()
                
                with open(video_kalman_path, 'rb') as f:
                    video_kalman_bytes = f.read()
                
                # Step 5: Calcola metriche dettagliate
                status_text.text("📊 Calcolo metriche quantitative...")
                
                # Prepara config per export
                config_export = {
                    'motion_estimation': {
                        'method': method_key,
                        'resolution_width': processing_width if processing_width else 'original'
                    },
                    'trajectory_smoothing': {
                        'moving_average_radius': smoothing_radius,
                        'kalman_process_noise': process_noise,
                        'kalman_measurement_noise': measurement_noise
                    },
                    'post_processing': {
                        'zoom_enabled': enable_zoom,
                        'zoom_percentage': zoom_percentage
                    }
                }
                
                if method_key == 'raft':
                    config_export['motion_estimation']['raft'] = {
                        'use_small_model': raft_use_small,
                        'num_samples': raft_num_samples
                    }
                else:
                    config_export['motion_estimation']['sift'] = {
                        'n_features': sift_n_features,
                        'contrast_threshold': sift_contrast_threshold,
                        'edge_threshold': sift_edge_threshold,
                        'sigma': sift_sigma,
                        'ratio_threshold': sift_ratio_threshold
                    }
                
                # Genera report metriche completi (USA LA STESSA trajectory_raw per entrambi!)
                metrics_ma = generate_metrics_report(
                    method_name='moving_average',
                    motion_estimation_method=method_key,
                    trajectory_raw=trajectory_raw,  # ← USA LA STESSA
                    trajectory_smoothed=traj_ma,
                    num_frames=len(frames),
                    processing_time=motion_time + ma_time,
                    config=config_export
                )
                
                metrics_kalman = generate_metrics_report(
                    method_name='kalman_filter',
                    motion_estimation_method=method_key,
                    trajectory_raw=trajectory_raw,  # ← USA LA STESSA
                    trajectory_smoothed=traj_kalman,
                    num_frames=len(frames),
                    processing_time=motion_time + kalman_time,
                    config=config_export
                )
                
                # Salva metriche come JSON temporanei
                with tempfile.NamedTemporaryFile(delete=False, suffix='_ma_metrics.json', mode='w') as tmp_ma_json:
                    json.dump(metrics_ma, tmp_ma_json, indent=2)
                    metrics_ma_path = tmp_ma_json.name
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='_kalman_metrics.json', mode='w') as tmp_kalman_json:
                    json.dump(metrics_kalman, tmp_kalman_json, indent=2)
                    metrics_kalman_path = tmp_kalman_json.name
                
                # Salva nei session state
                st.session_state.frames_original = frames
                st.session_state.frames_ma = frames_ma
                st.session_state.frames_kalman = frames_kalman
                st.session_state.video_path = video_path
                st.session_state.video_ma_bytes = video_ma_bytes
                st.session_state.video_kalman_bytes = video_kalman_bytes
                st.session_state.video_ma_path = video_ma_path  # Per download
                st.session_state.video_kalman_path = video_kalman_path  # Per download
                st.session_state.trajectory_orig = trajectory_raw  # ← USA trajectory_raw uniformata
                st.session_state.trajectory_ma = traj_ma
                st.session_state.trajectory_kalman = traj_kalman
                st.session_state.metrics_ma = metrics_ma
                st.session_state.metrics_kalman = metrics_kalman
                st.session_state.metrics_ma_path = metrics_ma_path
                st.session_state.metrics_kalman_path = metrics_kalman_path
                st.session_state.timing = {
                    'motion_estimation': motion_time,
                    'moving_average': ma_time,
                    'kalman': kalman_time,
                    'total': total_time
                }
                st.session_state.num_frames_used = min_len  # Per debug
                st.session_state.processed = True
                
                progress_bar.progress(100)
                status_text.text("✅ Processing completato!")
        
        # Mostra i risultati se processati
        if st.session_state.processed:
            st.markdown("---")
            st.header("📺 Confronto Video")
            
            # Leggi bytes video originale e codifica tutto in base64 per HTML sincronizzato
            with open(st.session_state.video_path, 'rb') as f:
                orig_b64 = base64.b64encode(f.read()).decode()
            ma_b64 = base64.b64encode(st.session_state.video_ma_bytes).decode()
            kalman_b64 = base64.b64encode(st.session_state.video_kalman_bytes).decode()

            sync_html = f"""
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{ background: transparent; }}

  .vgrid-wrapper {{
    font-family: "Source Sans Pro", "Segoe UI", sans-serif;
    background: transparent;
    padding: 0;
  }}

  /* Label sopra ogni video */
  .vgrid {{ display: flex; gap: 12px; }}
  .vgrid-item {{
    flex: 1;
    display: flex;
    flex-direction: column;
  }}
  .vgrid-item .vid-label {{
    color: rgba(250,250,250,0.9);
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 2px;
    letter-spacing: 0.01em;
  }}
  .vgrid-item .vid-sub {{
    color: rgba(250,250,250,0.4);
    font-size: 11px;
    margin-bottom: 6px;
  }}
  .vgrid-item video {{
    width: 100%;
    border-radius: 8px;
    background: #000;
    border: 1px solid rgba(250,250,250,0.08);
    display: block;
  }}

  /* Barra controlli — stile st.container con bordo sottile */
  .vctrl {{
    display: flex;
    align-items: center;
    gap: 12px;
    background: #262730;
    border: 1px solid rgba(250,250,250,0.08);
    border-radius: 8px;
    padding: 10px 16px;
    margin-top: 10px;
  }}

  /* Bottone — identico all'st.button primario di Streamlit */
  #playbtn {{
    background: #00CC66;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 5px 18px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    font-family: inherit;
    white-space: nowrap;
    transition: background 0.15s, box-shadow 0.15s;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
  }}
  #playbtn:hover {{ background: #00CC66; box-shadow: 0 2px 6px rgba(255,75,75,0.35); }}
  #playbtn:active {{ background: #00CC66; }}

  /* Slider — accent Streamlit */
  #vseek {{
    flex: 1;
    cursor: pointer;
    accent-color: #ff4b4b;
    height: 4px;
  }}

  /* Timestamp */
  #vtime {{
    color: rgba(250,250,250,0.45);
    font-size: 12px;
    min-width: 88px;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }}

  /* Caption sotto i controlli */
  .vcaption {{
    color: rgba(250,250,250,0.25);
    font-size: 11px;
    text-align: center;
    margin-top: 6px;
    letter-spacing: 0.02em;
  }}
</style>

<div class="vgrid-wrapper">
  <div class="vgrid">
    <div class="vgrid-item">
      <div class="vid-label">🔴 Originale</div>
      <div class="vid-sub">Video instabile</div>
      <video id="v0" preload="auto">
        <source src="data:video/mp4;base64,{orig_b64}" type="video/mp4">
      </video>
    </div>
    <div class="vgrid-item">
      <div class="vid-label">🔵 Media Mobile</div>
      <div class="vid-sub">Smoothing per finestra</div>
      <video id="v1" preload="auto">
        <source src="data:video/mp4;base64,{ma_b64}" type="video/mp4">
      </video>
    </div>
    <div class="vgrid-item">
      <div class="vid-label">🟢 Kalman Filter</div>
      <div class="vid-sub">Smoothing predittivo</div>
      <video id="v2" preload="auto">
        <source src="data:video/mp4;base64,{kalman_b64}" type="video/mp4">
      </video>
    </div>
  </div>

  <div class="vctrl">
    <button id="playbtn" onclick="togglePlay()">▶ Play</button>
    <input type="range" id="vseek" min="0" step="0.01" value="0" oninput="seekAll(this.value)">
    <span id="vtime">0:00 / 0:00</span>
  </div>
  <div class="vcaption">Riproduzione sincronizzata</div>
</div>

<script>
  const vids = [document.getElementById('v0'), document.getElementById('v1'), document.getElementById('v2')];
  const btn  = document.getElementById('playbtn');
  const seek = document.getElementById('vseek');
  const tdisp = document.getElementById('vtime');
  let syncing = false;

  function fmt(s) {{
    const m = Math.floor(s / 60);
    const ss = String(Math.floor(s % 60)).padStart(2, '0');
    return m + ':' + ss;
  }}

  function refreshMax() {{
    const dur = vids[0].duration;
    if (dur && isFinite(dur)) {{
      seek.max = dur;
      seek.step = dur / 1000;
    }}
  }}

  vids[0].addEventListener('loadedmetadata', refreshMax);
  vids[0].addEventListener('durationchange', refreshMax);
  vids[0].addEventListener('canplay', refreshMax);

  vids[0].addEventListener('timeupdate', () => {{
    if (syncing) return;
    refreshMax();
    seek.value = vids[0].currentTime;
    const dur = vids[0].duration;
    tdisp.textContent = fmt(vids[0].currentTime) + ' / ' + (isFinite(dur) ? fmt(dur) : '--:--');
  }});

  vids[0].addEventListener('ended', () => {{
    btn.textContent = '▶ Play';
  }});

  function togglePlay() {{
    if (vids[0].paused) {{
      vids.forEach(v => v.play());
      btn.textContent = '⏸ Pausa';
    }} else {{
      vids.forEach(v => v.pause());
      btn.textContent = '▶ Play';
    }}
  }}

  function seekAll(t) {{
    syncing = true;
    vids.forEach(v => {{ v.currentTime = parseFloat(t); }});
    tdisp.textContent = fmt(t) + ' / ' + fmt(vids[0].duration || 0);
    syncing = false;
  }}
</script>
"""
            components.html(sync_html, height=420)
            
            # Grafici delle traiettorie
            st.markdown("---")
            st.header("📈 Analisi Traiettorie")
            
            # Crea tabs per i diversi assi
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Confronto X/Y", "↔️ Asse X", "↕️ Asse Y", "🔄 Rotazione"])
            
            with tab1:
                st.markdown("""  
                
                La linea grigia tratteggiata mostra il movimento RAW stimato, mentre le altre linee mostrano come ogni algoritmo lo smootha.
                """)
                
                # Grafici matplotlib completi
                fig = plot_trajectory_comparison(
                    st.session_state.trajectory_orig,
                    st.session_state.trajectory_ma,
                    st.session_state.trajectory_kalman,
                )
                st.pyplot(fig)
            
            with tab2:
                st.subheader("Traiettoria Asse X (pixel)")
                
                # Uniforma lunghezze per evitare errori di dimensione
                min_len = len(st.session_state.trajectory_orig)
                # Prepara dati per Streamlit line chart
                import pandas as pd
                df_x = pd.DataFrame({
                    'Frame': range(min_len),
                    'RAW (Motion Est.)': st.session_state.trajectory_orig[:min_len, 0],
                    'Media Mobile': st.session_state.trajectory_ma[:min_len, 0],
                    'Kalman': st.session_state.trajectory_kalman[:min_len, 0]
                })
                st.line_chart(df_x.set_index('Frame'))
            
            with tab3:
                st.subheader("Traiettoria Asse Y (pixel)")
                
                # Uniforma lunghezze
                min_len = len(st.session_state.trajectory_orig)
                df_y = pd.DataFrame({
                    'Frame': range(min_len),
                    'RAW (Motion Est.)': st.session_state.trajectory_orig[:min_len, 1],
                    'Media Mobile': st.session_state.trajectory_ma[:min_len, 1],
                    'Kalman': st.session_state.trajectory_kalman[:min_len, 1]
                })
                st.line_chart(df_y.set_index('Frame'))
            
            with tab4:
                st.subheader("Rotazione (gradi)")
                
                # Uniforma lunghezze
                min_len = len(st.session_state.trajectory_orig)
                df_angle = pd.DataFrame({
                    'Frame': range(min_len),
                    'RAW (Motion Est.)': np.degrees(st.session_state.trajectory_orig[:min_len, 2]),
                    'Media Mobile': np.degrees(st.session_state.trajectory_ma[:min_len, 2]),
                    'Kalman': np.degrees(st.session_state.trajectory_kalman[:min_len, 2])
                })
                st.line_chart(df_angle.set_index('Frame'))
            
            # Metriche quantitative DETTAGLIATE
            st.markdown("---")
            st.header("📊 Metriche Quantitative Dettagliate")
            
            # Estrai metriche dai report
            metrics_ma_display = format_metrics_for_display(st.session_state.metrics_ma)
            metrics_kalman_display = format_metrics_for_display(st.session_state.metrics_kalman)
            
            # Tab per organizzare le metriche
            tab_overview, tab_comparison = st.tabs(["📈 Panoramica", "⚖️ Confronto"])
            
            with tab_overview:
                st.markdown("### Stability Score (0-100)")
                st.markdown("*Score globale basato su riduzione jitter pesata - misura la **fluidità** del movimento*")
                
                col_score1, col_score2 = st.columns(2)
                
                with col_score1:
                    score_ma = metrics_ma_display['stability_score']
                    st.metric(
                        "🔵 Media Mobile",
                        f"{score_ma:.1f} / 100",
                        help="Score basato su riduzione jitter X/Y (40% ciascuno) + rotazione (20%)"
                    )
                    # Progress bar visiva
                    st.progress(score_ma / 100.0)
                
                with col_score2:
                    score_kalman = metrics_kalman_display['stability_score']
                    st.metric(
                        "🟢 Kalman Filter",
                        f"{score_kalman:.1f} / 100",
                        help="Score basato su riduzione jitter X/Y (40% ciascuno) + rotazione (20%)"
                    )
                    st.progress(score_kalman / 100.0)
                
                st.markdown("---")
                
                # Fidelity Score (preservazione traiettoria)
                if metrics_ma_display['fidelity_score'] is not None:
                    st.markdown("### Fidelity Score (0-100)")
                    st.markdown("*Misura quanto la traiettoria smussata **preserva i movimenti intenzionali** della camera originale (basato su RMSE)*")
                    
                    col_fid1, col_fid2 = st.columns(2)
                    
                    with col_fid1:
                        fid_ma = metrics_ma_display['fidelity_score']
                        st.metric(
                            "🔵 Media Mobile",
                            f"{fid_ma:.1f} / 100",
                            help="Score alto = segue bene la traiettoria originale (preserva movimenti intenzionali)"
                        )
                        st.progress(fid_ma / 100.0)
                    
                    with col_fid2:
                        fid_kalman = metrics_kalman_display['fidelity_score']
                        st.metric(
                            "🟢 Kalman Filter",
                            f"{fid_kalman:.1f} / 100",
                            help="Score alto = segue bene la traiettoria originale (preserva movimenti intenzionali)"
                        )
                        st.progress(fid_kalman / 100.0)
                    
                    st.markdown("---")
                st.markdown("### Riduzione Jitter per Asse")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Asse X**")
                    st.metric("Media Mobile", f"{metrics_ma_display['jitter_reduction_x']:.1f}%")
                    st.metric("Kalman Filter", f"{metrics_kalman_display['jitter_reduction_x']:.1f}%")
                
                with col2:
                    st.markdown("**Asse Y**")
                    st.metric("Media Mobile", f"{metrics_ma_display['jitter_reduction_y']:.1f}%")
                    st.metric("Kalman Filter", f"{metrics_kalman_display['jitter_reduction_y']:.1f}%")
                
                with col3:
                    st.markdown("**Rotazione**")
                    st.metric("Media Mobile", f"{metrics_ma_display['jitter_reduction_angle']:.1f}%")
                    st.metric("Kalman Filter", f"{metrics_kalman_display['jitter_reduction_angle']:.1f}%")
            
            with tab_comparison:
                st.markdown("### Confronto Diretto")
                timing = st.session_state.timing
                
                # Costruisci lista metriche base
                metric_names = [
                    'Stability Score',
                ]
                ma_values = [f"{metrics_ma_display['stability_score']:.1f}"]
                kalman_values = [f"{metrics_kalman_display['stability_score']:.1f}"]
                
                # Aggiungi Fidelity Score se disponibile
                if metrics_ma_display['fidelity_score'] is not None:
                    metric_names.append('Fidelity Score')
                    ma_values.append(f"{metrics_ma_display['fidelity_score']:.1f}")
                    kalman_values.append(f"{metrics_kalman_display['fidelity_score']:.1f}")
                
                # Aggiungi altre metriche
                metric_names.extend([
                    'Jitter Reduction X (%)',
                    'Jitter Reduction Y (%)',
                    'Jitter Reduction Angle (%)',
                    'RMS X (px)',
                    'RMS Y (px)',
                    'RMS Angle (°)',
                    'Max Offset X (px)',
                    'Max Offset Y (px)',
                    'Processing Time (s)'
                ])
                ma_values.extend([
                    f"{metrics_ma_display['jitter_reduction_x']:.1f}",
                    f"{metrics_ma_display['jitter_reduction_y']:.1f}",
                    f"{metrics_ma_display['jitter_reduction_angle']:.1f}",
                    f"{metrics_ma_display['rms_x']:.2f}",
                    f"{metrics_ma_display['rms_y']:.2f}",
                    f"{metrics_ma_display['rms_angle_deg']:.3f}",
                    f"{metrics_ma_display['max_offset_x']:.1f}",
                    f"{metrics_ma_display['max_offset_y']:.1f}",
                    f"{timing['motion_estimation'] + timing['moving_average']:.2f}"
                ])
                kalman_values.extend([
                    f"{metrics_kalman_display['jitter_reduction_x']:.1f}",
                    f"{metrics_kalman_display['jitter_reduction_y']:.1f}",
                    f"{metrics_kalman_display['jitter_reduction_angle']:.1f}",
                    f"{metrics_kalman_display['rms_x']:.2f}",
                    f"{metrics_kalman_display['rms_y']:.2f}",
                    f"{metrics_kalman_display['rms_angle_deg']:.3f}",
                    f"{metrics_kalman_display['max_offset_x']:.1f}",
                    f"{metrics_kalman_display['max_offset_y']:.1f}",
                    f"{timing['motion_estimation'] + timing['kalman']:.2f}"
                ])
                
                # Tabella comparativa
                comparison_data = {
                    'Metrica': metric_names,
                    '🔵 Media Mobile': ma_values,
                    '🟢 Kalman Filter': kalman_values
                }
                
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, width="stretch", hide_index=True)
                
                st.markdown("---")
                st.markdown("### 💾 Export Metriche")
                st.markdown("Scarica i file JSON con tutte le metriche.")
                
                col_json1, col_json2 = st.columns(2)
                
                with col_json1:
                    with open(st.session_state.metrics_ma_path, 'r') as f:
                        st.download_button(
                            label="📥 Download Metriche Media Mobile (JSON)",
                            data=f.read(),
                            file_name="metrics_moving_average.json",
                            mime="application/json",
                            help="File JSON con tutte le metriche del metodo Media Mobile"
                        )
                
                with col_json2:
                    with open(st.session_state.metrics_kalman_path, 'r') as f:
                        st.download_button(
                            label="📥 Download Metriche Kalman (JSON)",
                            data=f.read(),
                            file_name="metrics_kalman_filter.json",
                            mime="application/json",
                            help="File JSON con tutte le metriche del Filtro di Kalman"
                        )
                
                # Export combinato
                combined_metrics = {
                    'experiment': {
                        'motion_estimation_method': st.session_state.params['method'],
                        'timestamp': st.session_state.metrics_ma['metadata']['timestamp'],
                        'configuration': st.session_state.metrics_ma['configuration']
                    },
                    'moving_average': st.session_state.metrics_ma['performance'],
                    'kalman_filter': st.session_state.metrics_kalman['performance'],
                    'timing': timing
                }
                
                st.download_button(
                    label="📥 Download Confronto Completo (JSON)",
                    data=json.dumps(combined_metrics, indent=2),
                    file_name="comparison_complete.json",
                    mime="application/json",
                    help="File JSON con confronto completo tra i due metodi"
                )

                st.markdown("---")
                st.markdown("### ⏱️ Timing")
                timing = st.session_state.timing
                col_time1, col_time2, col_time3, col_time4 = st.columns(4)
                with col_time1:
                    st.metric("Motion Estimation", f"{timing['motion_estimation']:.2f}s")
                with col_time2:
                    st.metric("Media Mobile", f"{timing['moving_average']:.2f}s")
                with col_time3:
                    st.metric("Kalman Filter", f"{timing['kalman']:.2f}s")
                with col_time4:
                    st.metric("Totale", f"{timing['total']:.2f}s")

            # Download dei video
            st.markdown("---")
            st.header("💾 Download Video Stabilizzati")
            
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                with open(st.session_state.video_ma_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Media Mobile",
                        data=f,
                        file_name="stabilized_moving_average.mp4",
                        mime="video/mp4"
                    )
            
            with col_dl2:
                with open(st.session_state.video_kalman_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Kalman Filter",
                        data=f,
                        file_name="stabilized_kalman.mp4",
                        mime="video/mp4"
                    )
    
    else:
        # Messaggio di benvenuto
        st.info("👈 Carica un video dalla sidebar per iniziare")
        
        st.markdown("""
        ### Come funziona?
        
        1. **Carica** un video instabile (es. registrato a mano)
        2. **Configura** i parametri nella sidebar
        3. **Avvia** la stabilizzazione
        4. **Confronta** i risultati tra i due metodi
        5. **Analizza** i grafici delle traiettorie
        """)


if __name__ == "__main__":
    main()
