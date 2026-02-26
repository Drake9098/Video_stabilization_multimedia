"""
Metrics Computation Module
Modulo per il calcolo di metriche quantitative per la video stabilization.
Permette confronti oggettivi tra diversi algoritmi e configurazioni.
"""

import numpy as np
import json
from datetime import datetime


def compute_rms_displacement(trajectory):
    """
    Calcola Root Mean Square (RMS) displacement per X, Y e angolo.
    
    RMS misura l'ampiezza media dei movimenti incrementali.
    Valori bassi = movimenti piccoli/fluidi, valori alti = movimenti bruschi.
    
    Args:
        trajectory: Array numpy (n_frames, 3) con [x, y, angle]
        
    Returns:
        Tuple (rms_x, rms_y, rms_angle) in pixel e radianti
    """
    if len(trajectory) < 2:
        return 0.0, 0.0, 0.0
    
    # Calcola spostamento incrementale tra frame consecutivi
    motion_incremental = np.diff(trajectory, axis=0)
    
    # RMS = sqrt(mean(squares))
    rms_x = float(np.sqrt(np.mean(motion_incremental[:, 0]**2)))
    rms_y = float(np.sqrt(np.mean(motion_incremental[:, 1]**2)))
    rms_angle = float(np.sqrt(np.mean(motion_incremental[:, 2]**2)))
    
    return rms_x, rms_y, rms_angle


def compute_jitter_reduction(trajectory_raw, trajectory_smoothed):
    """
    Calcola la riduzione percentuale del jitter su ogni asse.
    
    JITTER = varianza degli spostamenti incrementali (shake ad alta frequenza)
    RIDUZIONE = (1 - var_smoothed / var_raw) * 100%
    
    Valori alti (>80%) = ottima riduzione jitter
    Valori negativi = algoritmo ha peggiorato la stabilità (raro)
    
    Args:
        trajectory_raw: Traiettoria originale (instabile)
        trajectory_smoothed: Traiettoria dopo smoothing
        
    Returns:
        Tuple (reduction_x, reduction_y, reduction_angle) in percentuale
    """
    # Uniforma lunghezze (potrebbero differire leggermente)
    min_len = min(len(trajectory_raw), len(trajectory_smoothed))
    traj_raw = trajectory_raw[:min_len]
    traj_smooth = trajectory_smoothed[:min_len]
    
    if len(traj_raw) < 3:
        return 0.0, 0.0, 0.0
    
    # Calcola varianza del moto incrementale (jitter)
    raw_motion = np.diff(traj_raw, axis=0)
    smooth_motion = np.diff(traj_smooth, axis=0)
    
    var_raw = np.var(raw_motion, axis=0)
    var_smooth = np.var(smooth_motion, axis=0)
    
    # Calcola riduzione percentuale per ogni asse
    reduction = np.zeros(3)
    for i in range(3):
        if var_raw[i] > 1e-10:  # Evita divisione per zero
            reduction[i] = (1.0 - var_smooth[i] / var_raw[i]) * 100.0
        else:
            reduction[i] = 0.0
    
    return float(reduction[0]), float(reduction[1]), float(reduction[2])


def compute_max_offset(trajectory_raw, trajectory_smoothed):
    """
    Calcola l'offset massimo applicato per la stabilizzazione.
    
    Indica quanto il frame è stato spostato/ruotato per compensare il movimento.
    Valori alti = video molto instabile o smoothing molto aggressivo.
    
    Args:
        trajectory_raw: Traiettoria originale
        trajectory_smoothed: Traiettoria smoothata
        
    Returns:
        Tuple (max_x, max_y, max_angle) offset massimi assoluti
    """
    min_len = min(len(trajectory_raw), len(trajectory_smoothed))
    
    # Calcola differenza (offset applicato)
    offset = trajectory_smoothed[:min_len] - trajectory_raw[:min_len]
    
    # Trova massimo assoluto per ogni asse
    max_x = float(np.max(np.abs(offset[:, 0])))
    max_y = float(np.max(np.abs(offset[:, 1])))
    max_angle = float(np.max(np.abs(offset[:, 2])))
    
    return max_x, max_y, max_angle


def compute_mean_displacement(trajectory):
    """
    Calcola spostamento medio per frame.
    
    Args:
        trajectory: Array numpy (n_frames, 3)
        
    Returns:
        Tuple (mean_dx, mean_dy, mean_dangle)
    """
    if len(trajectory) < 2:
        return 0.0, 0.0, 0.0
    
    motion = np.diff(trajectory, axis=0)
    mean_dx = float(np.mean(np.abs(motion[:, 0])))
    mean_dy = float(np.mean(np.abs(motion[:, 1])))
    mean_dangle = float(np.mean(np.abs(motion[:, 2])))
    
    return mean_dx, mean_dy, mean_dangle


def compute_stability_score(jitter_reduction_x, jitter_reduction_y, jitter_reduction_angle):
    """
    Calcola uno score globale di stabilità [0-100].
    
    Score = media pesata delle riduzioni jitter (più peso su X/Y che su rotazione).
    
    Args:
        jitter_reduction_x: Riduzione jitter asse X (%)
        jitter_reduction_y: Riduzione jitter asse Y (%)
        jitter_reduction_angle: Riduzione jitter rotazione (%)
        
    Returns:
        Score di stabilità [0-100]
    """
    # Peso maggiore su traslazioni (più visibili) che rotazione
    weights = np.array([0.4, 0.4, 0.2])
    reductions = np.array([jitter_reduction_x, jitter_reduction_y, jitter_reduction_angle])
    
    # Clamp a [0, 100] (valori negativi = 0)
    reductions = np.clip(reductions, 0, 100)
    
    score = float(np.dot(weights, reductions))
    return score


def compute_fidelity_score(trajectory_raw, trajectory_smoothed):
    """
    Calcola score di fedeltà alla traiettoria originale [0-100].

    Misura quanto il metodo di smoothing preserva i movimenti intenzionali.
    Score alto = segue bene la traiettoria raw (Kalman tipicamente vince)
    Score basso = over-smoothing, la curva ignora le oscillazioni (Media Mobile)

    Formula: R² nello spazio incrementale (differenze frame-to-frame).
    Lavorare sugli incrementi elimina il trend/panning dalla normalizzazione.

        R²_xy    = 1 - Var(Δraw_xy - Δsmooth_xy) / Var(Δraw_xy)
        R²_angle = 1 - Var(Δraw_a  - Δsmooth_a)  / Var(Δraw_a)

    - MA rimuove tutte le oscillazioni → Δsmooth ≈ 0 → differenza ≈ Δraw → R² ≈ 0
    - Kalman segue le oscillazioni     → differenza piccola             → R² alto

    Non ha parametri arbitrari: la normalizzazione è la varianza del raw stesso.

    Args:
        trajectory_raw: Traiettoria originale (N, 3) — [x_px, y_px, angle_rad]
        trajectory_smoothed: Traiettoria smoothata (N, 3)

    Returns:
        Score di fidelity [0-100]
    """
    min_len = min(len(trajectory_raw), len(trajectory_smoothed))
    raw = trajectory_raw[:min_len]
    smooth = trajectory_smoothed[:min_len]

    raw_incr = np.diff(raw, axis=0)
    smooth_incr = np.diff(smooth, axis=0)
    diff_incr = raw_incr - smooth_incr

    var_raw_xy = np.var(raw_incr[:, :2]) + 1e-12
    var_raw_deg = np.var(np.degrees(raw_incr[:, 2])) + 1e-12

    r2_xy = max(0.0, 1.0 - np.var(diff_incr[:, :2]) / var_raw_xy)
    r2_angle = max(0.0, 1.0 - np.var(np.degrees(diff_incr[:, 2])) / var_raw_deg)

    # Media pesata: traslazione 60%, rotazione 40%
    r2_combined = 0.6 * r2_xy + 0.4 * r2_angle

    # Scala con radice quadrata per espandere i valori medi verso l'alto:
    # R²=0.34 → sqrt → 0.58 → 58 punti (più intuitivo da leggere)
    # R²=0.07 → sqrt → 0.26 → 26 punti
    fidelity_score = 100.0 * np.sqrt(r2_combined)

    return float(fidelity_score)



def generate_metrics_report(
    method_name,
    motion_estimation_method,
    trajectory_raw,
    trajectory_smoothed,
    num_frames,
    processing_time=None,
    config=None
):
    """
    Genera un report completo delle metriche in formato dict.
    
    STRUTTURA REPORT:
    - metadata: timestamp, metodo, configurazione
    - raw_stats: statistiche traiettoria originale
    - smoothed_stats: statistiche dopo smoothing
    - performance: RMS, jitter reduction, score
    - trajectories: dati raw per plotting (opzionale)
    
    Args:
        method_name: Nome metodo smoothing ("moving_average" o "kalman")
        motion_estimation_method: Metodo motion estimation ("sift", "raft", ecc.)
        trajectory_raw: Traiettoria originale
        trajectory_smoothed: Traiettoria smoothata
        num_frames: Numero totale di frame processati
        processing_time: Tempo elaborazione in secondi (opzionale)
        config: Dict con configurazione parametri (opzionale)
        
    Returns:
        Dict con tutte le metriche
    """
    # Calcola tutte le metriche
    rms_raw_x, rms_raw_y, rms_raw_angle = compute_rms_displacement(trajectory_raw)
    rms_smooth_x, rms_smooth_y, rms_smooth_angle = compute_rms_displacement(trajectory_smoothed)
    
    jitter_x, jitter_y, jitter_angle = compute_jitter_reduction(trajectory_raw, trajectory_smoothed)
    
    max_offset_x, max_offset_y, max_offset_angle = compute_max_offset(trajectory_raw, trajectory_smoothed)
    
    mean_dx_raw, mean_dy_raw, mean_da_raw = compute_mean_displacement(trajectory_raw)
    mean_dx_smooth, mean_dy_smooth, mean_da_smooth = compute_mean_displacement(trajectory_smoothed)
    
    # Calcola scores
    stability_score = compute_stability_score(jitter_x, jitter_y, jitter_angle)
    fidelity_score = compute_fidelity_score(trajectory_raw, trajectory_smoothed)
    
    # Costruisci report
    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'smoothing_method': method_name,
            'motion_estimation_method': motion_estimation_method,
            'num_frames': int(num_frames),
            'processing_time_seconds': float(processing_time) if processing_time else None,
        },
        
        'performance': {
            'stability_score': round(stability_score, 2),
            'fidelity_score': round(fidelity_score, 2),
            
            'rms_displacement': {
                'raw': {
                    'x_px': round(rms_raw_x, 3),
                    'y_px': round(rms_raw_y, 3),
                    'angle_rad': round(rms_raw_angle, 5),
                    'angle_deg': round(np.degrees(rms_raw_angle), 3)
                },
                'smoothed': {
                    'x_px': round(rms_smooth_x, 3),
                    'y_px': round(rms_smooth_y, 3),
                    'angle_rad': round(rms_smooth_angle, 5),
                    'angle_deg': round(np.degrees(rms_smooth_angle), 3)
                }
            },
            
            'jitter_reduction_percent': {
                'x': round(jitter_x, 2),
                'y': round(jitter_y, 2),
                'angle': round(jitter_angle, 2)
            },
            
            'mean_displacement_per_frame': {
                'raw': {
                    'x_px': round(mean_dx_raw, 3),
                    'y_px': round(mean_dy_raw, 3),
                    'angle_deg': round(np.degrees(mean_da_raw), 3)
                },
                'smoothed': {
                    'x_px': round(mean_dx_smooth, 3),
                    'y_px': round(mean_dy_smooth, 3),
                    'angle_deg': round(np.degrees(mean_da_smooth), 3)
                }
            },
            
            'max_compensation_offset': {
                'x_px': round(max_offset_x, 2),
                'y_px': round(max_offset_y, 2),
                'angle_deg': round(np.degrees(max_offset_angle), 3)
            }
        },
        
        'configuration': config if config else {}
    }
    
    return report


def save_metrics_to_json(metrics, output_path):
    """
    Salva le metriche in un file JSON.
    
    Args:
        metrics: Dict con metriche (da generate_metrics_report)
        output_path: Path del file JSON di output
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def format_metrics_for_display(metrics):
    """
    Formatta le metriche in un dict semplificato per display UI.
    
    Args:
        metrics: Dict completo da generate_metrics_report
        
    Returns:
        Dict con metriche formattate per display
    """
    perf = metrics['performance']
    
    return {
        'stability_score': perf['stability_score'],
        'fidelity_score': perf.get('fidelity_score', None),
        'jitter_reduction_x': perf['jitter_reduction_percent']['x'],
        'jitter_reduction_y': perf['jitter_reduction_percent']['y'],
        'jitter_reduction_angle': perf['jitter_reduction_percent']['angle'],
        'rms_x': perf['rms_displacement']['smoothed']['x_px'],
        'rms_y': perf['rms_displacement']['smoothed']['y_px'],
        'rms_angle_deg': perf['rms_displacement']['smoothed']['angle_deg'],
        'max_offset_x': perf['max_compensation_offset']['x_px'],
        'max_offset_y': perf['max_compensation_offset']['y_px'],
    }


def compare_methods(metrics_list):
    """
    Confronta metriche di più metodi e identifica il migliore.
    
    Args:
        metrics_list: Lista di dict metriche da generate_metrics_report
        
    Returns:
        Dict con confronto e classifiche
    """
    if not metrics_list:
        return {}
    
    comparison = {
        'methods': [],
        'best_method': None,
        'rankings': {}
    }
    
    # Estrai score per ogni metodo
    scores = []
    for metrics in metrics_list:
        method_name = metrics['metadata']['smoothing_method']
        score = metrics['performance']['stability_score']
        comparison['methods'].append({
            'name': method_name,
            'score': score,
            'jitter_reduction_avg': np.mean([
                metrics['performance']['jitter_reduction_percent']['x'],
                metrics['performance']['jitter_reduction_percent']['y']
            ])
        })
        scores.append(score)
    
    # Identifica migliore
    best_idx = np.argmax(scores)
    comparison['best_method'] = comparison['methods'][best_idx]['name']
    
    return comparison
