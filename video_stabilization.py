"""
Video Stabilization Module
Modulo per applicare la stabilizzazione ai frame video.
Include gestione dello zoom adattivo per eliminare i bordi neri.
"""

import cv2
import numpy as np
from motion_estimation import estimate_motion_sift, estimate_motion_raft
from trajectory_smoothing import (
    compute_trajectory_from_transforms,
    moving_average_smooth,
    kalman_smooth,
    compute_smoothed_transforms
)


def resize_frame(frame, width=640):
    """
    Ridimensiona il frame mantenendo l'aspect ratio.
    Importante per performance real-time.
    
    Args:
        frame: Frame da ridimensionare
        width: Larghezza target in pixel (None = mantieni dimensione originale)
        
    Returns:
        Frame ridimensionato (o originale se width=None)
    """
    if frame is None:
        return None
    
    # Se width è None, ritorna il frame originale senza resize
    if width is None:
        return frame
    
    h, w = frame.shape[:2]
    aspect_ratio = h / w
    new_height = int(width * aspect_ratio)
    
    return cv2.resize(frame, (width, new_height), interpolation=cv2.INTER_AREA)


def apply_transform_with_zoom(frame, dx, dy, da, zoom_factor=1.0):
    """
    Applica trasformazione affine al frame con zoom opzionale.
    
    ZOOM ADATTIVO: Il tocco in più!
    Lo zoom elimina i bordi neri che appaiono dopo la stabilizzazione.
    Un leggero crop è preferibile rispetto ai bordi neri evidenti.
    
    Args:
        frame: Frame da trasformare
        dx: Correzione traslazione X
        dy: Correzione traslazione Y
        da: Correzione rotazione (radianti)
        zoom_factor: Fattore di zoom (>1.0 per zoomare, 1.0 per nessuno zoom)
        
    Returns:
        Frame trasformato e stabilizzato
    """
    h, w = frame.shape[:2]
    center = (w / 2, h / 2)
    
    # Crea matrice di rotazione con zoom integrato
    # Il zoom è applicato come scaling nella matrice
    rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(da), zoom_factor)
    
    # Aggiungi la traslazione alla matrice
    rotation_matrix[0, 2] += dx
    rotation_matrix[1, 2] += dy
    
    # Applica la trasformazione
    stabilized = cv2.warpAffine(
        frame,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    
    return stabilized


def process_video(video_path, width=640, method='sift',
                 sift_params=None, raft_params=None):
    """
    Processa il video estraendo i frame e stimando il movimento.
    
    Args:
        video_path: Percorso del file video
        width: Larghezza di ridimensionamento
        method: Metodo di motion estimation ('sift' o 'raft')
        sift_params: Dict con parametri SIFT (n_features, contrast_threshold, edge_threshold, sigma, ratio_threshold)
        raft_params: Dict con parametri RAFT (use_small, num_samples)
        
    Returns:
        Tuple (frames, transforms) dove:
        - frames: Lista di frame processati
        - transforms: Lista di tuple (dx, dy, da) per ogni frame
    """
    # Parametri di default
    if sift_params is None:
        sift_params = {
            'n_features': 500,
            'contrast_threshold': 0.04,
            'edge_threshold': 10,
            'sigma': 1.6,
            'ratio_threshold': 0.75
        }
    
    if raft_params is None:
        raft_params = {
            'use_small': False,  # False = RAFT Large (più accurato), True = RAFT Small (più veloce)
            'num_samples': 200   # Numero di punti da campionare dal flow denso
        }
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Impossibile aprire il video: {video_path}")
    
    frames = []
    transforms = []
    prev_gray = None
    prev_kp = None
    prev_desc = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Ridimensiona per performance (se width=None, mantieni originale)
        frame = resize_frame(frame, width)
        frames.append(frame)
        
        # Converti in scala di grigi
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is not None:
            # Stima il movimento tra frame consecutivi
            if method == 'sift':
                # Usa SIFT feature matching con parametri configurabili
                dx, dy, da, prev_kp, prev_desc = estimate_motion_sift(
                    prev_gray, gray, prev_kp, prev_desc,
                    **sift_params
                )
            elif method == 'raft':
                # Usa RAFT deep learning optical flow
                dx, dy, da = estimate_motion_raft(
                    prev_gray, gray,
                    **raft_params
                )
            transforms.append((dx, dy, da))
        else:
            # Primo frame: nessun movimento
            transforms.append((0.0, 0.0, 0.0))
        
        prev_gray = gray
    
    cap.release()
    return frames, transforms


def stabilize_video_moving_average(frames, transforms, radius=30, zoom_factor=1.0):
    """
    Stabilizza il video usando Media Mobile.
    
    Args:
        frames: Lista di frame originali
        transforms: Lista di trasformazioni (dx, dy, da)
        radius: Raggio della finestra mobile
        zoom_factor: Fattore di zoom per eliminare bordi neri
        
    Returns:
        Tuple (stabilized_frames, trajectory, smoothed_trajectory)
    """
    # Calcola traiettoria cumulativa
    trajectory = compute_trajectory_from_transforms(transforms)
    
    # Liscia con media mobile
    smoothed_trajectory = moving_average_smooth(trajectory, radius)
    
    # Calcola le correzioni
    corrections = compute_smoothed_transforms(trajectory, smoothed_trajectory)
    
    # Applica le correzioni ai frame
    # IMPORTANTE: Le correzioni vanno applicate con segno OPPOSTO
    # Se il contenuto si è spostato di -10 ma vogliamo -2, la correzione è -8
    # Ma dobbiamo spostare il frame di +8 per compensare!
    stabilized_frames = []
    for i, frame in enumerate(frames):
        dx, dy, da = corrections[i]
        stabilized = apply_transform_with_zoom(frame, -dx, -dy, -da, zoom_factor)
        stabilized_frames.append(stabilized)
    
    return stabilized_frames, trajectory, smoothed_trajectory


def stabilize_video_kalman(frames, transforms, process_noise=0.01, 
                          measurement_noise=1.0, zoom_factor=1.0):
    """
    Stabilizza il video usando il Filtro di Kalman.
    
    SUPERIORITÀ DEL KALMAN:
    - Nessuna latenza (predice invece di mediare)
    - Segue i panning intenzionali
    - Elimina solo il jitter ad alta frequenza
    
    Args:
        frames: Lista di frame originali
        transforms: Lista di trasformazioni (dx, dy, da)
        process_noise: Q - rumore del processo
        measurement_noise: R - rumore della misura
        zoom_factor: Fattore di zoom per eliminare bordi neri
        
    Returns:
        Tuple (stabilized_frames, trajectory, smoothed_trajectory)
    """
    # Calcola traiettoria cumulativa
    trajectory = compute_trajectory_from_transforms(transforms)
    
    # Liscia con Kalman filter
    smoothed_trajectory = kalman_smooth(trajectory, process_noise, measurement_noise)
    
    # Calcola le correzioni
    corrections = compute_smoothed_transforms(trajectory, smoothed_trajectory)
    
    # Applica le correzioni ai frame
    # IMPORTANTE: Le correzioni vanno applicate con segno OPPOSTO
    # Se il contenuto si è spostato di -10 ma vogliamo -2, la correzione è -8
    # Ma dobbiamo spostare il frame di +8 per compensare!
    stabilized_frames = []
    for i, frame in enumerate(frames):
        dx, dy, da = corrections[i]
        stabilized = apply_transform_with_zoom(frame, -dx, -dy, -da, zoom_factor)
        stabilized_frames.append(stabilized)
    
    return stabilized_frames, trajectory, smoothed_trajectory


def get_video_info(video_path):
    """
    Estrae informazioni sul video.
    
    Args:
        video_path: Percorso del file video
        
    Returns:
        Dict con informazioni del video (fps, frame_count, width, height)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }
    
    cap.release()
    return info
