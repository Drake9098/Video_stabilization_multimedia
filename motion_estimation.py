"""
Motion Estimation Module
Modulo per la stima robusta del movimento globale tra frame consecutivi.
Utilizza feature tracking e RANSAC per ignorare oggetti in movimento.
"""

import cv2
import numpy as np


# ============================================================================
# SIFT-Based Motion Estimation
# ============================================================================

def extract_sift_features(frame_gray, n_features=500, contrast_threshold=0.04, 
                          edge_threshold=10, sigma=1.6):
    """
    Estrae keypoints e descrittori SIFT dal frame.
    SIFT è invariante a scala e rotazione, più robusto di corner detection.
    
    Args:
        frame_gray: Frame in scala di grigi
        n_features: Numero massimo di feature da estrarre
        contrast_threshold: Soglia contrasto (più alto = meno keypoints ma più robusti)
        edge_threshold: Soglia per filtrare edge responses
        sigma: Sigma per Gaussian blur
        
    Returns:
        Tuple (keypoints, descriptors)
    """
    # Inizializza SIFT detector con parametri configurabili
    sift = cv2.SIFT_create(
        nfeatures=n_features,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=sigma
    )
    
    # Rileva keypoints e calcola descrittori
    keypoints, descriptors = sift.detectAndCompute(frame_gray, None)
    
    return keypoints, descriptors


def match_sift_features(desc1, desc2, ratio_threshold=0.75):
    """
    Matcha i descrittori SIFT tra due frame usando Brute Force Matcher.
    Applica il Lowe's ratio test per filtrare match ambigui.
    
    Args:
        desc1: Descrittori del frame precedente
        desc2: Descrittori del frame corrente
        ratio_threshold: Soglia per Lowe's ratio test (0.75 = standard)
        
    Returns:
        Lista di good matches (DMatch objects)
    """
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []
    
    # Crea Brute Force Matcher (L2 norm per SIFT)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    # KNN matching (k=2 per Lowe's ratio test)
    try:
        matches = bf.knnMatch(desc1, desc2, k=2)
    except:
        return []
    
    # Applica Lowe's ratio test per filtrare match ambigui
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            # Se il miglior match è significativamente migliore del secondo
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    return good_matches


def estimate_motion_sift(prev_gray, curr_gray, prev_kp=None, prev_desc=None,
                        n_features=500, contrast_threshold=0.04, edge_threshold=10, 
                        sigma=1.6, ratio_threshold=0.75):
    """
    Stima il movimento globale usando SIFT feature matching.
    VANTAGGIO: Invariante a rotazione e scala, più robusto in scene complesse.
    
    Args:
        prev_gray: Frame precedente in scala di grigi
        curr_gray: Frame corrente in scala di grigi
        prev_kp: Keypoints del frame precedente (opzionali)
        prev_desc: Descrittori del frame precedente (opzionali)
        n_features: Numero massimo di features
        contrast_threshold: Soglia contrasto SIFT
        edge_threshold: Soglia edge SIFT
        sigma: Sigma Gaussian
        ratio_threshold: Soglia per Lowe's ratio test
        
    Returns:
        Tuple (dx, dy, da, new_kp, new_desc) - movimento stimato e feature per prossimo frame
    """
    # Estrai feature SIFT dal frame precedente (se non già fornite)
    if prev_kp is None or prev_desc is None:
        prev_kp, prev_desc = extract_sift_features(
            prev_gray, n_features, contrast_threshold, edge_threshold, sigma
        )
    
    # Estrai feature SIFT dal frame corrente
    curr_kp, curr_desc = extract_sift_features(
        curr_gray, n_features, contrast_threshold, edge_threshold, sigma
    )
    
    # Matcha le feature
    good_matches = match_sift_features(prev_desc, curr_desc, ratio_threshold)
    
    # Serve un minimo di match per stimare la trasformazione
    if len(good_matches) < 4:
        return 0.0, 0.0, 0.0, curr_kp, curr_desc
    
    # Estrai le coordinate dei punti matchati
    prev_pts = np.float32([prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    curr_pts = np.float32([curr_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Stima trasformazione affine con RANSAC
    # RANSAC elimina outlier (oggetti in movimento)
    transform, inliers = cv2.estimateAffinePartial2D(
        prev_pts, curr_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0
    )
    
    if transform is None:
        return 0.0, 0.0, 0.0, curr_kp, curr_desc
    
    # Estrai parametri di movimento
    dx = transform[0, 2]  # Traslazione X
    dy = transform[1, 2]  # Traslazione Y
    da = np.arctan2(transform[1, 0], transform[0, 0])  # Rotazione
    
    return dx, dy, da, curr_kp, curr_desc


# ============================================================================
# RAFT-Based Motion Estimation (Deep Learning)
# ============================================================================

# Cache del modello RAFT (caricato una sola volta)
_raft_model = None
_raft_device = None
_raft_transforms = None  # Trasformazioni ufficiali (normalizzazione corretta)


def get_raft_model(use_small=False):
    """
    Carica il modello RAFT pre-trainato (singleton per efficienza).
    
    Args:
        use_small: Se True usa raft_small (più veloce), altrimenti raft_large (più accurato)
        
    Returns:
        Tuple (model, device)
    """
    global _raft_model, _raft_device, _raft_transforms
    
    if _raft_model is None:
        try:
            import torch
            from torchvision.models.optical_flow import raft_large, raft_small
            from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
            
            # Determina device (GPU se disponibile, altrimenti CPU)
            _raft_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"🚀 RAFT: Caricamento modello su {_raft_device}...")
            
            # Carica modello con pesi pre-trainati
            if use_small:
                weights = Raft_Small_Weights.DEFAULT
                _raft_model = raft_small(weights=weights).to(_raft_device)
                print("✅ RAFT Small caricato (veloce)")
            else:
                weights = Raft_Large_Weights.DEFAULT
                _raft_model = raft_large(weights=weights).to(_raft_device)
                print("✅ RAFT Large caricato (accurato)")
            
            # Cache trasformazioni ufficiali: si aspettano uint8 [0,255] e
            # normalizzano internamente a [-1, 1] come richiesto dal modello.
            _raft_transforms = weights.transforms()
            
            _raft_model.eval()  # Modalità inferenza
            
        except ImportError:
            raise ImportError(
                "RAFT richiede PyTorch e torchvision >= 0.15.0\n"
                "Installa con: pip install torch torchvision"
            )
    
    return _raft_model, _raft_device, _raft_transforms


def preprocess_frame_for_raft(frame_gray):
    """
    Prepara un frame per RAFT restituendo un tensore uint8 [0, 255] (C, H, W).
    La normalizzazione vera e propria viene applicata dai transforms ufficiali
    in estimate_motion_raft, che convertono a float e normalizzano a [-1, 1].
    
    Args:
        frame_gray: Frame in scala di grigi (H, W)
        
    Returns:
        Tensor PyTorch uint8 (C, H, W) — senza batch dimension, senza /255
    """
    import torch
    
    # Converti grayscale a RGB (RAFT si aspetta 3 canali)
    frame_rgb = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)
    
    # (H, W, C) -> (C, H, W), uint8 [0, 255]
    # NON dividere per 255: i transforms ufficiali se lo aspettano intero
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1)  # uint8
    
    return frame_tensor


def sample_flow_points(flow, num_samples=200):
    """
    Campiona punti rappresentativi dall'optical flow denso.
    Usa una griglia uniforme per avere una distribuzione spaziale bilanciata.
    
    Args:
        flow: Optical flow denso (H, W, 2) - componenti (u, v)
        num_samples: Numero di punti da campionare
        
    Returns:
        Tuple (prev_pts, curr_pts) - coordinate nel frame precedente e corrente
    """
    h, w = flow.shape[:2]
    
    # Crea griglia uniforme di punti
    grid_size = int(np.sqrt(num_samples))
    y_indices = np.linspace(0, h - 1, grid_size, dtype=np.int32)
    x_indices = np.linspace(0, w - 1, grid_size, dtype=np.int32)
    
    prev_pts = []
    curr_pts = []
    
    for y in y_indices:
        for x in x_indices:
            # Punto nel frame precedente
            prev_pt = [x, y]
            
            # Punto corrispondente nel frame corrente (prev + flow)
            flow_x = flow[y, x, 0]
            flow_y = flow[y, x, 1]
            curr_pt = [x + flow_x, y + flow_y]
            
            prev_pts.append(prev_pt)
            curr_pts.append(curr_pt)
    
    return np.array(prev_pts, dtype=np.float32), np.array(curr_pts, dtype=np.float32)


def estimate_motion_raft(prev_gray, curr_gray, use_small=False, num_samples=200):
    """
    Stima il movimento globale usando RAFT (Deep Learning Optical Flow).
    
    VANTAGGI RISPETTO A LUCAS-KANADE:
    - Optical flow denso (tutti i pixel, non solo corner)
    - Gestisce grandi spostamenti senza piramidi
    - Robusto a cambi di illuminazione e occlusioni
    - State-of-the-art accuracy
    
    Args:
        prev_gray: Frame precedente in scala di grigi
        curr_gray: Frame corrente in scala di grigi
        use_small: Se True usa RAFT Small (più veloce), altrimenti Large (più accurato)
        num_samples: Numero di punti da campionare dal flow denso per stimare la trasformazione
        
    Returns:
        Tuple (dx, dy, da) - movimento stimato (traslazione X, Y, rotazione)
    """
    import torch
    
    # Carica modello RAFT (singleton) + transforms ufficiali
    model, device, raft_transforms = get_raft_model(use_small=use_small)
    
    # Prepara i frame come tensori uint8 (C, H, W)
    prev_tensor = preprocess_frame_for_raft(prev_gray)
    curr_tensor = preprocess_frame_for_raft(curr_gray)
    
    # Applica i transforms ufficiali: uint8 [0,255] -> float normalizzato [-1,1]
    # (si aspettano due tensori e restituiscono una coppia batch-ready)
    prev_batch, curr_batch = raft_transforms(prev_tensor, curr_tensor)
    prev_batch = prev_batch.unsqueeze(0).to(device)
    curr_batch = curr_batch.unsqueeze(0).to(device)
    
    # Inferenza RAFT (no gradients per efficienza)
    with torch.no_grad():
        # RAFT restituisce una lista di flow predetti (iterazioni coarse->fine)
        flow_predictions = model(prev_batch, curr_batch)
        # Ultima predizione = la più accurata
        flow_tensor = flow_predictions[-1]
    
    # Converti flow da tensor a numpy: (1, 2, H, W) -> (H, W, 2)
    flow = flow_tensor[0].permute(1, 2, 0).cpu().numpy()
    
    # --- Estrazione moto globale con mediana sul flow denso ---
    # La mediana è intrinsecamente robusta agli outlier (oggetti in movimento
    # in foreground) senza dover campionare punti né usare RANSAC.
    dx = float(np.median(flow[..., 0]))
    dy = float(np.median(flow[..., 1]))
    
    # Stima rotazione con RANSAC su campione dal flow (mediana non dà da)
    prev_pts, curr_pts = sample_flow_points(flow, num_samples=num_samples)
    da = 0.0
    if len(prev_pts) >= 4:
        transform, _ = cv2.estimateAffinePartial2D(
            prev_pts.reshape(-1, 1, 2),
            curr_pts.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0
        )
        if transform is not None:
            da = float(np.arctan2(transform[1, 0], transform[0, 0]))
    
    return dx, dy, da
