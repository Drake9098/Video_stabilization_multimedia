"""
Trajectory Smoothing Module
Modulo per il lisciamento delle traiettorie usando Media Mobile e Filtro di Kalman.
Dimostra la superiorità del Kalman Filter rispetto alla Moving Average.
"""

import numpy as np
import cv2


def moving_average_smooth(trajectory, radius=30):
    """
    Applica Media Mobile alla traiettoria.
    PROBLEMA: Introduce latenza (ritardo) perché guarda sia avanti che indietro.
    
    Args:
        trajectory: Array numpy (n_frames, 3) con [x, y, angle]
        radius: Raggio della finestra mobile (più grande = più smooth ma più latenza)
        
    Returns:
        Array numpy con traiettoria lisciata
    """
    smoothed = np.copy(trajectory)
    
    for i in range(3):  # Per ogni componente (x, y, angle)
        # Padding per gestire i bordi
        padded = np.pad(trajectory[:, i], (radius, radius), mode='edge')
        
        # Applica convoluzione con finestra uniforme
        kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
        smoothed[:, i] = np.convolve(padded, kernel, mode='valid')
    
    return smoothed


class KalmanTrajectoryFilter:
    """
    Filtro di Kalman avanzato per traiettorie video con modello a 6 stati.
    
    MODELLO CINEMATICO:
    Stati: [x, y, θ, vx, vy, vθ] - posizione + velocità per ogni grado di libertà
    
    EQUAZIONI:
    x(k) = x(k-1) + vx(k-1) * dt
    y(k) = y(k-1) + vy(k-1) * dt
    θ(k) = θ(k-1) + vθ(k-1) * dt
    vx(k) = vx(k-1)
    vy(k) = vy(k-1)
    vθ(k) = vθ(k-1)
    
    VANTAGGIO: Predice la posizione futura, quindi è reattivo e senza latenza.
    """
    
    def __init__(self, process_noise=0.01, measurement_noise=1.0, dt=1.0):
        """
        Inizializza un Filtro di Kalman unificato con 6 stati.
        
        Args:
            process_noise: Q - quanto rumore nel modello di movimento (basso = assume moto fluido)
            measurement_noise: R - quanto rumore nelle misure (alto = ignora jitter)
            dt: Time step (normalmente 1 frame = 1.0)
        """
        # 6 stati (x, y, θ, vx, vy, vθ), 3 misure (x, y, θ)
        self.kf = cv2.KalmanFilter(6, 3, 0)
        self.dt = dt
        
        # Matrice di transizione (constant velocity model)
        # [ x_new  ]   [ 1  0  0  dt  0  0  ] [ x  ]
        # [ y_new  ]   [ 0  1  0  0  dt  0  ] [ y  ]
        # [ θ_new  ] = [ 0  0  1  0  0  dt  ] [ θ  ]
        # [ vx_new ]   [ 0  0  0  1  0  0   ] [ vx ]
        # [ vy_new ]   [ 0  0  0  0  1  0   ] [ vy ]
        # [ vθ_new ]   [ 0  0  0  0  0  1   ] [ vθ ]
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, dt, 0,  0],   # x = x + vx*dt
            [0, 1, 0, 0,  dt, 0],   # y = y + vy*dt
            [0, 0, 1, 0,  0,  dt],  # θ = θ + vθ*dt
            [0, 0, 0, 1,  0,  0],   # vx = vx
            [0, 0, 0, 0,  1,  0],   # vy = vy
            [0, 0, 0, 0,  0,  1]    # vθ = vθ
        ], dtype=np.float32)
        
        # Matrice di misura: osserviamo solo posizione (x, y, θ)
        # [ z_x ]   [ 1  0  0  0  0  0 ] [ x  ]
        # [ z_y ] = [ 0  1  0  0  0  0 ] [ y  ]
        # [ z_θ ]   [ 0  0  1  0  0  0 ] [ θ  ]
        #                                 [ vx ]
        #                                 [ vy ]
        #                                 [ vθ ]
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance Q (6x6)
        # Quanto ci fidiamo del modello (basso = moto fluido)
        # Le velocità hanno un po' più rumore delle posizioni, ma non troppo
        q_pos = process_noise  # Rumore sulla posizione
        q_vel = process_noise * 2.0  # Rumore sulle velocità (era 10x, ora 2x per più smoothing)
        
        self.kf.processNoiseCov = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]).astype(np.float32)
        
        # Measurement noise covariance R (3x3)
        # Quanto rumore c'è nelle misure (alto = ignora jitter ad alta frequenza)
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * measurement_noise
        
        # Stima iniziale dell'errore (6x6)
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        
        # Stato iniziale [x, y, θ, vx, vy, vθ] = [0, 0, 0, 0, 0, 0]
        self.kf.statePost = np.zeros((6, 1), dtype=np.float32)
        # Stato iniziale [x, y, θ, vx, vy, vθ] = [0, 0, 0, 0, 0, 0]
        self.kf.statePost = np.zeros((6, 1), dtype=np.float32)
    
    def predict(self):
        """
        Predice lo stato futuro (step di predizione del Kalman).
        
        Returns:
            Array [x, y, θ] predetto
        """
        pred = self.kf.predict()
        # Ritorna solo posizione (x, y, θ), non le velocità
        return np.array([pred[0, 0], pred[1, 0], pred[2, 0]])
    
    def correct(self, measurement):
        """
        Corregge la predizione con la misura reale (step di correzione del Kalman).
        
        Args:
            measurement: Array [x, y, θ] misurato
            
        Returns:
            Array [x, y, θ] corretto
        """
        # Converti measurement in formato OpenCV (3x1 matrix)
        meas = np.array([[measurement[0]], [measurement[1]], [measurement[2]]], dtype=np.float32)
        
        # Corregge lo stato
        corr = self.kf.correct(meas)
        
        # Ritorna solo posizione (x, y, θ)
        return np.array([corr[0, 0], corr[1, 0], corr[2, 0]])
    
    def update(self, measurement):
        """
        Esegue un ciclo completo: predict + correct.
        
        Args:
            measurement: Array [x, y, θ] misurato
            
        Returns:
            Array [x, y, θ] filtrato
        """
        self.predict()
        return self.correct(measurement)
    
    def initialize(self, initial_state):
        """
        Inizializza lo stato del filtro con valori specifici.
        
        Args:
            initial_state: Array [x, y, θ] iniziale
        """
        self.kf.statePost = np.array([
            [initial_state[0]],  # x
            [initial_state[1]],  # y
            [initial_state[2]],  # θ
            [0.0],               # vx
            [0.0],               # vy
            [0.0]                # vθ
        ], dtype=np.float32)


def kalman_smooth(trajectory, process_noise=0.01, measurement_noise=1.0):
    """
    Applica il Filtro di Kalman all'intera traiettoria.
    VANTAGGIO: No latenza, segue il panning ma elimina il jitter.
    
    Args:
        trajectory: Array numpy (n_frames, 3) con [x, y, angle]
        process_noise: Q - rumore del processo (più basso = più smooth)
        measurement_noise: R - rumore della misura (più alto = più filtering)
        
    Returns:
        Array numpy con traiettoria lisciata
    """
    n_frames = len(trajectory)
    smoothed = np.zeros_like(trajectory)
    
    # Inizializza il filtro con modello a 6 stati
    kf = KalmanTrajectoryFilter(process_noise, measurement_noise)
    
    # Inizializza lo stato con la prima misura
    kf.initialize(trajectory[0])
    
    # Processa ogni frame
    for i in range(n_frames):
        measurement = trajectory[i]
        smoothed[i] = kf.update(measurement)
    
    return smoothed


def compute_trajectory_from_transforms(transforms):
    """
    Converte i delta di movimento (dx, dy, da) in traiettoria assoluta cumulativa.
    
    Args:
        transforms: Lista di tuple (dx, dy, da)
        
    Returns:
        Array numpy (n_frames, 3) con traiettoria cumulativa [x, y, angle]
    """
    trajectory = np.zeros((len(transforms), 3))
    
    for i, (dx, dy, da) in enumerate(transforms):
        if i == 0:
            trajectory[i] = [dx, dy, da]
        else:
            trajectory[i] = trajectory[i - 1] + [dx, dy, da]
    
    return trajectory


def compute_smoothed_transforms(trajectory, smoothed_trajectory):
    """
    Calcola le trasformazioni necessarie per correggere il movimento.
    
    Args:
        trajectory: Traiettoria originale (n_frames, 3)
        smoothed_trajectory: Traiettoria lisciata (n_frames, 3)
        
    Returns:
        Array numpy (n_frames, 3) con le correzioni [dx, dy, da]
    """
    # IMPORTANTE: trajectory - smoothed ci dà la CORREZIONE da applicare
    # Se camera va a +10 e smooth dice +2, la correzione è +8 per compensare
    difference = trajectory - smoothed_trajectory
    
    return difference
