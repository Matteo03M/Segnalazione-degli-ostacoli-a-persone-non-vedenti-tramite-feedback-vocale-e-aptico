"""Analizzatore spaziale e calcolo del Time-to-Collision
Risolve i problemi del codice precedente (coordinate box ai bordi e calcolo TTC)"""

from typing import Dict, Optional, Tuple
from collections import deque
from models import Detection


class SpatialAnalyzer:
    def __init__(self, frame_width: float = 640.0, frame_height: float = 480.0):
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Storico delle tracce per calcolare velocita' e TTC nel tempo
        self.track_history: Dict[int, deque] = {}

        # Soglie metriche
        self.CRITICAL_DISTANCE_M = 1.0   # Sotto 1m: arresto immediato
        self.WARNING_DISTANCE_M = 2.5    # Sotto 2,5m: pre-allarme deviazione
        self.CRITICAL_TTC_S = 1.5        # Sotto 1,5 secondi all'impatto: emergenza assoluta

    def get_direction(self, detection: Detection) -> str:
        """Divide l'inquadratura in 3 zone orizzontali:
        - Sinistra: primo terzo del frame (0% - 33%)
        - Centro: terzo centrale (33% - 66%) -> zona di massimo pericolo cammino
        - Destra: ultimo terzo (66% - 100%)"""
        cx = detection.center_x
        one_third = self.frame_width / 3.0
        two_thirds = 2.0 * one_third

        if cx < one_third:
            return "sinistra"
        elif cx <= two_thirds:
            return "centro"
        else:
            return "destra"

    def is_touching_borders(self, detection: Detection) -> bool:
        """Verifica se il box tocca i bordi dell'immagine (errore del codice precedente che scartava il box,
        noi diciamo che invece e' molto vicino e occupa gran parte della visuale)"""
        margin = 5.0  # margine di tolleranza di 5 pixel (cuscinetto che se ad esempio il mio bb si trova a 5 pixel dal margine per me e' come se gia' mi stesse addosso)
        x1, y1, x2, y2 = detection.bbox_xyxy
        return (
            x1 <= margin or 
            y1 <= margin or 
            x2 >= (self.frame_width - margin) or 
            y2 >= (self.frame_height - margin)
        )

    def update_history(self, detection: Detection, ts_ms: int):
        """Aggiorna lo storico temporale dell'oggetto per calcolare la traiettoria."""
        if detection.track_id is None:
            return

        tid = detection.track_id
        if tid not in self.track_history:
            self.track_history[tid] = deque(maxlen=10)

        self.track_history[tid].append((ts_ms, detection.depth_m, detection.area))

    def estimate_ttc(self, detection: Detection, ts_ms: int) -> Optional[float]:
        """Calcola il Time-to-Collision (TTC) in secondi
        - Se c'e' la distanza metrica (depth_m), usa la derivata temporale della distanza: TTC = d / v_avvicinamento.
        - Se manca la profondita', usa l'espansione dell'area del box (metodo di approccio trovato grazie all'analisi dei paper)"""
        tid = detection.track_id
        if tid is None or tid not in self.track_history:
            return None

        history = self.track_history[tid]
        if len(history) < 3:
            return None  # Servono almeno 3 frame per una stima stabile

        # Primo metodo: Derivata temporale della profondita' metrica
        curr_ts, curr_depth, curr_area = history[-1]
        prev_ts, prev_depth, prev_area = history[0]
        dt_seconds = (curr_ts - prev_ts) / 1000.0

        if dt_seconds <= 0:
            return None

        if curr_depth is not None and prev_depth is not None:
            delta_depth = prev_depth - curr_depth  # Positivo se si sta avvicinando
            if delta_depth > 0.05:  # Si avvicina ad almeno 5 cm
                approach_speed = delta_depth / dt_seconds  # m/s
                ttc = curr_depth / (approach_speed + 1e-6)
                return max(0.0, round(ttc, 2))

        # Secondo metodo: Espansione dell'area del riquadro
        if curr_area > prev_area * 1.10:  # L'area e' cresciuta di oltre il 10%
            area_ratio = (curr_area / (prev_area + 1e-6)) ** 0.5
            if area_ratio > 1.0:
                ttc_visual = dt_seconds / (area_ratio - 1.0)
                return max(0.0, round(ttc_visual, 2))

        return None

    def assess_urgency(self, detection: Detection, ttc: Optional[float]) -> str:
        """Determina il livello di urgenza dell'ostacolo:
        - CRITICA: sotto 1m o impatto in meno di 1,5 secondi
        - ALTA: tra 1m e 2,5m in rapido avvicinamento
        - MEDIA: ostacolo fermo entro 2,5m
        - BASSA: ostacolo oltre 2,5m o in allontanamento"""
        dist = detection.depth_m

        # Priorita' 1: Emergenza collisione immediata
        if dist is not None and dist <= self.CRITICAL_DISTANCE_M:
            return "CRITICA"
        if ttc is not None and ttc <= self.CRITICAL_TTC_S:
            return "CRITICA"

        # Se l'oggetto e' enorme e tocca i bordi, e' critico (mi sta praticamente addosso)
        if self.is_touching_borders(detection) and detection.area > (self.frame_width * self.frame_height * 0.35):
            return "CRITICA"

        # Priorita' 2: Pre-allarme in avvicinamento
        if dist is not None and dist <= self.WARNING_DISTANCE_M:
            if ttc is not None and ttc <= 3.0:
                return "ALTA"
            return "MEDIA"

        if ttc is not None and ttc <= 4.0:
            return "ALTA"

        return "BASSA"