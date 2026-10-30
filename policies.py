"""Implementa le politiche di segnalazione con il Pattern Strategy"""

from abc import ABC, abstractmethod
from typing import List
from models import FrameData, AlertEvent
from spatial_analyzer import SpatialAnalyzer


class SignalingPolicy(ABC):
    """Interfaccia base per il Pattern Strategy: ogni politica deve avere il metodo evaluate (che valuta se avvisare o meno il non vedente)"""
    
    @abstractmethod
    def evaluate(self, frame: FrameData, analyzer: SpatialAnalyzer) -> List[AlertEvent]:
        pass


class DistanceThresholdPolicy(SignalingPolicy):
    """Politica a Soglia di Distanza:
    - Se l'ostacolo e' entro la soglia (default 2.0 metri) o in rotta critica: genera l'avviso vocale
    - Se l'ostacolo e' lontano (oltre 2.0 metri) non dice nulla per evitare il sovraccarico cognitivo"""
    def __init__(self, distance_threshold_m: float = 2.0):
        self.distance_threshold_m = distance_threshold_m

    def evaluate(self, frame: FrameData, analyzer: SpatialAnalyzer) -> List[AlertEvent]:
        alerts = []

        for det in frame.detections:
            # 1. Aggiorna lo storico e calcola TTC e direzione
            analyzer.update_history(det, frame.ts_ms)
            direction = analyzer.get_direction(det)
            ttc = analyzer.estimate_ttc(det, frame.ts_ms)
            urgency = analyzer.assess_urgency(det, ttc)

            # 2. Verifica se l'ostacolo supera la soglia di allerta
            should_alert = False
            dist = det.depth_m

            # Caso A: c'e' la distanza ed e' sotto la soglia di sicurezza
            if dist is not None and dist <= self.distance_threshold_m:
                should_alert = True

            # Caso B: urgenza CRITICA (es. TTC basso o tocca i bordi)
            if urgency == "CRITICA":
                should_alert = True

            # 3. Se l'ostacolo e' pericoloso, genera l'evento di allarme
            if should_alert:
                # Costruisce la frase vocale telegrafica: oggetto + direzione + distanza
                dist_str = f"{dist:.1f} metri" if dist is not None else "vicino"
                speech_text = f"{det.cls}, {direction}, {dist_str}"

                alert = AlertEvent(
                    track_id=det.track_id,
                    cls=det.cls,
                    direction=direction,
                    distance_m=dist,
                    urgency=urgency,
                    speech_text=speech_text,
                    timestamp_ms=frame.ts_ms
                )
                alerts.append(alert)

        return alerts