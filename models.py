"""Definisce le strutture dati, qui dentro risolvo anche il bug del codice precedente"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Detection:
    """Singolo ostacolo rilevato in un fotogramma."""
    bbox_xyxy: List[float]       # Coordinate [x1, y1, x2, y2] del riquadro
    cls: str                     # Nome della classe (es. 'person', 'pole', 'car')
    score: float                 # Livello di confidenza (tra 0 e 1)
    track_id: Optional[int] = None   # ID univoco della traccia
    depth_m: Optional[float] = None  # Distanza stimata in metri

    @property
    def center_x(self) -> float:
        """Coordinata X del centro del riquadro."""
        return (self.bbox_xyxy[0] + self.bbox_xyxy[2]) / 2.0

    @property
    def center_y(self) -> float:
        """Coordinata Y del centro del riquadro."""
        return (self.bbox_xyxy[1] + self.bbox_xyxy[3]) / 2.0

    @property
    def width(self) -> float:
        """Larghezza del riquadro in pixel."""
        return max(0.0, self.bbox_xyxy[2] - self.bbox_xyxy[0])

    @property
    def height(self) -> float:
        """Altezza del riquadro in pixel."""
        return max(0.0, self.bbox_xyxy[3] - self.bbox_xyxy[1])

    @property
    def area(self) -> float:
        """Area del riquadro in pixel quadrati."""
        return self.width * self.height


@dataclass
class FrameData:
    """Rappresenta un singolo frame del flusso temporale"""
    frame_idx: int
    ts_ms: int
    detections: List[Detection] = field(default_factory=list)
    schema: str = "boss.perception/0.1"
    source_id: str = "simulazione"


@dataclass
class AlertEvent:
    """Avviso generato per avvisare il non vedente."""
    track_id: Optional[int]
    cls: str                     # es. 'persona', 'palo', 'gradino'
    direction: str               # 'sinistra', 'centro', 'destra'
    distance_m: Optional[float]  # es. 1.5 metri
    urgency: str                 # 'BASSA', 'MEDIA', 'ALTA', 'CRITICA'
    speech_text: str             # Frase emessa dalla voce
    timestamp_ms: int            # Istante in cui e' generato l'allarme