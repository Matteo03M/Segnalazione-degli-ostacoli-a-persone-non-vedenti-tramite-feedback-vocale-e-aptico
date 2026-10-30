"""Gestione dei canali di output con il Pattern Observer"""

from abc import ABC, abstractmethod
import pyttsx3
from models import AlertEvent


class AlertObserver(ABC):
    """Interfaccia base per il Pattern Observer: il sistema che ascolta gli allarmi deve avere il metodo update."""
    @abstractmethod
    def update(self, alert: AlertEvent):
        pass


class VoiceOutputChannel(AlertObserver):
    """Canale di Uscita Vocale:Riceve l'allarme e lo pronuncia a voce alta tramite pyttsx3"""
    def __init__(self, speech_rate: int = 160):
        self.speech_rate = speech_rate

    def update(self, alert: AlertEvent):
        # 1. Stampa a video l'avviso per il log
        print(f"🔊 [VOCE]: \"{alert.speech_text}\" (Urgenza: {alert.urgency})")

        # Voce
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', self.speech_rate)
            engine.say(alert.speech_text)
            engine.runAndWait()
            del engine
        except Exception as e:
            print(f"Errore audio: {e}")