import json
import os
import time
from models import FrameData, Detection
from spatial_analyzer import SpatialAnalyzer
from policies import DistanceThresholdPolicy
from outputs import VoiceOutputChannel


def run_simulation(scenario_path: str = "scenario_test.jsonl"):
    print("=" * 60)
    print("SIMULAZIONE POLITICA DI SEGNALAZIONE")
    print("=" * 60)

    #Verifica che il file dello scenario esista
    if not os.path.exists(scenario_path):
        print(f"❌ Errore: file {scenario_path} non trovato!")
        return

    #Inizializzazione dei moduli
    print("[1] Inizializzazione analizzatore spaziale...")
    analyzer = SpatialAnalyzer(frame_width=640.0, frame_height=480.0)

    print("[2] Inizializzazione politica: DistanceThresholdPolicy (soglia <= 2.0 m)...")
    policy = DistanceThresholdPolicy(distance_threshold_m=2.0)

    print("[3] Inizializzazione canale vocale (pyttsx3)...")
    voice_channel = VoiceOutputChannel()

    print(f"[4] Caricamento scenario: {scenario_path}")
    print("-" * 60)

    total_frames = 0
    total_alerts = 0

    #Lettura del file JSON Lines riga per riga (un frame alla volta)
    with open(scenario_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parsing del JSON del frame corrente
            raw_frame = json.loads(line)
            total_frames += 1

            # Conversione dei dati grezzi negli oggetti del nostro modello
            detections = []
            for d in raw_frame.get("detections", []):
                det = Detection(
                    bbox_xyxy=d["bbox_xyxy"],
                    cls=d["cls"],
                    score=float(d["score"]),
                    track_id=d.get("track_id"),
                    depth_m=float(d["depth_m"]) if d.get("depth_m") is not None else None
                )
                detections.append(det)

            frame = FrameData(
                frame_idx=raw_frame["frame_idx"],
                ts_ms=raw_frame["ts_ms"],
                source_id=raw_frame.get("source_id", "simulazione"),
                detections=detections
            )

            #Log informativo del frame
            print(f"\n[FRAME {frame.frame_idx} | t = {frame.ts_ms} ms] Ostacoli rilevati: {len(frame.detections)}")
            for det in frame.detections:
                dist_info = f"{det.depth_m:.1f}m" if det.depth_m else "N/D"
                print(f"   -> ID {det.track_id}: '{det.cls}' a distanza {dist_info} (Confidenza: {det.score:.0%})")

            #Valutazione della politica decisionale (Pattern Strategy)
            alerts = policy.evaluate(frame, analyzer)

            #Se ci sono allarmi, notifica il canale vocale (Pattern Observer)
            if alerts:
                for alert in alerts:
                    total_alerts += 1
                    voice_channel.update(alert)
            else:
                print("   (Nessun allarme emesso: ostacolo fuori soglia di sicurezza)")

            # Piccola pausa simulata tra i frame per simulare il passaggio del tempo
            time.sleep(0.3)

    # 4. Riepilogo finale
    print("\n" + "=" * 60)
    print(f"SIMULAZIONE COMPLETATA!")
    print(f"Fotogrammi analizzati: {total_frames}")
    print(f"Allarmi vocali generati: {total_alerts}")
    print("-" * 60)


if __name__ == "__main__":
    run_simulation()