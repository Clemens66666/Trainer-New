from datetime import datetime

history = """
# Patch-Historie – hybrid_longtrend_trainer.py

Alle Änderungen (chronologisch nach unserer Konversation) – geeignet zum Ablegen im Projektordner
===============================================================================================

## 1. Daten‑Handling & Splits
- **Gap im TimeSeriesSplit** auf `2 * seq_len` erhöht, um Überschneidungen zu vermeiden.
- Funktion `last_fold_with_enough_pos()` eingeführt → Val‑Fold garantiert ≥ 5 % positive Labels.
- `seq_len` jetzt Optuna‑Hyperparameter (`12…48`, Schritt 4).

## 2. Label‑Themen
- **Label‑Smoothing** (ε = 0.10) in `FTWrapped`.
- **pos_weight** automatisch aus Klassenverhältnis berechnet.
- **Focal‑Loss** (γ = 2.0) optional in `FTWrapped`.
- Wahrscheinlichkeiten vor LogLoss via `np.clip(..., 1e‑6, 1-1e‑6)` stabilisiert.

## 3. Architektur & Regularisierung
- `attention_dropout` & `ffn_dropout` auf **0.2** (vorher default 0.0).
- LoRA‑Adapter nur für letzte **2 Transformer‑Blöcke**, Rank **r = 4** (statt 8).
- Weight‑Decay **1e‑2** in allen `TrainingArguments`.
- 1‑D‑CNN als leichtes Basismodell.

## 4. Training‑Loop
- **EarlyStoppingCallback(patience = 2)** aktiviert  
  → benötigte Flags: `eval_strategy='epoch'`, `metric_for_best_model='eval_loss'`, `load_best_model_at_end=True`.
- Lernraten‑Suchraum verengt auf **1e‑5 … 3e‑4**.
- Mini‑Batch‑LogLoss (`_batch_log_loss`) statt komplettes Val‑Set im RAM.
- HF‑Checkpoint‑Ordner nach jedem Trial via `_cleanup_callback` gelöscht + GPU/CPU‑Cache freigegeben.

## 5. Hyperparameter‑Suche
- Optuna sucht jetzt zusätzlich `seq_len`.
- Callback‐basierte RAM‑Bereinigung integriert.

## 6. Fehler‑ & Bugfixes
- Import `IntervalStrategy` ergänzt.
- `focal_gamma`‑Tippfehler korrigiert.
- `evaluation_strategy` entfernt (legacy), stattdessen direkte Attribut‑Zuweisung.
- Diverse NameError‑Fixes (`study`, `trial`, …).
- `pin_memory=False`, `fp16=False` wenn keine GPU.

---

*Generiert am {ts}*
""".format(ts=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))

with open('/mnt/data/patch_history_hybrid_longtrend.txt', 'w', encoding='utf-8') as f:
    f.write(history)

history
