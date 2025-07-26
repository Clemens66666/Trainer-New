# train.py  –  zentraler Launcher für alle Trainer
import argparse
import importlib
import torch               #  ← diese Zeile ergänzen
from pathlib import Path

import argparse, importlib, sys, torch
from pathlib import Path

def main() -> None:
    # 0) Argument‑Parser
    p = argparse.ArgumentParser(
        description="Trainings‑Launcher: entry | exit | longtrend | hybrid"
    )
    p.add_argument("--cfg", required=True, help="Pfad zur YAML‑Config")
    p.add_argument("--type", required=True,
                   choices=["entry", "exit", "longtrend", "hybrid"])
    p.add_argument("--pretrain", action="store_true",
                   help="Führe vorher Self‑Supervised Pretraining aus")
    p.add_argument("--ssl-weights", default="ft_ssl.pt",
                   help="Pfad zur SSL‑Gewichtsdatei")
    p.add_argument("--run-name", default=None,
                   help="Optionaler Name für den run‑Ordner")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # 1) Optional: Self‑Supervised Pretraining
    if args.pretrain and not Path(args.ssl_weights).exists():
        import pretrain
        print("▶  Starte Self‑Supervised Pretraining …")
        pretrain.main()                     # erzeugt ft_ssl.pt
        print(f"✅  SSL‑Weights gespeichert → {args.ssl_weights}")

    # 2) Dynamische Trainer‑Klasse
    trainer_map = {
        "entry":     "entry_trainer.EntryTrainer",
        "exit":      "exit_trainer.ExitTrainer",
        "longtrend": "longtrend_trainer.LongTrendTrainer",
        "hybrid":    "hybrid_longtrend_trainer.HybridLongTrendTrainer",
    }
    module_name, class_name = trainer_map[args.type].split(".")
    TrainerCls = getattr(
        importlib.import_module(f"trainers.{module_name}"),
        class_name
    )

    # 3) Reproduzierbarkeit
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 4) Trainer instanziieren & starten
    cfg_path = Path(args.cfg).expanduser().resolve()
    if not cfg_path.exists():
        sys.exit(f"❌  Config {cfg_path} nicht gefunden")

    print(f"▶  Starte {trainer_map[args.type]} mit {cfg_path}")
    trainer = TrainerCls(cfg_path)          # nur cfg_path benötigt
    trainer.ssl_weights = args.ssl_weights  # falls Klasse das nutzt
    if args.run_name:
        trainer.run_name = args.run_name

    trainer.run()
    print("✅  Training abgeschlossen.")

if __name__ == "__main__":
    main()

