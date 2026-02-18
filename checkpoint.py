"""
Sistema di checkpoint/resume per ricerche lunghe.

Ogni coppia/grado testata viene salvata su disco. Se il programma
viene interrotto, riparte dal punto in cui si era fermato.
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any


class CheckpointManager:
    """Gestisce checkpoint per la ricerca profonda."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = checkpoint_dir / "search_state.json"
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except (json.JSONDecodeError, OSError):
                return {"completed": {}, "in_progress": None, "started_at": None}
        return {"completed": {}, "in_progress": None, "started_at": None}

    def _save_state(self):
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2))
        tmp.replace(self.state_file)

    def is_completed(self, pair_name: str, degree: int, max_coeff: int) -> bool:
        """Controlla se questa ricerca è già stata completata."""
        key = f"{pair_name}_d{degree}_c{max_coeff}"
        return key in self.state["completed"]

    def mark_started(self, pair_name: str, degree: int, max_coeff: int):
        """Segna l'inizio di una ricerca."""
        key = f"{pair_name}_d{degree}_c{max_coeff}"
        self.state["in_progress"] = key
        self.state["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._save_state()

    def mark_completed(
        self,
        pair_name: str,
        degree: int,
        max_coeff: int,
        result: dict,
    ):
        """Segna il completamento di una ricerca."""
        key = f"{pair_name}_d{degree}_c{max_coeff}"
        self.state["completed"][key] = {
            "result": result,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.state["in_progress"] = None
        self._save_state()

    def get_completed_count(self) -> int:
        return len(self.state["completed"])

    def get_max_completed_degree(self, pair_name: str, max_coeff: int) -> int:
        """Restituisce il grado massimo completato per una coppia/coeff."""
        max_deg = 0
        prefix = f"{pair_name}_d"
        suffix = f"_c{max_coeff}"
        for key in self.state["completed"]:
            if key.startswith(prefix) and key.endswith(suffix):
                deg_str = key[len(prefix):-len(suffix)]
                try:
                    deg = int(deg_str)
                    max_deg = max(max_deg, deg)
                except ValueError:
                    pass
        return max_deg
