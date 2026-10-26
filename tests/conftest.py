"""
conftest.py — Configurazione pytest per la cartella tests/.

Aggiunge automaticamente la root del progetto a sys.path in modo che
il pacchetto `paperhunter` sia importabile da tutti i file di test,
indipendentemente da come viene invocato pytest.
"""
import sys
from pathlib import Path

# Root del progetto (cartella padre di tests/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
