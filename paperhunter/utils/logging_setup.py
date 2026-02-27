"""
Logging professionale con Rich + tqdm per PaperHunter v2.

Fornisce:
- RichHandler come handler principale (output formattato con colori, timestamp, livello)
- Progress bar helper per operazioni batch (Rich progress)
- Tabelle riassuntive per statistiche (Rich table)
- Livelli configurabili da config.yaml

Uso:
    from paperhunter.utils.logging_setup import get_logger, progress_bar, print_table

    logger = get_logger(__name__)
    logger.info("Ingestion avviata")

    # Progress bar per batch
    for item in progress_bar(items, description="Processing PDFs"):
        process(item)

    # Tabella riassuntiva
    print_table("Statistiche", {"Processati": 100, "Errori": 2})
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Console globale Rich (riusata ovunque)
console = Console(stderr=True)

# Flag per evitare doppia inizializzazione
_initialized = False


def setup_logging(
    level: str = "INFO",
    rich_enabled: bool = True,
    log_file: str = "",
) -> None:
    """Configura il logging globale dell'applicazione.

    Args:
        level: Livello di log (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        rich_enabled: Se True, usa RichHandler per output formattato.
        log_file: Percorso al file di log (stringa vuota = nessun file).
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Rimuovi handler esistenti
    root_logger.handlers.clear()

    if rich_enabled:
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            show_level=True,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root_logger.addHandler(handler)

    # File handler opzionale
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        file_handler.setLevel(logging.DEBUG)  # File: tutto
        root_logger.addHandler(file_handler)

    # Silenzia logger rumorosi
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Ottieni un logger configurato per il modulo indicato.

    Se il logging non è ancora stato inizializzato, lo inizializza con i
    valori di default dal config.yaml.

    Args:
        name: Nome del modulo (tipicamente __name__).

    Returns:
        Logger configurato.
    """
    if not _initialized:
        try:
            from paperhunter.config import settings
            setup_logging(
                level=settings.logging.level,
                rich_enabled=settings.logging.rich_enabled,
                log_file=settings.logging.log_file,
            )
        except Exception:
            setup_logging()  # Fallback con defaults

    return logging.getLogger(name)


def progress_bar(
    iterable: Iterable,
    description: str = "Processing",
    total: Optional[int] = None,
) -> Iterable:
    """Wrapper per progress bar Rich su un iterabile.

    Args:
        iterable: L'iterabile su cui iterare.
        description: Descrizione mostrata accanto alla barra.
        total: Numero totale di elementi (auto-rilevato se possibile).

    Returns:
        Iterabile wrappato con progress bar.
    """
    try:
        from paperhunter.config import settings
        show = settings.logging.show_progress_bars
    except Exception:
        show = True

    if not show:
        yield from iterable
        return

    if total is None and hasattr(iterable, "__len__"):
        total = len(iterable)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task(description, total=total)
        for item in iterable:
            yield item
            progress.advance(task_id)


@contextmanager
def progress_context(description: str = "Processing", total: Optional[int] = None) -> Iterator:
    """Context manager per progress bar manuale (quando serve update esplicito).

    Uso:
        with progress_context("Ingesting PDFs", total=100) as (progress, task_id):
            for pdf in pdfs:
                process(pdf)
                progress.advance(task_id)

    Args:
        description: Descrizione della barra.
        total: Numero totale di step.

    Yields:
        Tupla (Progress, task_id) per controllo manuale.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task(description, total=total)
        yield progress, task_id


def print_table(
    title: str,
    data: dict[str, Any] | list[dict[str, Any]],
    columns: Optional[Sequence[str]] = None,
) -> None:
    """Stampa una tabella formattata con Rich.

    Args:
        title: Titolo della tabella.
        data: Dizionario {chiave: valore} per tabella a 2 colonne,
              oppure lista di dizionari per tabella multi-colonna.
        columns: Nomi delle colonne (auto-rilevati se non specificati).
    """
    table = Table(title=title, show_header=True, header_style="bold magenta")

    if isinstance(data, dict):
        table.add_column("Campo", style="cyan", no_wrap=True)
        table.add_column("Valore", style="green")
        for key, value in data.items():
            table.add_row(str(key), str(value))
    elif isinstance(data, list) and data:
        cols = columns or list(data[0].keys())
        for col in cols:
            table.add_column(col, style="cyan")
        for row in data:
            table.add_row(*[str(row.get(c, "")) for c in cols])
    else:
        table.add_column("Info")
        table.add_row("Nessun dato")

    console.print(table)


def print_panel(title: str, content: str, style: str = "bold green") -> None:
    """Stampa un pannello formattato con Rich.

    Args:
        title: Titolo del pannello.
        content: Contenuto testuale.
        style: Stile Rich del bordo.
    """
    console.print(Panel(content, title=title, border_style=style))


def print_success(message: str) -> None:
    """Stampa un messaggio di successo."""
    console.print(f"[bold green]✓[/bold green] {message}")


def print_error(message: str) -> None:
    """Stampa un messaggio di errore."""
    console.print(f"[bold red]✗[/bold red] {message}")


def print_warning(message: str) -> None:
    """Stampa un messaggio di warning."""
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")
