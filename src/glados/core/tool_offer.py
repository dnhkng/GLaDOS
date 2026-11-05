"""Which tools a turn is offered: agent tool groups by keyword, and hints read from tool results.

Agent tools cost ~800 prompt tokens, so each group is offered only when the request sounds like it.
"""

from __future__ import annotations

import json
import re
from typing import Any
import unicodedata

INTERNET_SEARCH_TOOL = "mcp.internet_search.web_search_exa"
SYSTEM_KEYWORDS = (
    "system",
    "status",
    "cpu",
    "memory",
    "ram",
    "disk",
    "storage",
    "network",
    "ip",
    "uptime",
    "temperature",
    "temp",
    "process",
    "battery",
    "power",
    "load",
)

# The owner asking GLaDOS to change herself (accent-free, lower case): "améliore-toi", "modifie ton code"...
SELF_IMPROVE_REQUEST = re.compile(
    r"\b(amelior|modifi|chang|corrig|reprogramm|optimis|mets? a jour)(e|er|es|ez)?[- ](toi|ton|ta|tes)\b"
    r"|\bauto-?amelior|\bself[-_ ]?improv|\bimprove (yourself|your (own )?code)"
    r"|\b(chang|pass|bascul)(e|er|es|ez)? (de|sur|vers|le|ton|ta) (modele|model|llm|voix)"
)

AGENT_TOOL_GROUPS = {
    "run_shell": "shell",
    "delegate_task": "shell",
    "agent_notes": "any",
    "read_file": "files",
    "write_file": "files",
    "edit_file": "files",
    "list_dir": "files",
    "desktop": "desktop",
    "schedule": "schedule",
    "browser": "web",
    "self_improve": "self",
}
# Accent-free, lower-case stems in French and English; "any" offers every agent tool.
AGENT_TOOL_KEYWORDS = {
    "any": ("agent", "tache de fond", "en arriere-plan", "background", "delegue", "delegate", "automatis"),
    "shell": (
        "command",
        "lance",
        "execute",
        "run ",
        "terminal",
        "shell",
        "bash",
        "script",
        "install",
        "mets a jour",
        "update",
        "git",
        "compile",
        "build",
        "processus",
        "process",
        "paquet",
        "package",
        "service",
        "systemctl",
        "redemarre",
        "restart",
        "ping",
        "reseau",
        "network",
        "telecharge",
        "download",
        "projet",
        "project",
        "code",
        "programme",
        "cherche",
        "trouve",
        "find",
        "search",
        "combien",
        "how many",
        "tache",
        "task",
    ),
    "files": (
        "fichier",
        "file",
        "dossier",
        "folder",
        "repertoire",
        "directory",
        "lis ",
        "lire",
        "read",
        "ecris",
        "write",
        "modifie",
        "edit",
        "cree",
        "create",
        "supprime",
        "delete",
        "deplace",
        "move",
        "copie",
        "copy",
        "renomme",
        "rename",
        "note",
        "document",
        "projet",
        "project",
        "config",
        "~/",
        "/home",
        ".txt",
        ".md",
        ".py",
        ".json",
        ".yaml",
        "liste",
        "list",
    ),
    "self": (
        "ameliore-toi",
        "ameliore toi",
        "ton code",
        "ton propre code",
        "ton modele",
        "tes modeles",
        "change de modele",
        "modifie-toi",
        "modifie toi",
        "toi-meme",
        "toi meme",
        "auto-amelior",
        "self-improv",
        "your code",
        "your model",
        "improve yourself",
        "proposition",
        "proposal",
        "applique",
        "apply",
        "ta config",
        "ta configuration",
        "ton fonctionnement",
        "nouvelle fonction",
        "fonctionnalite",
        "feature",
    ),
    "web": (
        "site",
        "page",
        "web",
        "internet",
        "navigu",
        "browse",
        "va sur",
        "go to",
        "url",
        "http",
        "www",
        ".com",
        ".fr",
        ".org",
        "formulaire",
        "form",
        "connecte-toi",
        "log in",
        "wikipedia",
        "youtube",
        "google",
        "recherche",
        "cherche",
        "search",
        "lien",
        "link",
        "navigateur",
        "browser",
        "commande",
        "achete",
        "reserve",
    ),
    "schedule": (
        "rappel",
        "remind",
        "planifi",
        "programme",
        "schedule",
        "timer",
        "minuteur",
        "alarme",
        "alarm",
        "demain",
        "tomorrow",
        "tous les",
        "chaque",
        "every",
        "dans ",
        " in ",
        "ce soir",
        "tonight",
        "matin",
        "morning",
        " a ",
        "heure",
        "h00",
        "o'clock",
    ),
    "desktop": (
        "ecran",
        "screen",
        "fenetre",
        "window",
        "ouvre",
        "open",
        "ferme",
        "close",
        "lance",
        "launch",
        "app",
        "workspace",
        "bureau",
        "desktop",
        "volume",
        "son ",
        "sound",
        "musique",
        "music",
        "pause",
        "lecture",
        "play",
        "suivant",
        "next",
        "luminosite",
        "brightness",
        "notif",
        "capture",
        "screenshot",
        "presse-papier",
        "clipboard",
        "copie",
        "colle",
        "paste",
        "tape",
        "type",
        "touche",
        "key",
        "clique",
        "click",
        "souris",
        "mouse",
        "verrouille",
        "lock",
        "navigateur",
        "browser",
        "firefox",
        "chrome",
        "site",
        "url",
        "http",
        "vois",
        "regarde",
        "look",
        "see",
        "affiche",
        "plein ecran",
        "fullscreen",
    ),
}


def fold(text: str) -> str:
    """Lower case without accents, as the keyword tables are written."""
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))


def is_self_improve_request(content: str) -> bool:
    return SELF_IMPROVE_REQUEST.search(fold(content)) is not None


# Accent-free stems of an explicit web lookup, and of a short "yes, go ahead".
SEARCH_REQUEST = re.compile(
    r"cherch|\bgoogl|\bsur (internet|le web|le net)\b|\ben ligne\b|\bsearch\b|\blook (it |that )?up\b"
)
AFFIRMATION = re.compile(
    r"^\W*(oui|ouais|ok|okay|d'?accord|vas[- ]y|go|fais[- ]le|lance|c'?est ca|exactement|carrement|bien sur"
    r"|yes|yep|sure)\b"
)
SEARCH_OFFER = re.compile(r"cherch|verifi|chiffres? exacts?|\bsource")
# "Cherche le fichier..." is an agent request on this PC, not a web lookup.
LOCAL_TARGET = re.compile(
    r"fichier|dossier|repertoire|\bfiles?\b|folder|director|\bpc\b|ordi|disque|\bdisk|\bcode\b|projet|project"
    r"|\bgit\b|\blogs?\b|\bprocess|\bpaquet|package"
)


def wants_search(content: str, last_reply: str = "") -> bool:
    """An explicit lookup, or a yes to her own offer to search: small models answer both without searching."""
    text = fold(content)
    if SEARCH_REQUEST.search(text):
        return LOCAL_TARGET.search(text) is None
    offered = fold(last_reply)
    return (len(text.split()) <= 8 and AFFIRMATION.search(text) is not None and "?" in offered) and SEARCH_OFFER.search(
        offered
    ) is not None


def filter_tools_for_message(tools: list[dict[str, Any]], content: str) -> list[dict[str, Any]]:
    text = content.casefold()
    wants_system = any(keyword in text for keyword in SYSTEM_KEYWORDS)
    wants_clap = "clap" in text
    folded = fold(content)
    wanted_groups = {group for group, stems in AGENT_TOOL_KEYWORDS.items() if any(s in folded for s in stems)}
    filtered: list[dict[str, Any]] = []
    for tool in tools:
        name = tool.get("function", {}).get("name", "")
        if name in ("slow_clap", "slow clap") and not wants_clap:
            continue
        group = AGENT_TOOL_GROUPS.get(name)
        if group is not None and not ({group, "any"} & wanted_groups):
            continue
        if name.startswith("mcp.") and name != INTERNET_SEARCH_TOOL and not wants_system:
            continue
        filtered.append(tool)
    return filtered


def measurement_hint(content: str) -> str:
    """Make structured CPU readings explicit without confusing periods with values."""
    try:
        result = json.loads(content)
    except (ValueError, TypeError):
        return ""
    if not isinstance(result, dict):
        return ""
    if all(key in result for key in ("load_1m", "load_5m", "load_15m")):
        values = [str(result[key]) for key in ("load_1m", "load_5m", "load_15m")]
    elif result.get("task") == "cpu_load" and result.get("ok") is True:
        values = str(result.get("stdout", "")).split()[:3]
    else:
        return ""
    if len(values) != 3 or any(not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", v) for v in values):
        return ""
    readings = [f"{float(v):.2f}" for v in values]
    return (
        " Verified measurement, rounded to two decimals: the CPU load averages are "
        + ", ".join(readings)
        + " for the one-minute, five-minute and fifteen-minute periods respectively. "
        "Use this exact factual sentence in your reply. The period names are not the measured values."
    )
