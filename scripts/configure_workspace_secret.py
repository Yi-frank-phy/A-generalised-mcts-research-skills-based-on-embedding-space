"""Store a Gemini API key outside the repository for Workspace DTE runs."""

from __future__ import annotations

import getpass
import os
from pathlib import Path


def default_secret_path() -> Path:
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "DTE" / "secrets.env"
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_home / "dte" / "secrets.env"


def main() -> None:
    key = getpass.getpass("Gemini API key (input hidden): ").strip()
    if not key:
        raise SystemExit("No key supplied; nothing written.")
    if "\n" in key or "\r" in key:
        raise SystemExit("Invalid key: line breaks are not allowed.")
    path = default_secret_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(f"GEMINI_API_KEY={key}\n", encoding="utf-8")
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass
    temporary.replace(path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    print(f"Gemini key stored outside the repository at: {path}")
    print("The key value was not printed. Do not add this file to Git.")


if __name__ == "__main__":
    main()
