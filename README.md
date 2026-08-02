# HarnessDP

hdp is a DeepSeek V4 Flash agent harness: a Textual TUI plus a `hdp run` CLI that drives gateway-backed agent sessions. The core is stdlib-only — `textual` is the only runtime dependency.

## Requirements

- Python >= 3.12
- git (recommended; the installer falls back to a tarball without it)

## Install

Linux/macOS:

```sh
curl -fsSL https://raw.githubusercontent.com/shivamnarkar47/hdp/main/install.sh | bash
```

Windows (PowerShell):

```powershell
powershell -c "irm https://raw.githubusercontent.com/shivamnarkar47/hdp/main/install.ps1 | iex"
```

Both installers honor the `HDP_REPO_URL`, `HDP_INSTALL_DIR`, and `HDP_BIN_DIR` environment overrides, and re-running them updates an existing install.

## Usage

- `hdp` — launch the Textual TUI
- `hdp run "PROMPT"` — one-shot session from the CLI (`--dir`, `--model`, `--max-steps`, `--resume SESSION_ID`, `--json`, …)
- `hdp sessions list` — list past sessions
- TUI slash commands: `/help`, `/new`, `/resume <id>`, `/sessions`, `/memory`, `/model`, `/verbose`, `/quit`

## API key

Set `OPENCODE_API_KEY` in your environment; otherwise the harness reads the omp auth store.

## Tests

```sh
.venv/bin/python -m unittest discover -s tests -v
```

## Uninstall

Linux/macOS:

```sh
rm -rf ~/.local/share/hdp ~/.local/bin/hdp
```

Windows: delete the install dir (`%USERPROFILE%\.local\share\hdp`) and the launcher (`%USERPROFILE%\.local\bin\hdp.cmd`), then remove any `PATH` entry pointing at the bin dir.
