# HarnessDP

[![python](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org/)
[![last commit](https://img.shields.io/github/last-commit/shivamnarkar47/hdp)](https://github.com/shivamnarkar47/hdp)
[![code size](https://img.shields.io/github/languages/code-size/shivamnarkar47/hdp)](https://github.com/shivamnarkar47/hdp)
[![tests](https://img.shields.io/github/actions/workflow/status/shivamnarkar47/hdp/tests.yml?label=tests)](https://github.com/shivamnarkar47/hdp/actions)
[![core stdlib only](https://img.shields.io/badge/core-stdlib%20only-green)](https://github.com/shivamnarkar47/hdp)

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
- `hdp run "PROMPT"` — one-shot session from the CLI (`--dir`, `--model`, `--max-steps`, `--resume SESSION_ID`, `--json`, …); `--batch FILE --workers N` runs many prompts (one per line or a JSON array) as parallel sessions; `--no-tool-cache` and `--no-verify` disable the read-result cache and post-mutation verify hooks
- `hdp sessions list` — list past sessions
- `hdp sessions show|delete|prune` — inspect/delete/prune sessions; `hdp doctor` — self-check; `hdp run -` — read the prompt from stdin; `hdp --version`
- TUI slash commands: `/help`, `/new`, `/resume <id>`, `/sessions`, `/connect`, `/structure`, `/memory`, `/model`, `/verbose`, `/quit`
- terminal font: Fira Sans Condensed — see `docs/terminal-setup.md`

The project tree is scanned automatically and cached in `.hdp/STRUCTURE.md` (git-ignored, regenerable); it's injected into the system prompt and refreshed when files change. `.hdp/` also holds the read-only tool-result cache (`tool-cache.json`, keyed to the tree signature, `--no-tool-cache` to disable) and optional verify hooks (`hooks.json`, run after mutating steps, `--no-verify` to disable).

## How it works

```mermaid
flowchart TD
    ENTRY["hdp run / TUI"] --> LOOP["AgentLoop — stream → heal → execute → persist"]
    LOOP -->|"to_wire_messages() → stream()"| GW["Gateway (SSE, OpenAI-compatible)"]
    GW -->|"content deltas"| DIALECT["DialectFeed — heal DSML envelopes"]
    DIALECT -->|"ToolCall"| LOOP
    LOOP -->|"execute()"| TOOLS["ToolRegistry — path-confined + DENY list"]
    TOOLS --> BASH["bash — ≤300s, 10k-char cap"] & FS["read / write / edit / grep / glob"] & MAP["memory_append → .agent-memory/"]
    LOOP -->|"persist / resume"| SESS["sessions JSONL"]
    SESS -->|"replay history"| LOOP
    GW -->|"wire body: no tool_choice / temperature / stream_options"| API["DeepSeek V4 Flash gateway"]
    API -->|"SSE StreamEvents"| GW
    LOOP -->|"answer"| OUT["done(answer) → exit 0"]
```

## API key

Set `OPENCODE_API_KEY` in your environment, or save a key from the TUI with `/connect` (resolution order: env → saved key → omp auth store).

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
