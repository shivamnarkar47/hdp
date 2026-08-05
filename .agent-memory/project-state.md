# Project State


## 2026-08-02 19:36
session: List the files in this directory, then create hello.txt containing exactly 'hi', then tell me the first line of hello.txt → ok

## 2026-08-02 19:37
session: Say 'resumed ok' → ok

## 2026-08-02 19:55
session: Ok, tell me what's in this repo ? → ok

## 2026-08-02 20:23
session: What is this fking repo ? → ok

## 2026-08-02 20:24
Added docs/architecture.md with two mermaid diagrams: (1) module architecture + data-flow flowchart (entry points, runtime assembly, core loop, wire, guarded tools, persistence), (2) sequence diagram of one headless `hdp run` (SSE stream → DSML healing → tool execution → JSONL persist → answer). Diagram facts verified against actual imports in harness/*.py.

## 2026-08-02 20:25
session: Create a mermaid diagram of it and store in docs. → ok

## 2026-08-02 20:40
session: Hello → ok

## 2026-08-02 20:41
session: Ok, give me a brief about what does loop and dialect do. → ok

## 2026-08-02 20:42
session: It stucked on half. → ok

## 2026-08-02 20:42
session: Again stucked → ok

## 2026-08-02 20:43
session: How good is this compared to Oh-my-pi ? → ok

## 2026-08-02 20:46
session: Who is Prime Minister of India → ok

## 2026-08-02 21:47
session: Hi, what's my project about → ok

## 2026-08-02 22:05
session: Hi → ok

## 2026-08-02 22:11
session: OK, what's this project about → ok

## 2026-08-02 22:58
session: Who am I → ok

## 2026-08-02 23:51
session: List the top-level entries of this directory, using a tool. Answer in one short sentence. → ok

## 2026-08-02 23:52
session: Say 'two' → ok

## 2026-08-02 23:52
session: Say 'one' → ok

## 2026-08-02 23:55
session: hello → ok

## 2026-08-02 23:55
session: What this repo all about ? → ok

## 2026-08-02 23:57
session: Yes tell me about loop and DSML dialect. → ok

## 2026-08-03 00:10
session: Give me details about loop and dialect → ok

## 2026-08-03 00:11
session: Hey → ok

## 2026-08-03 00:12
session: What's the tradeoffs now → ok


## 2026-08-04 — TUI workbench redesign
Rebuilt `harness/tui.py` around a compact session bar, framed conversation, context sidebar, compact composer, explicit Send/Cancel state, and clickable empty-state starters. Navigation locks during turns; session and agent context refresh outside the transcript. Added two interaction tests in `tests/test_tui.py`; full suite: 251 tests green.

## 2026-08-04 — home hero art + voice doctrine
Home screen now leads with the KAAL figlet wordmark (`KAAL_LOGO`) plus a simple Panchajanya conch (`MAHABHARATA_ART`) in `harness/art.py`, mirrored into the transcript. AGENTS.md gains §0 Voice & Output Doctrine: epic Mahabharata cadence in plain modern English (never pseudo-archaic) + strict i-have-adhd skill mandate with condensed core rules inlined; skill installed at `~/.agents/skills/i-have-adhd/SKILL.md` (visible to new sessions). Suite: 251 green.

## 2026-08-04 — response latency fixes
MAX_OUTPUT_TOKENS 384k → 32k (bounds runaway reasoning, the dominant slow-response cost; 32k ≈ 96k chars keeps big tool payloads working). PROMPT_BUDGET now explicit 128k, not window-derived (was 616k); overflow-retry truncation uses PROMPT_BUDGET//2. TUI streams reasoning live (transcript mirror stays verbose-only). Tests updated (test_context boundary 968k, test_loop uses PROMPT_BUDGET). Suite: 275 green.

## 2026-08-04 — perceived-latency package
Gateway.warm() pre-opens the keep-alive connection (TCP+TLS handshake at TUI mount, background thread; first turn skips connect RTT; no-op on plain-urllib path). TUI thinking indicator shows live elapsed seconds. `kaal run` shows a live `💭 working Ns` ticker on TTY stderr (off for --verbose/--batch/pipes). Suite: 277 green.

## 2026-08-04 — kaal update
New `kaal update` subcommand: resolves the checkout (KAAL_INSTALL_DIR → ~/.local/share/kaal → running package's .git), git pull --ff-only, then reinstalls into the checkout's .venv (uv or pip, mirroring install.sh). Reports `updated: sha -> sha (subject)` or `up to date`. No checkout → clear error, exit 1. Tested against a real local git origin. Suite: 279 green.

## 2026-08-04 — termaid diagram support
`kaal diagrams <file.mmd>` renders mermaid via termaid (optional dep: pip install kaal[diagrams]); TUI gains `/diagram <path>` printing the Unicode-art render into the conversation. AGENTS.md §0 adds the STRICT mermaid rule: every plan must ship with an accurate ```mermaid diagram, previewed via termaid before delivery. Suite: 282 green.

## 2026-08-04 — mermaid auto-render
TUI turn end scans the finished markdown for ```mermaid fences and pipes each to termaid (stdin, worker thread, ≤3/turn); the Unicode art mounts below the answer in an accent diagram-box; transcript keeps verbatim source. Missing termaid → one dim notice. Cancelled turns skip rendering. Suite: 284 green.

## 2026-08-04 — update rebuilds on pull
`kaal update` now rebuilds (pip/uv install . into the checkout venv) ONLY when the pull brought a new commit; up-to-date runs skip the rebuild; a pulled checkout with no .venv errors with a re-run-install.sh hint. Tests assert exactly one rebuild command and its skip on the second run. Suite: 285 green.
## 2026-08-04 00:58
session: Help me plan the next feature for this project. → ok
