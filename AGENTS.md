# AGENTS.md — HarnessDP (`hdp`)

> **Durable anchor memory.** The first 200 lines of this file are auto-injected into every hdp agent's system prompt (`harness/prompts.py` → `build_project_context`), so the top is the highest-signal content. Dynamic state lives in `.agent-memory/` (see §6).

## TL;DR / 30-Second Orientation

**What this is:** `harnessdp` = "hdp", a stdlib-only Python agent harness running **DeepSeek V4 Flash** with tools, persistent memory, sessions, and a Textual TUI. Only dependency: `textual`, used ONLY by the TUI.

**Get productive immediately:**
- `hdp` — launch the Textual TUI (default surface; requires an API key)
- `hdp run "PROMPT" [flags]` — one-shot headless agent run
- `hdp sessions list` — show persisted sessions
- `.venv/bin/python -m unittest discover -s tests -v` — all 200+ unit tests (stdlib unittest)

**GOTCHA: the two hard things (know these before touching anything):**
1. **DSML healing.** The model emits tool calls as a DSML XML envelope (`<｜DSML｜tool_calls>`, fullwidth pipe **U+FF5C**) that leaks into visible `delta.content` instead of arriving as structured `tool_calls`. `harness/dialect.py` `DialectFeed` heals it incrementally and strips leaked chat-template tokens (`<｜begin▁of▁sentence｜>`, `<｜Assistant｜>`, …). When both arrive, **structured tool_calls win** over healed ones (`loop.py`).
2. **`reasoning_content` replay.** Assistant turns that made tool calls MUST re-send their streamed `reasoning_content` verbatim on the next request, or the gateway **400s on turn 2+**. `harness/messages.py` `AssistantMessage.to_wire()` always replays it. NEVER synthesize a placeholder.

## Table of Contents

| § | Section | When |
|---|---|---|
| 1 | [Commands](#1-commands) | build / test / run anything |
| 2 | [Architecture](#2-architecture) | understand data flow |
| 3 | [QUICK START MAP](#3-quick-start-map) | per-file first read |
| 4 | [IF YOU SEE X → Y](#4-if-you-see-x--it-means-y) | confusing output |
| 5 | [PITFALLS](#5-pitfalls) | before editing |
| 6 | [Memory](#6-memory) | when to update what |
| 7 | [Tool preferences](#7-tool-preferences) | efficient exploration |
| 8 | [Navigation order](#8-navigation-order) | new-agent ramp |

## 1. Commands

*Purpose: every real command, copy-paste verified against the installed CLI.*

| Command | What it does |
|---|---|
| `hdp` | Launch the Textual TUI (default surface; needs API key) |
| `hdp run "PROMPT"` | One-shot headless run; answer to stdout |
| `hdp run --help` | All run flags |
| `hdp sessions list` | List sessions as `<id> <ts> <prompt>` |
| `hdp sessions show <id>` | Show one session's details/prompt |
| `hdp sessions delete <id>` | Delete one session |
| `hdp sessions prune [--keep N]` | Delete all but the newest N sessions |
| `hdp doctor` | Self-check: python, textual, api key, gateway, structure cache, sessions dir |
| `hdp --version` | Print `hdp 0.1.0` and exit |
| `.venv/bin/python -m unittest discover -s tests -v` | All unit tests |

`hdp run -` reads the prompt from stdin instead of `"PROMPT"` (useful for pipes).

**`hdp run` flags** (from `--help`):

| Flag | Meaning | Default |
|---|---|---|
| `prompt` | task to run (positional) | required |
| `--dir DIR` | project directory — tools are cwd-constrained to it | cwd |
| `--model MODEL` | model id | `deepseek-v4-flash` |
| `--max-steps MAX_STEPS` | max agent turns | 20 |
| `--memory-root MEMORY_ROOT` | memory directory | `<dir>/.agent-memory` |
| `--allow-dangerous` | skip the destructive-command DENY list | off |
| `--resume SESSION_ID` | continue a session | none |
| `--verbose` | print reasoning to stderr | off |
| `--json` | final JSON line `{"session_id","answer","steps","tool_calls"}` | off |
| `--batch FILE` | run prompts from FILE (one per line, or a JSON array), one session each | none |
| `--workers N` | max concurrent `--batch` tasks | `min(4, cpu count)` |
| `--no-tool-cache` | disable the read-only tool-result cache (`.hdp/tool-cache.json`) | off (cache on) |
| `--no-verify` | disable verify hooks after mutation (`.hdp/hooks.json`) | off (verify on) |

**Exit codes:** `0` answer produced · `1` config/key/gateway error · `2` loop error (max steps, context overflow, tool loop, 5 consecutive tool failures).

**API key:** env `OPENCODE_API_KEY`, else the user key store `~/.config/harnessdp/api_key` (0600; saved from the TUI via `/connect`), else omp auth store `~/.omp/agent/agent.db` (read-only sqlite), else exit 1 with instructions. Never cache or write it outside `config.save_user_api_key`.

**Sessions:** JSONL at `~/.local/share/harnessdp/sessions/` (override: env `HARNESSDP_SESSIONS_DIR`). Id format `%Y%m%d-%H%M%S` (`sessions.py`).

**TUI slash commands** (`tui.py`):

| Command | Action |
|---|---|
| `/help` | list commands |
| `/new` | fresh session id, clear pane |
| `/resume <id>` | continue a session |
| `/sessions` | popup session switcher (Enter resume · Esc close) |
| `/connect` | popup to save the API key (or `/connect <key>` inline) |
| `/memory` | show memory digest + file paths |
| `/model` | show current model id |
| `/verbose` | toggle reasoning display |
| `/quit` | exit |

**TUI keys:** `Ctrl+C` cancels the running turn (cooperative) · `Ctrl+Q` quits · `up`/`down` walk prompt history.

## 2. Architecture

*Purpose: the shape of the code in plain text — layering, data flow, no diagram.*

**Data flow (one `hdp run`):** `cli.py` builds `Gateway` + `Memory` + `ToolRegistry` + `AgentLoop` → `loop.run(prompt, emit)`. Per turn: `to_wire_messages()` → `gateway.stream()` (SSE) → `DialectFeed` heals DSML out of `content` deltas → resolved `ToolCall`s → `ToolRegistry.execute()` → result appended as a tool message and persisted to the session JSONL → repeat until the model answers (no tool calls) or `max_steps`.

**Layering:**
- **Core — stdlib-only, ports 1:1 to Rust/Go:** `config.py`, `gateway.py`, `dialect.py`, `messages.py`, `context.py`, `loop.py`. No new dependencies, ever.
- **Persistence:** `memory.py` (`.agent-memory/`), `sessions.py` (JSONL store).
- **Tools:** `tools.py` — OpenAI function schemas + guarded execution (path confinement, DENY list).
- **Front-end:** `tui.py` — the ONLY module importing `textual`; thin, disposable. `cli.py` lazy-imports it only on the no-subcommand path.
- **Port seam:** the `AgentEvent` stream in `loop.py` — the TUI and `hdp run` both consume exactly this; never bypass it.
- **Gateway behavior:** retries 5xx/network up to 3× (1s/2s/4s backoff); 4xx raises immediately; never retries after visible content.
- **Parallel tool batches:** all-read batches (`read`/`grep`/`glob`) run concurrently (≤4 workers); any batch containing a mutator runs serially in call order. Events, persistence, tool-loop detection, and failure counting are recorded in call order on the main thread.
- **grep:** rg-backed when `rg` is on PATH (streamed so scanning stops at the result cap); pure-Python scan is the fallback (missing binary, exit 2, empty pattern, OSError). `.hdp` joins grep's skip dirs.
- **HTTP keep-alive:** one connection reused across turns (per-thread sockets, so `--batch` workers never share one); off with `HARNESSDP_NO_KEEPALIVE=1` or any proxy env var; reconnect-on-error degenerates to the plain urllib path.
- **Tool-result cache:** read/grep/glob results cached in `.hdp/tool-cache.json` (git-ignored, atomic write, 4 MB cap) keyed by `tool|sha256(args)|structure_signature` — a changed tree auto-misses. Staleness is only possible for external edits between refreshes; a mutating batch bypasses lookups for the whole step and drops the cache at refresh; `--no-tool-cache` disables.
- **Verify hooks:** after a mutating batch, the configured `.hdp/hooks.json` `verify` command runs (30 s timeout) and its output is appended as a `user` message (`[verify] …`, dimmed in TUI, stderr in `hdp run`) — content for the model, never a loop abort. No hooks file = off; `--no-verify` disables.
- **spawn_agent:** nested `AgentLoop` on a sub-task (own session id, visible in `hdp sessions list`; serially for v1). Recursion depth-capped at 2; nested runs get `allow_dangerous=False` and no tool cache.

**Events:**
- `AgentEvent` (loop → front end): `("content",str) | ("reasoning",str) | ("tool_start",ToolCall) | ("tool_result",id,str) | ("done",str) | ("error",str)`
- `StreamEvent` (gateway → loop): `("content",str) | ("reasoning",str) | ("tool_call",ToolCall) | ("done",finish_reason) | ("error",str)`

**PATTERN — canonical example:** `tests/test_loop.py::test_two_turn_tool_call_flow` shows the whole loop contract: fake gateway streams DSML → loop heals → executes → replays reasoning verbatim on turn 2 → persists → answers. Read this one test to learn the loop.

## 3. QUICK START MAP

*Purpose: file → purpose → when to open.*

| File | Purpose | Open when… |
|---|---|---|
| `harness/cli.py` | Entry point: subcommands, flags, exit codes | tracing a command or exit code |
| `harness/tui.py` | Textual split-pane app: conversation pane + Trace/Memory/Sessions sidebar + status bar; slash commands; worker thread | working on the UI |
| `harness/loop.py` | `AgentLoop`: stream→heal→execute→persist; `AgentEvent` seam | tracing agent behavior end-to-end |
| `harness/gateway.py` | SSE client; wire body/headers; retries; port-boundary file | touching the wire protocol |
| `harness/dialect.py` | DSML state machine + leaked-token stripper | healing bugs — every agent touches this eventually |
| `harness/messages.py` | Wire model; `reasoning_content` replay rule | message shape, or 400s on turn 2+ |
| `harness/context.py` | Token estimate + history truncation | budget / overflow logic |
| `harness/tools.py` | Tool registry, schemas, path safety, DENY list | tools or safety |
| `harness/memory.py` | `.agent-memory/` persistence, digest, caps | memory behavior |
| `harness/sessions.py` | JSONL session store, resume replay | sessions / resume |

## 4. IF YOU SEE X → IT MEANS Y

*Purpose: decode confusing output instantly.*

| You see… | It means… |
|---|---|
| DSML tags (`<｜DSML｜invoke …>`) in output | a tool call was healed by `DialectFeed` — don't strip it manually |
| HTTP 400 on turn 2+ | `reasoning_content` was dropped; replay it verbatim (`messages.py`) |
| Model never calls tools | no `tool_choice` support — never send it (nor `temperature` / `stream_options` / `store`) |
| `…[truncated]` at the end of tool output | 10k-char cap (`MAX_RESULT_CHARS`, `tools.py`) |
| `blocked by harness policy (destructive command)` | DENY list fired; re-run with `--allow-dangerous` only if intentional |
| `old_text matches N times; pass all=true to replace all` | `edit` refuses ambiguous replaces |
| `<think>…</think>` inside content | reasoning span — routed to `("reasoning", …)`, not answer text |
| `Discarding unclosed DSML section…` (log) | unclosed envelope that parsed ≥1 invoke — `flush()` discards it (a malformed real call is better lost than executed); sections with **0 invokes** are now RECOVERED as visible text, not discarded (they were prose quotes of the envelope) |
| `tool loop detected` / `5 consecutive tool failures` | loop aborted; exit 2 |
| `(busy — Ctrl+C cancels the current turn)` | TUI turn in flight; input disabled until done |
| `HARNESSDP_NO_KEEPALIVE=1` | keep-alive transport off (plain urllib path); also auto-off with any proxy env var |
| stale tool results after external edits | read-only tool cache is signature-keyed (changed tree = miss) with a same-step write/read bypass; opt out with `--no-tool-cache` |
| `[verify] …` user message after a mutation batch | post-mutation self-check ran (`.hdp/hooks.json`); its output is fed back to the model as content |
| `spawn_agent: recursion limit reached` | nested-agent depth cap (2 loops) — an expected guardrail, not an error |

## 5. PITFALLS

*Purpose: mistakes that cost hours — read before editing.*

- **PITFALL: core must stay stdlib-only.** `gateway/dialect/messages/context/loop` (plus `config/prompts/tools/memory/sessions`) map 1:1 to a Rust/Go port. No new deps in core. `textual` is legal ONLY in `harness/tui.py` (`cli.py` lazy-imports it).
- **PITFALL: never send `tool_choice`, `temperature`, `stream_options`, or `store`.** This model rejects them (`gateway._build_body`); `test_build_body` asserts their absence.
- **PITFALL: unicode markers are load-bearing.** ｜ = U+FF5C, ▁ = U+2581. Match them exactly (`FW = "\uff5c"`, `B = "\u2581"`; build fixtures from escapes, never paste glyphs). The model never trained on ASCII substitutes — transliterating breaks healing.
- **PITFALL: reasoning replay is mandatory.** `AssistantMessage.to_wire()` re-sends `reasoning_content` when present; never synthesize a placeholder. Dropping it 400s on the next turn.
- **PITFALL: call `DialectFeed.flush()` at end of stream** (the loop does); unclosed sections that parsed ≥1 invoke are deliberately discarded there, not raised. Unclosed sections with 0 invokes are recovered as visible text — the model quoted the envelope in prose — and an envelope that follows any visible text in the same turn is treated as a prose quote, never healed (real envelopes are generation-leading).
- **PITFALL: structured beats healed.** `calls = structured_calls if structured_calls else healed_calls` in `loop.py` — don't "fix" that precedence.
- **PITFALL: tool results are strings.** 10k-char cap; `bash` timeout 30s default / 300s max; `grep` is case-insensitive unless `case:true`; `read` `offset` is 1-based.
- **PITFALL: TUI thread rules.** The loop runs on a worker thread; that thread never touches widgets — events marshal via `call_from_thread`. Keep it that way. Streaming markdown re-renders the whole document per update, so the TUI accumulates chunks and flushes at most every ~100 ms (timer owned by the main thread); don't update the `Markdown` widget from the emit callback directly.
- **PITFALL: don't name an attribute `_loop` on the App** — Textual's `App._loop` is internal (`tui.py` comment; the field is `_agent_loop`).

## 6. Memory

*Purpose: what to persist, where, and when.*

**Files** (committed, in `.agent-memory/`): `project-state.md` · `decisions.md` · `patterns.md` · `lessons-learned.md`.

**Update triggers** — write after: milestone completion; architectural decisions; discovering non-obvious gotchas; fixing anything that consumed excessive time. Use the `memory_append` tool (sections: `project-state | decisions | patterns | lessons-learned`) or edit the files directly.

**Rules** (`memory.py`): 200-line cap per file (oldest `##` section pruned); digest capped at 4000 est. tokens and 60 lines/section, head-biased — put critical notes early, keep entries self-contained; verbatim dedupe returns `already recorded`; each session outcome is auto-appended to `project-state.md` (`record_session_summary`).

**AGENTS.md = durable anchor; `.agent-memory/` = dynamic state.** Edit AGENTS.md only for stable, load-bearing facts; use memory files for evolving state.

### `.hdp/` files — caches & config, NOT memory

Memory lives only in `.agent-memory/`; everything under `.hdp/` is regenerable cache or explicit config: `STRUCTURE.md` (tree cache, below), `tool-cache.json` (read-only tool-result cache, §2), and `hooks.json` (verify-hook config, §2). `harness/structure.py` scans the project tree (noise dirs skipped: `.git` `.venv` `node_modules` `.hdp` `dist` `build` `.omp` `__pycache__` + caches; depth ≤ 6, ≤ 20k entries, ≤ 500 lines) and writes a markdown tree under `.hdp/` (git-ignored; atomic temp+replace write). A signature (`<!-- sig: … -->` comment at the end) hashes (relpath, size, mtime_ns); `refresh()` regenerates only when it changed, `ensure()` never rescans an existing cache. The first ~120 lines are injected into the system prompt (`prompts.build_project_context`) so reopen is instant. Refreshed after every tool batch (`loop._one_step`) and between TUI turns (`turn_finished`); TUI shows a one-line summary on mount and `/structure` dumps the doc.

## 7. Tool preferences

*Purpose: explore with the least context spend.*

1. **Directory listing** (`read` on a directory → ~2-level listing) to orient → **`grep`** to locate → **line-selected `read`** (`offset`/`limit` or `:N-M`) to read only what's needed.
2. `grep` before whole-file reads, always. `glob` to map structure.
3. Never whole-file-read to find one symbol; never re-read files you already have.

## 8. Navigation order

*Purpose: fastest ramp for a new agent.*

1. This file (you're here).
2. `tests/test_loop.py::test_two_turn_tool_call_flow` — the whole loop contract in one test.
3. `harness/loop.py` → `harness/dialect.py` → `harness/messages.py` (the two hard things).
4. `harness/cli.py` + `harness/gateway.py` (entry + wire).
5. `harness/tools.py` (safety) → `harness/memory.py` + `harness/sessions.py` (persistence).
6. `harness/tui.py` last — it's a thin consumer of the `AgentEvent` stream.
