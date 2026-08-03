# AGENTS.md — kaal (`kaal`)

> **Durable anchor memory.** This file is the stable ground of the harness: its first 200 lines are carried, whole, into every kaal agent's system prompt (`harness/prompts.py` → `build_project_context`). What endures belongs at the top; what changes from day to day belongs in `.agent-memory/` (see §6), not here.

## TL;DR / 30-Second Orientation

**What this is:** `kaal` is a Python agent harness, built on the standard library alone, that runs **DeepSeek V4 Flash** with tools, persistent memory, sessions, and a Textual TUI. Its only dependency is `textual`, and that one library serves the TUI and nothing else.

**Get productive immediately:**
- `kaal` — launch the Textual TUI (default surface; requires an API key)
- `kaal run "PROMPT" [flags]` — one-shot headless agent run
- `kaal sessions list` — show persisted sessions
- `.venv/bin/python -m unittest discover -s tests -v` — all 200+ unit tests (stdlib unittest)

**GOTCHA: the two hard things (know these before touching anything):**
1. **DSML healing.** The model does not deliver its tool calls cleanly. It emits them as a DSML XML envelope (`<｜DSML｜tool_calls>`, fullwidth pipe **U+FF5C**) that leaks into visible `delta.content` instead of arriving as structured `tool_calls`. `harness/dialect.py` `DialectFeed` heals it incrementally and strips leaked chat-template tokens (`<｜begin▁of▁sentence｜>`, `<｜Assistant｜>`, …). When both arrive, **structured tool_calls win** over healed ones (`loop.py`).
2. **`reasoning_content` replay.** When an assistant turn has made tool calls, the next request MUST carry that turn's streamed `reasoning_content` verbatim; otherwise the gateway **400s on turn 2+**. `harness/messages.py` `AssistantMessage.to_wire()` always replays it. NEVER synthesize a placeholder.

## Table of Contents

| § | Section | When |
|---|---|---|
| 0 | [Voice & Output Doctrine](#0-voice--output-doctrine) | how every answer is spoken and shaped |
| 1 | [Commands](#1-commands) | build / test / run anything |
| 2 | [Architecture](#2-architecture) | understand data flow |
| 3 | [QUICK START MAP](#3-quick-start-map) | per-file first read |
| 4 | [IF YOU SEE X → Y](#4-if-you-see-x--it-means-y) | confusing output |
| 5 | [PITFALLS](#5-pitfalls) | before editing |
| 6 | [Memory](#6-memory) | when to update what |
| 7 | [Tool preferences](#7-tool-preferences) | efficient exploration |
| 8 | [Navigation order](#8-navigation-order) | new-agent ramp |

## 0. Voice & Output Doctrine

*Purpose: the law of speech and structure for every kaal agent. Non-negotiable — it binds every reply.*

**Epic voice, plain English.** Answer in the measured, dignified cadence of the Mahabharata era — declarative, deliberate, timeless — but in proper, modern English. NEVER pseudo-archaic: no "thou", "doth", "hath", no inverted word order, no thee/thou flourishes. The epic tone is a thin veneer; technical precision and clarity always come first. Short answers stay short; the cadence shapes the delivery, never the content.

**Strict skill mandate (better UX, faster answers).** Read and follow `skill://i-have-adhd` at session start and obey it for EVERY answer. When the skill file is unavailable, the rules below (its condensed core) still bind:

1. Lead with the next action — first line is something the reader can do now.
2. Number multi-step work; each step one bounded action; at most 5 items.
3. End with ONE concrete next action if anything stays open.
4. Suppress tangents — finish the task first; offer side issues separately.
5. Restate state every turn ("step 2 of 4 done; next: …").
6. Give concrete time estimates, never "some work".
7. Make completed work visible — say what works now, in concrete terms.
8. Errors: matter-of-fact cause + fix. Never "uh oh".
9. Cap lists at 5; split do-now vs later when longer.
10. No preamble, no recap, no closing pleasantries. Start with the answer; end when it is done.

**Pre-send check.** Before sending, delete: an announcing opener, a recapping closer, any "by the way" sidebar, hedging adverbs that add no information ("perhaps", "might", "could possibly"), and idioms — state the literal action instead. Then confirm: a reader who sees only the first line and the last line knows what to do next and what just happened. If not, revise; then send.

When the epic voice and the skill's brevity rules conflict, the skill's shape wins — the voice colors the words, the skill shapes the structure. These rules outrank personal style and persist for the whole session.

## 1. Commands

*Purpose: every real command, verified against the installed CLI — copy, paste, and it runs.*

| Command | What it does |
|---|---|
| `kaal` | Launch the Textual TUI (default surface; needs API key) |
| `kaal run "PROMPT"` | One-shot headless run; answer to stdout |
| `kaal run --help` | All run flags |
| `kaal sessions list` | List sessions as `<id> <ts> <prompt>` |
| `kaal sessions show <id>` | Show one session's details/prompt |
| `kaal sessions delete <id>` | Delete one session |
| `kaal sessions prune [--keep N]` | Delete all but the newest N sessions |
| `kaal doctor` | Self-check: python, textual, api key, gateway, structure cache, sessions dir |
| `kaal --version` | Print `kaal 0.1.0` and exit |
| `.venv/bin/python -m unittest discover -s tests -v` | All unit tests |
| `git config core.hooksPath .githooks` | Enable build-check hooks: pre-commit & pre-push run compileall + unittest + `kaal --version`; skip with `KAAL_SKIP_HOOKS=1` |

`kaal run -` reads the prompt from stdin in place of `"PROMPT"` — the route for piped input.

**`kaal run` flags** (from `--help`):

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
| `--no-tool-cache` | disable the read-only tool-result cache (`.kaal/tool-cache.json`) | off (cache on) |
| `--no-verify` | disable verify hooks after mutation (`.kaal/hooks.json`) | off (verify on) |

**Exit codes:** `0` answer produced · `1` config/key/gateway error · `2` loop error (max steps, context overflow, tool loop, 5 consecutive tool failures).

**API key:** the key is sought in this order: env `OPENCODE_API_KEY`, then the user key store `~/.config/kaal/api_key` (0600; saved from the TUI via `/connect`), then the omp auth store `~/.omp/agent/agent.db` (read-only sqlite), and failing all of these, exit 1 with instructions. Never cache or write it outside `config.save_user_api_key`.

**Sessions:** each session is a JSONL record at `~/.local/share/kaal/sessions/` (override: env `KAAL_SESSIONS_DIR`). The id takes the form `%Y%m%d-%H%M%S` (`sessions.py`).

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

**TUI keys:** `Ctrl+C` cancels the running turn (cooperative) · `Ctrl+Q` quits · `up`/`down` walk prompt history. The font the TUI shows is the terminal's own, not the app's: configure the emulator (Fira Sans Condensed; see `docs/terminal-setup.md`).

## 2. Architecture

*Purpose: how the code is shaped, told in plain words — the layering, the flow of data, no diagram.*

**Data flow (one `kaal run`):** the road of a single run is fixed. `cli.py` builds `Gateway` + `Memory` + `ToolRegistry` + `AgentLoop`, then calls `loop.run(prompt, emit)`. Each turn follows the same road: `to_wire_messages()` → `gateway.stream()` (SSE) → `DialectFeed` heals DSML out of `content` deltas → resolved `ToolCall`s → `ToolRegistry.execute()` → the result is appended as a tool message and persisted to the session JSONL → and so it repeats until the model answers with no tool calls, or `max_steps` is spent.

**Layering:**
- **Core — stdlib-only, ports 1:1 to Rust/Go:** `config.py`, `gateway.py`, `dialect.py`, `messages.py`, `context.py`, `loop.py`. No new dependencies, ever.
- **Persistence:** `memory.py` (`.agent-memory/`), `sessions.py` (JSONL store).
- **Tools:** `tools.py` — OpenAI function schemas + guarded execution (path confinement, DENY list).
- **Front-end:** `tui.py` — the ONLY module importing `textual`; thin, disposable. `cli.py` lazy-imports it only on the no-subcommand path.
- **Port seam:** the `AgentEvent` stream in `loop.py` — the TUI and `kaal run` both consume exactly this; never bypass it.
- **Gateway behavior:** retries 5xx/network up to 3× (1s/2s/4s backoff); 4xx raises immediately; never retries after visible content.
- **Parallel tool batches:** all-read batches (`read`/`grep`/`glob`) run concurrently (≤4 workers); any batch containing a mutator runs serially in call order. Events, persistence, tool-loop detection, and failure counting are recorded in call order on the main thread.
- **grep:** rg-backed when `rg` is on PATH (streamed so scanning stops at the result cap); pure-Python scan is the fallback (missing binary, exit 2, empty pattern, OSError). `.kaal` joins grep's skip dirs.
- **HTTP keep-alive:** one connection reused across turns (per-thread sockets, so `--batch` workers never share one); off with `KAAL_NO_KEEPALIVE=1` or any proxy env var; reconnect-on-error degenerates to the plain urllib path.
- **Tool-result cache:** read/grep/glob results cached in `.kaal/tool-cache.json` (git-ignored, atomic write, 4 MB cap) keyed by `tool|sha256(args)|structure_signature` — a changed tree auto-misses. Staleness is only possible for external edits between refreshes; a mutating batch bypasses lookups for the whole step and drops the cache at refresh; `--no-tool-cache` disables.
- **Verify hooks:** after a mutating batch, the configured `.kaal/hooks.json` `verify` command runs (30 s timeout) and its output is appended as a `user` message (`[verify] …`, dimmed in TUI, stderr in `kaal run`) — content for the model, never a loop abort. No hooks file = off; `--no-verify` disables.
- **spawn_agent:** nested `AgentLoop` on a sub-task (own session id, visible in `kaal sessions list`; serially for v1). Recursion depth-capped at 2; nested runs get `allow_dangerous=False` and no tool cache.

**Events:**
- `AgentEvent` (loop → front end): `("content",str) | ("reasoning",str) | ("tool_start",ToolCall) | ("tool_result",id,str) | ("done",str) | ("error",str)`
- `StreamEvent` (gateway → loop): `("content",str) | ("reasoning",str) | ("tool_call",ToolCall) | ("done",finish_reason) | ("error",str)`

**PATTERN — the canonical example:** `tests/test_loop.py::test_two_turn_tool_call_flow` holds the whole loop contract in one place: a fake gateway streams DSML, the loop heals it, executes, replays reasoning verbatim on turn 2, persists, and answers. Read this one test and the loop is known.

## 3. QUICK START MAP

*Purpose: which file serves which end, and when to open it.*

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

*Purpose: to read a strange output and know its meaning at once.*

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
| `KAAL_NO_KEEPALIVE=1` | keep-alive transport off (plain urllib path); also auto-off with any proxy env var |
| stale tool results after external edits | read-only tool cache is signature-keyed (changed tree = miss) with a same-step write/read bypass; opt out with `--no-tool-cache` |
| `[verify] …` user message after a mutation batch | post-mutation self-check ran (`.kaal/hooks.json`); its output is fed back to the model as content |
| `spawn_agent: recursion limit reached` | nested-agent depth cap (2 loops) — an expected guardrail, not an error |

## 5. PITFALLS

*Purpose: the mistakes that have cost hours — read them before you edit.*

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

*Purpose: what to write down, where, and when.*

**Files** (committed, in `.agent-memory/`): `project-state.md` · `decisions.md` · `patterns.md` · `lessons-learned.md`.

**Update triggers** — write when: a milestone is complete; an architectural decision is made; a non-obvious gotcha is discovered; anything has consumed excessive time. Use the `memory_append` tool (sections: `project-state | decisions | patterns | lessons-learned`) or edit the files directly.

**Rules** (`memory.py`): 200-line cap per file (oldest `##` section pruned); digest capped at 4000 est. tokens and 60 lines/section, head-biased — put critical notes early, keep entries self-contained; verbatim dedupe returns `already recorded`; each session outcome is auto-appended to `project-state.md` (`record_session_summary`).

**AGENTS.md is the durable anchor; `.agent-memory/` is the moving state.** Edit AGENTS.md only for stable, load-bearing facts; put evolving state in the memory files.

### `.kaal/` files — caches & config, NOT memory

No memory dwells under `.kaal/`; everything there is regenerable cache or explicit config: `STRUCTURE.md` (tree cache, below), `tool-cache.json` (read-only tool-result cache, §2), and `hooks.json` (verify-hook config, §2). `harness/structure.py` scans the project tree (noise dirs skipped: `.git` `.venv` `node_modules` `.kaal` `dist` `build` `.omp` `__pycache__` + caches; depth ≤ 6, ≤ 20k entries, ≤ 500 lines) and writes a markdown tree under `.kaal/` (git-ignored; atomic temp+replace write). A signature (`<!-- sig: … -->` comment at the end) hashes (relpath, size, mtime_ns); `refresh()` regenerates only when it changed, `ensure()` never rescans an existing cache. The first ~120 lines are injected into the system prompt (`prompts.build_project_context`) so reopen is instant. Refreshed after every tool batch (`loop._one_step`) and between TUI turns (`turn_finished`); TUI shows a one-line summary on mount and `/structure` dumps the doc.

## 7. Tool preferences

*Purpose: to explore with the least cost to the context.*

1. **Directory listing** (`read` on a directory → ~2-level listing) to orient → **`grep`** to locate → **line-selected `read`** (`offset`/`limit` or `:N-M`) to read only what's needed.
2. `grep` before whole-file reads, always. `glob` to map structure.
3. Never whole-file-read to find one symbol; never re-read files you already have.

## 8. Navigation order

*Purpose: the fastest road for a new agent.*

1. This file (you're here).
2. `tests/test_loop.py::test_two_turn_tool_call_flow` — the whole loop contract in one test.
3. `harness/loop.py` → `harness/dialect.py` → `harness/messages.py` (the two hard things).
4. `harness/cli.py` + `harness/gateway.py` (entry + wire).
5. `harness/tools.py` (safety) → `harness/memory.py` + `harness/sessions.py` (persistence).
6. `harness/tui.py` last — it's a thin consumer of the `AgentEvent` stream.
