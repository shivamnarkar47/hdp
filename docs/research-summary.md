# Research Summary — Harness Engineering Findings for kala

A digest of harness-engineering research (context engineering, agent memory, operational boundaries) as applied to this repo. Companion to the root `AGENTS.md`.

## Context Engineering

- **Progressive disclosure is already implemented.** `harness/prompts.py::build_project_context` auto-injects the first 200 lines of `AGENTS.md` into every system prompt. The repo's pattern is "anchor + satellite": `AGENTS.md` is the durable anchor; `.agent-memory/` files carry dynamic state. Consequence: the top of `AGENTS.md` must be the highest-signal content an agent needs on turn one.
- **Signal-to-noise is enforced structurally, not by prose.** The system prompt is a fixed prefix (boundaries, output contract, tool rules) + memory digest + project context. Tool schemas travel only in the API `tools` parameter, never in prose — no duplicated schema tokens in the prompt.
- **Chunking is built in.** Memory digest caps at 4000 estimated tokens and 60 lines per section; each memory file caps at 200 lines (oldest section pruned); history truncation drops oldest turns but never the system message, last user message, or the turn after it. Token estimation is deliberately rough (`len // 3`) — fine for budgets, too coarse for anything else.

## Agent Memory Systems

- **What's worth persisting:** decisions, patterns, lessons learned, and session outcomes (auto-appended to `project-state.md` by `record_session_summary`). **What stays ephemeral:** per-turn content, tool results, reasoning — they live in the JSONL session store, not memory files.
- **Update protocol:** append via the `memory_append` tool (sections `project-state|decisions|patterns|lessons-learned`) at milestone completion, after architectural decisions, and after any time-consuming fix or non-obvious gotcha. Verbatim dedupe returns `already recorded`; the 200-line cap prunes the oldest `##` section.
- **Digest asymmetry:** the model sees a head-biased, truncated digest, not the full files. Recent entries must be self-contained; critical notes belong early in a file.

## Operational Boundaries

- **A safe default beats a clever one.** Tools are cwd-constrained (path escapes blocked) and a DENY list blocks destructive commands unless `--allow-dangerous`. The DENY message is a hard stop, not a prompt to work around.
- **The two hard things are wire-contract requirements, not cosmetics.** DSML healing and `reasoning_content` replay exist because the gateway 400s on turn 2+ without exact reasoning replay and leaks tool calls through `delta.content`. Any port (Rust/Go) must reproduce both exactly, including the load-bearing Unicode markers (U+FF5C, U+2581).
- **Boundary discipline keeps the port cheap.** Core (`gateway`, `dialect`, `messages`, `context`, `loop`) is stdlib-only; the `AgentEvent` stream is the front-end seam; the Textual TUI is a thin disposable consumer confined to `tui.py`.
- **Tool-selection guidance** (directory listing → grep → line-selected read) already lives in the system prompt and should be followed to keep context spend flat.

## Repo-vs-brief Discrepancies

None found. Config constants match the catalog compat block referenced in `config.py`; all documented commands in `AGENTS.md` were verified against the installed CLI; all 79 unit tests pass. Note: `.agent-memory/` files are currently empty stubs (headers only) — the memory system is wired but unused so far.
