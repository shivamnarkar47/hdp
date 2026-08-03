# kaal — Architecture Diagrams

Mermaid diagrams for the `kaal` agent harness. Render them with any Mermaid
viewer (GitHub, `mmdc`, VS Code extension).

## 1. Module architecture & data flow

```mermaid
flowchart TD
    subgraph ENTRY["Entry points"]
        CLI["harness/cli.py — argparse; kaal run / kaal sessions / lazy TUI launch"]
        TUI["harness/tui.py — Textual split-pane UI (the ONLY textual import)"]
    end

    subgraph RUNTIME["Runtime assembly (built per run)"]
        LOOP["harness/loop.py — AgentLoop (stream → heal → execute → persist)"]
        GW["harness/gateway.py — Gateway, SSE client, 3x retry"]
        MEM["harness/memory.py — Memory (.agent-memory/ digest)"]
        REG["harness/tools.py — ToolRegistry, schemas + guards"]
    end

    subgraph CORE["Core support (stdlib-only, port seam)"]
        DIALECT["harness/dialect.py — DialectFeed, DSML healing + token stripping"]
        MSG["harness/messages.py — wire model, reasoning_content replay"]
        CTX["harness/context.py — token estimate, history truncation"]
        PROMPTS["harness/prompts.py — system prompt + AGENTS.md context"]
        CFG["harness/config.py — constants, API-key resolution"]
    end

    subgraph WIRE["Wire"]
        API["DeepSeek V4 Flash gateway (SSE, OpenAI-compatible)"]
    end

    subgraph TOOLS["Guarded tool execution"]
        BASH["bash — DENY list, timeout ≤300s, 10k-char cap"]
        FS["read / write / edit / grep / glob — path-confined"]
        MAP["memory_append — .agent-memory/ sections"]
    end

    subgraph PERSIST["Persistence"]
        AM[".agent-memory/ — project-state, decisions, patterns, lessons"]
        SESS["harness/sessions.py — JSONL store, resume"]
        AGENTS["AGENTS.md — durable anchor, first 200 lines injected into every prompt"]
    end

    CLI --> LOOP
    TUI --> LOOP
    CLI --> GW
    CLI --> MEM
    CLI --> REG
    TUI --> GW
    TUI --> MEM
    TUI --> REG

    LOOP -->|"to_wire_messages() → stream()"| GW
    LOOP -->|"content deltas"| DIALECT
    DIALECT -->|"healed ToolCalls"| LOOP
    LOOP -->|"truncate_history()"| CTX
    LOOP -->|"build_system_prompt(digest, ctx)"| PROMPTS
    PROMPTS --> AGENTS
    MEM -->|"memory digest"| PROMPTS
    MEM -->|"estimate_tokens"| CTX
    GW -->|"ToolCall / wire shape"| MSG
    DIALECT -->|"ToolCall"| MSG
    GW -->|"wire body: no tool_choice / temperature / stream_options / store"| API
    API -->|"SSE StreamEvent stream"| GW

    LOOP -->|"execute(tool_call)"| REG
    REG --> BASH
    REG --> FS
    REG --> MAP
    MAP --> AM

    LOOP -->|"append tool msg / final answer"| SESS
    SESS -->|"resume: replay history"| LOOP
    LOOP -->|"AgentEvent: content / reasoning / tool_start / tool_result / done / error"| TUI
```

## 2. One headless run — sequence

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant CLI as harness/cli.py
    participant LOOP as harness/loop.py (AgentLoop)
    participant GW as harness/gateway.py (SSE)
    participant DL as harness/dialect.py (DialectFeed)
    participant TOOLS as harness/tools.py (ToolRegistry)
    participant SESS as harness/sessions.py (JSONL)

    U->>CLI: kaal run "PROMPT" [--dir, --model, --max-steps, --resume]
    CLI->>CLI: resolve API key (env OPENCODE_API_KEY → ~/.omp/agent/agent.db)
    CLI->>LOOP: AgentLoop(Gateway, Memory, ToolRegistry).run(prompt, emit)

    loop per turn (≤ max-steps)
        LOOP->>GW: stream(messages, tools)
        GW-->>LOOP: StreamEvent (content / reasoning / tool_call / done)
        LOOP->>DL: feed content deltas → heal DSML envelope
        LOOP->>LOOP: structured tool_calls win over healed ones
        alt tool calls present
            LOOP->>TOOLS: execute(call) — path-confinement + DENY list
            TOOLS-->>LOOP: result string (≤10k chars, truncation suffix)
            LOOP->>SESS: append tool message (persist)
            LOOP->>GW: next turn — replay reasoning_content verbatim
        else no tool calls
            LOOP->>SESS: append final answer
            LOOP-->>CLI: done(answer)
        end
    end

    CLI-->>U: answer to stdout (or --json line), exit 0/1/2
```

## Reading the diagram

- **Two hard things** live in the dashed path `GW → DIALECT → LOOP`: DSML
  healing (fullwidth-pipe `｜` U+FF5C envelope leaked into `content`) and
  mandatory `reasoning_content` replay (dropping it → HTTP 400 on turn 2+).
- **Port seam**: everything in `ENTRY`/`RUNTIME`/`CORE` is stdlib-only;
  `textual` appears only in `harness/tui.py`. The core maps 1:1 to a Rust/Go port.
- **Never sent on the wire**: `tool_choice`, `temperature`, `stream_options`,
  `store` — the model rejects them (`harness/gateway.py::_build_body`).
- **Loop exits**: answer produced (0) · config/key/gateway error (1) ·
  loop error — max steps, context overflow, tool loop, 5 consecutive tool
  failures (2).
