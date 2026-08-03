# Decisions

## 2026-08-04 — rename kala → kaal

Full product rename (third name, after hdp → kala): CLI binary, package name, `.kaal/` cache dir, `KAAL_*` env vars, `~/.config/kaal` + `~/.local/share/kaal` paths, TUI title/labels, system prompt identity, installer defaults, README/docs, and GitHub URLs (`shivamnarkar47/kaal`). Mechanical string rename; zero API/schema changes; 249 tests green. Follow-ups: rename the GitHub repo, re-run install.sh, optionally delete stale `.venv/bin/kala`/`hdp` binaries.

