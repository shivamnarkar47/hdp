"""Home-screen ASCII hero for the TUI: a sea lion (side-on, facing right).

Stdlib-only; printable ASCII + box-drawing characters. Every line is padded
to the same width so the art renders as a clean rectangle via
``Static(markup=False)`` and is safe to mirror into the plain-text transcript.
"""

from __future__ import annotations

_RAW = """\
      ▄▄▄  ▄▄▄
    ╭────────╮
   ╭╯ o  o  ╰╮
   ╰╮   ^   ╭╯╭────────────────────────────────╮
    │ ~ ~ ~ ~│ │                                ╰─╮
    │  ~   ~ │ │                                   ╰─╮
    ╰╮  .  ╭╯ │                                      ╰─╮
     ╰──╮╭──╯ │                                         ╰─╮
       ╰─╯    │                                            ╰─╮
              │                                               ╰─╮
              │                                                  ╰─╮
              ╰───────────────────────────────────────────────────╯
   ╭────╮     ╭────╮                      ╭──────────╮
  ╭╯    ╰──╮ ╭╯    ╰╮                   ╭╯          ╰╮
  │        ╰─╯      │                   ╭╯            ╰╮
  ╰─────────────────╯                   ╰──────────────╯
    ▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂
   ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~
   ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~
   ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~
"""


def _build() -> str:
    lines = [line.rstrip() for line in _RAW.strip("\n").splitlines()]
    width = max(len(line) for line in lines)
    return "\n".join(line.ljust(width) for line in lines)


SEA_LION = _build()
