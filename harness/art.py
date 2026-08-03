"""ASCII hero art for the TUI.

Exports, all stdlib-only and printable:

* ``KAAL_LOGO`` — the figlet-style KAAL wordmark, the home-screen hero.
* ``MAHABHARATA_ART`` — a simple Panchajanya conch, the Mahabharata mark
  shown beneath the wordmark on the home screen.
* ``SEA_LION`` — the original home-screen hero (a sea lion, side-on, facing
  right). Kept for tests and anyone who wants it; the home screen now shows
  the KAAL wordmark + conch instead.
* ``BANNER_TITLE`` / ``BANNER_TAGLINE`` — the KESHAVLOK home banner.
  KESHAVLOK (Keshav + lok, "Keshav's world") is the brand term: Keshav —
  Krishna's name — was the mastermind of the Mahabharata, and the harness is
  the mastermind orchestrating its five Pandava agents.

Every art block: every line is padded to the same width so it renders as a
clean rectangle via ``Static(markup=False)`` and is safe to mirror into the
plain-text transcript.
"""

from __future__ import annotations

# KESHAVLOK = Keshav + lok ("Keshav's world"). Keshav (Krishna) masterminded
# the Mahabharata from behind the scenes; the harness is the same kind of
# mastermind orchestrating the five Pandava agents.
BANNER_TITLE = "KESHAVLOK"
BANNER_TAGLINE = (
    "Keshav's world — the Mahabharata of the AI world: "
    "five Pandava agents, one mastermind"
)

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


_KAAL_RAW = """\
____  __.            .__                .____           __
|    |/ _|____   _____|  |__ _____ ___  _|    |    ____ |  | __
|      <_/ __ \\ /  ___/  |  \\\\__  \\\\  \\/ /    |   /  _ \\|  |/ /
|    |  \\  ___/ \\___ \\|   Y  \\/ __ \\\\   /|    |__(  <_> )    <
|____|__ \\___  >____  >___|  (____  /\\_/ |_______ \\____/|__|_ \\
        \\/   \\/     \\/     \\/             \\/          \\/
"""

_MAHABHARATA_RAW = """\
        ▄▄▄
      ▄█   █▄
     █       █
    █  ▄▄▄▄▄  █
    █ █     █ █
    █ █     █ █
     █ █▄▄▄█ █
      ██   ██
        █████
"""


def _build(raw: str) -> str:
    lines = [line.rstrip() for line in raw.strip("\n").splitlines()]
    width = max(len(line) for line in lines)
    return "\n".join(line.ljust(width) for line in lines)


SEA_LION = _build(_RAW)
KAAL_LOGO = _build(_KAAL_RAW)
MAHABHARATA_ART = _build(_MAHABHARATA_RAW)
