"""ASCII hero art for the TUI.

Three exports, all stdlib-only and printable:

* ``SEA_LION`` — the original home-screen hero (a sea lion, side-on, facing
  right). Kept for tests and anyone who wants it.
* ``CHARIOT_WHEEL`` — the home-screen hero: Krishna's chariot wheel, the
  KESHAVLOK symbol. A felloe, a hub, and eight spokes radiating
  chakra-style, with rivets on the rim. Mounted above the KESHAVLOK banner.
* ``BANNER_TITLE`` / ``BANNER_TAGLINE`` — the KESHAVLOK home banner.
  KESHAVLOK (Keshav + lok, "Keshav's world") is the brand term: Keshav —
  Krishna's name — was the mastermind of the Mahabharata, and the harness is
  the mastermind orchestrating its five Pandava agents. The banner renders as
  plain text (no figlet dependency): the title in accent/bold, the tagline
  dim below.

The art: every line is padded to the same width so the art renders as a clean
rectangle via ``Static(markup=False)`` and is safe to mirror into the
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


def _build(raw: str) -> str:
    lines = [line.rstrip() for line in raw.strip("\n").splitlines()]
    width = max(len(line) for line in lines)
    return "\n".join(line.ljust(width) for line in lines)


SEA_LION = _build(_RAW)

# Krishna's chariot wheel — the KESHAVLOK symbol: a felloe (double rim), a hub,
# and eight spokes radiating chakra-style, with rivets on the rim. Plain
# ASCII + box-drawing only; 21 lines, 52 columns, padded to a rectangle by
# _build, so it renders as a clean block and mirrors into the transcript.
_WHEEL_RAW = r"""
                    ---·-------·---
               -----       -       -----
            ---  ----------|----------  ---
         -·-  ---   \      |      /   ---  -·-
       /-  ---       \     |     /       ---  -\
      /  //           \    |    /           \\  \
     /  /              \   |   /              \  \
    /  /                \  |  /                \  \
   |  |                  \ | /                  |  |
   |  |                   \|/                   |  |
   |-|---------------------╬---------------------|-|
   |  |                   /|\                   |  |
   |  |                  / | \                  |  |
    \  \                /  |  \                /  /
     \  \              /   |   \              /  /
      \  \\           /    |    \           //  /
       \-  ---       /     |     \       ---  -/
         -·-  ---   /      |      \   ---  -·-
            ---  ----------|----------  ---
               -----       -       -----
                    ---·-------·---
"""

CHARIOT_WHEEL = _build(_WHEEL_RAW)
