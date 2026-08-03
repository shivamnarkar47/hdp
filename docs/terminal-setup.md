# Terminal setup — the intended kaal look

kaal is a terminal UI: Textual renders *characters*, and the terminal emulator
draws them. The font is therefore a terminal-side setting — kaal cannot change
it. To get the intended look (the KESHAVLOK banner, box-drawing panels, and
the tmux-style status bar), configure your emulator's font to
**Fira Sans Condensed** (or a mono fallback if alignment breaks).

## kitty

In `~/.config/kitty/kitty.conf`:

```conf
font_family Fira Sans Condensed
```

> **Mono-alignment note:** a proportional font can break the box-drawing
> alignment of the TUI's panels and status bar. If that happens, use the mono
> sibling instead — `font_family Fira Mono` — or keep the proportional font
> but force Fira Mono for the box-drawing ranges via `font_features`:
>
> ```conf
> font_features FiraSansCondensed +liga 0
> ```

## Windows Terminal

In `settings.json` (`Ctrl+,` → gear icon → "Open JSON file"), under the
profile's `"font"` object:

```json
{
    "profiles": {
        "defaults": {
            "font": {
                "face": "Fira Sans Condensed"
            }
        }
    }
}
```

(Older builds used `"fontFace": "Fira Sans Condensed"` directly on the profile
object; the `"font": {"face": ...}` form is the current schema.)

## iTerm2 (macOS)

`Settings → Profiles → Text → Font` — pick **Fira Sans Condensed**. (As in
kitty, switch to a mono face if the TUI's box drawing misaligns.)

## Checking alignment

After changing the font, restart kaal and look at the KESHAVLOK banner and the
status bar: `│` separators should line up in one vertical column across the
bar. If they don't, use the mono variant above.
