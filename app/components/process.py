"""The home page's process picture: one animated strip from the coater to the decision."""

from __future__ import annotations

from coatshield import style

STAGES = (
    ("1 · Coater", "Pellets circulate; a window sees a sliver of the batch"),
    ("2 · Camera and gate", "Single pellets pass; fused twins are held back"),
    ("3 · OCT scan", "Light finds the two surfaces of the coat"),
    ("4 · Thickness", "Index found, not assumed; refraction corrected"),
    ("5 · Correct the sample", "Undo the bias toward big pellets; read d10"),
    ("6 · Operator decides", "The system recommends; a person signs"),
)
_W, _GAP, _BOX = 150, 34, 150


def _bell(cx: float, base: float, height: float, half: float) -> str:
    return (f"M{cx - 2 * half:.0f} {base} C{cx - half:.0f} {base} {cx - half:.0f} {base - height} "
            f"{cx:.0f} {base - height} C{cx + half:.0f} {base - height} {cx + half:.0f} {base} "
            f"{cx + 2 * half:.0f} {base}")


def process_html() -> str:
    """Self-contained HTML (inline SVG, no scripts, no network) for st.components.html."""
    ink, muted, grid = style.TEXT, style.TEXT_SECONDARY, style.GRID
    blue, orange, yellow, grey = style.BLUE, style.ORANGE, style.YELLOW, style.NEUTRAL
    loop = "M75 120 L75 52 C75 30 118 30 118 60 L118 112 C118 132 75 140 75 120 Z"
    loop_left = "M75 120 L75 52 C75 30 32 30 32 60 L32 112 C32 132 75 140 75 120 Z"
    pellets = "".join(
        f'<circle r="{r}" fill="{grey}"><animateMotion dur="{dur}s" begin="-{begin}s" '
        f'repeatCount="indefinite" path="{path}"/></circle>'
        for r, dur, begin, path in ((4, 5, 0, loop), (5, 6, 2, loop), (3.5, 5.5, 3.6, loop),
                                    (4.5, 5, 1, loop_left), (3.5, 6, 3, loop_left),
                                    (5, 5.5, 4.4, loop_left)))
    speckle = "".join(
        f'<circle cx="{28 + (i * 37) % 96}" cy="{104 + (i * 23) % 34}" r="1.6" fill="#b9b8b2"/>'
        for i in range(26))
    drawings = (
        # 1 coater: vessel, draft tube, window, circulating pellets
        f'<path d="M40 138 L22 112 L22 22 L128 22 L128 112 L110 138 Z" fill="none" '
        f'stroke="{muted}" stroke-width="2"/>'
        f'<path d="M60 48 V118 M90 48 V118" stroke="{muted}" stroke-width="2"/>'
        f'<rect x="124" y="62" width="8" height="30" rx="2" fill="{blue}"/>{pellets}',
        # 2 gate: a single passes, a twin is held
        f'<circle cx="46" cy="58" r="22" fill="none" stroke="{blue}" stroke-width="3"/>'
        f'<path d="M36 58 l8 8 l14 -16" fill="none" stroke="{blue}" stroke-width="3" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="92" cy="106" r="18" fill="none" stroke="{grey}" stroke-width="3"/>'
        f'<circle cx="116" cy="106" r="18" fill="none" stroke="{grey}" stroke-width="3"/>'
        f'<text x="46" y="98" text-anchor="middle" font-size="11" fill="{muted}">single</text>'
        f'<text x="104" y="78" text-anchor="middle" font-size="11" fill="{muted}">'
        f'twin: held</text>',
        # 3 OCT scan: dark image, two surfaces, speckled core, sweeping beam
        f'<rect x="20" y="24" width="110" height="118" rx="6" fill="#1f1f1e"/>{speckle}'
        f'<path d="M26 92 Q75 20 124 92" fill="none" stroke="{blue}" stroke-width="3"/>'
        f'<path d="M26 122 Q75 56 124 122" fill="none" stroke="{yellow}" stroke-width="3"/>'
        f'<rect x="22" y="24" width="2" height="118" fill="#ffffff" opacity="0.7">'
        f'<animate attributeName="x" values="22;126;22" dur="4s" repeatCount="indefinite"/></rect>',
        # 4 thickness between the two surfaces
        f'<path d="M22 84 Q75 14 128 84" fill="none" stroke="{blue}" stroke-width="3"/>'
        f'<path d="M22 124 Q75 60 128 124" fill="none" stroke="{yellow}" stroke-width="3"/>'
        f'<path d="M75 52 V88 M70 58 L75 51 L80 58 M70 82 L75 89 L80 82" fill="none" '
        f'stroke="{ink}" stroke-width="2" stroke-linecap="round"/>'
        f'<text x="75" y="38" text-anchor="middle" font-size="13" font-weight="600" '
        f'fill="{ink}">16 µm</text>',
        # 5 raw sample against corrected estimate, with the spec line
        f'<path d="M20 128 H132" stroke="{muted}" stroke-width="1.5"/>'
        f'<path d="{_bell(88, 128, 66, 17)}" fill="none" stroke="{orange}" stroke-width="3"/>'
        f'<path d="{_bell(68, 128, 78, 17)}" fill="none" stroke="{blue}" stroke-width="3"/>'
        f'<path d="M52 34 V134" stroke="{ink}" stroke-width="1.5" stroke-dasharray="5 4"/>'
        f'<text x="48" y="30" text-anchor="end" font-size="11" fill="{muted}">spec</text>'
        f'<text x="118" y="56" text-anchor="middle" font-size="11" fill="{muted}">raw</text>',
        # 6 operator signs
        f'<circle cx="75" cy="52" r="17" fill="none" stroke="{muted}" stroke-width="3"/>'
        f'<path d="M42 122 C42 92 108 92 108 122" fill="none" stroke="{muted}" stroke-width="3"/>'
        f'<circle cx="112" cy="104" r="17" fill="{blue}"/>'
        f'<path d="M104 104 l6 6 l10 -12" fill="none" stroke="#ffffff" stroke-width="3" '
        f'stroke-linecap="round" stroke-linejoin="round"/>',
    )
    parts = []
    for i, ((title, text), drawing) in enumerate(zip(STAGES, drawings, strict=True)):
        x = 8 + i * (_W + _GAP)
        words, lines, line = text.split(), [], ""
        for word in words:  # wrap the caption to the box width
            if len(line) + len(word) > 24:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        lines.append(line)
        caption = "".join(f'<text x="{_W / 2}" y="{196 + 15 * j}" text-anchor="middle" '
                          f'font-size="11.5" fill="{muted}">{ln}</text>'
                          for j, ln in enumerate(lines))
        arrow = ""
        if i < len(STAGES) - 1:
            arrow = (f'<path d="M{_W + 6} 75 H{_W + _GAP - 8} m-7 -6 l7 6 l-7 6" fill="none" '
                     f'stroke="{muted}" stroke-width="2" stroke-linecap="round" '
                     f'stroke-linejoin="round"/>')
        parts.append(
            f'<g transform="translate({x} 6)">'
            f'<rect width="{_W}" height="{_BOX}" rx="12" fill="{style.SURFACE}" stroke="{grid}" '
            f'stroke-width="1.5"/>{drawing}{arrow}'
            f'<text x="{_W / 2}" y="176" text-anchor="middle" font-size="13.5" font-weight="650" '
            f'fill="{ink}">{title}</text>{caption}</g>')
    width = 16 + len(STAGES) * _W + (len(STAGES) - 1) * _GAP
    return (
        f'<div style="background:{style.SURFACE};font-family:system-ui,-apple-system,'
        f'Segoe UI,sans-serif;overflow-x:auto">'
        f'<svg viewBox="0 0 {width} 250" width="100%" style="min-width:760px" role="img" '
        f'aria-label="From the coater to the operator\'s decision in six steps">'
        f'{"".join(parts)}</svg></div>')
