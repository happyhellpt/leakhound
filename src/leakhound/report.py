"""Findings and report rendering for LeakHound."""
from __future__ import annotations

import html
import sys
from dataclasses import dataclass, field
from typing import Any

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "ok": 3}
_ICON_UNICODE = {"high": "✗", "medium": "!", "low": "·", "ok": "✓"}
_ICON_ASCII = {"high": "[X]", "medium": "[!]", "low": "[.]", "ok": "[OK]"}
_COLOR = {"high": "#d64545", "medium": "#d98a00", "low": "#6b7280", "ok": "#2f9e44"}


def _stdout_supports_unicode() -> bool:
    """True only if the current stdout can actually encode our symbols."""
    enc = getattr(sys.stdout, "encoding", None)
    if not enc:
        return False
    try:
        "✓✗─·→".encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


@dataclass
class Finding:
    check: str
    severity: str  # "high" | "medium" | "low" | "ok"
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    fix: str | None = None

    @property
    def is_leak(self) -> bool:
        return self.severity in ("high", "medium")


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    @property
    def leaks(self) -> list[Finding]:
        return [f for f in self.findings if f.is_leak]

    @property
    def clean(self) -> bool:
        return len(self.leaks) == 0

    def sorted(self) -> list[Finding]:
        return sorted(self.findings, key=lambda f: SEVERITY_ORDER.get(f.severity, 9))

    def render(self, ascii: bool | None = None) -> str:
        """Render the report to text. ascii=None auto-detects terminal support."""
        use_ascii = (not _stdout_supports_unicode()) if ascii is None else ascii
        icon = _ICON_ASCII if use_ascii else _ICON_UNICODE
        rule = "-" if use_ascii else "─"
        arrow = "->" if use_ascii else "→"

        lines = ["", "  LeakHound report", "  " + rule * 48]
        for f in self.sorted():
            lines.append(f"  {icon.get(f.severity, '?')} [{f.severity:<6}] {f.check}: {f.message}")
            for k, v in f.evidence.items():
                lines.append(f"        {k}: {v}")
            if f.fix:
                lines.append(f"        {arrow} fix: {f.fix}")
        lines.append("  " + rule * 48)
        if self.clean:
            lines.append("  No leakage detected. (Absence of a finding is not proof of a clean split.)")
        else:
            n = len(self.leaks)
            lines.append(f"  {n} likely leak{'s' if n != 1 else ''} found. "
                         f"Your reported metric is probably optimistic.")
        lines.append("")
        out = "\n".join(lines)
        if use_ascii:
            for uni, rep in (("—", "--"), ("–", "-"), ("→", "->"), ("’", "'"),
                             ("“", '"'), ("”", '"'), ("·", ".")):
                out = out.replace(uni, rep)
            out = out.encode("ascii", "replace").decode("ascii")
        return out

    def to_html(self, title: str = "LeakHound report") -> str:
        """A self-contained, shareable HTML report (no external assets)."""
        def esc(x: Any) -> str:
            return html.escape(str(x))

        if self.clean:
            banner_bg, banner = "#2f9e44", "No leakage detected by these checks."
        else:
            n = len(self.leaks)
            banner_bg = "#d64545"
            banner = (f"{n} likely leak{'s' if n != 1 else ''} found "
                      f"— your reported metric is probably optimistic.")

        cards = []
        for f in self.sorted():
            ev = "".join(
                f'<div class="ev"><span class="k">{esc(k)}</span>'
                f'<span class="v">{esc(v)}</span></div>'
                for k, v in f.evidence.items()
            )
            fix = f'<div class="fix">→ fix: {esc(f.fix)}</div>' if f.fix else ""
            cards.append(f"""
      <div class="card" style="border-left-color:{_COLOR.get(f.severity, '#888')}">
        <div class="head">
          <span class="sev" style="background:{_COLOR.get(f.severity, '#888')}">{esc(f.severity)}</span>
          <span class="check">{esc(f.check)}</span>
        </div>
        <div class="msg">{esc(f.message)}</div>
        {ev}
        {fix}
      </div>""")

        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.5 -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         margin: 0; background: #f6f7f9; color: #1b1f24; }}
  .wrap {{ max-width: 820px; margin: 0 auto; padding: 32px 20px 60px; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .sub {{ color: #6b7280; margin: 0 0 20px; }}
  .banner {{ color: #fff; background: {banner_bg}; padding: 14px 18px;
            border-radius: 10px; font-weight: 600; margin-bottom: 22px; }}
  .card {{ background: #fff; border: 1px solid #e5e7eb; border-left: 5px solid #888;
          border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; }}
  .head {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }}
  .sev {{ color: #fff; font-size: 11px; font-weight: 700; text-transform: uppercase;
         letter-spacing: .04em; padding: 2px 8px; border-radius: 999px; }}
  .check {{ font-weight: 600; }}
  .msg {{ margin-bottom: 8px; }}
  .ev {{ display: flex; gap: 8px; font: 12.5px/1.5 ui-monospace, Menlo, Consolas, monospace;
        color: #374151; }}
  .ev .k {{ color: #9aa0a6; min-width: 150px; }}
  .fix {{ margin-top: 8px; font-size: 13px; color: #0b7285; font-weight: 600; }}
  footer {{ margin-top: 26px; color: #9aa0a6; font-size: 12px; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #0f1115; color: #e6e6e6; }}
    .card {{ background: #181b20; border-color: #2a2f37; }}
    .sub, .ev {{ color: #aab; }} .ev .k {{ color: #6b7280; }}
  }}
</style></head>
<body><div class="wrap">
  <h1>🐶 {esc(title)}</h1>
  <p class="sub">Data-leakage audit of your train/test split.</p>
  <div class="banner">{esc(banner)}</div>
  {''.join(cards)}
  <footer>A clean report is not proof of a clean split — it means these common,
  high-impact leaks are absent. Generated by LeakHound.</footer>
</div></body></html>"""

    def __str__(self) -> str:  # pragma: no cover
        return self.render()
