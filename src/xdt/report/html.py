"""Maquetacion del reporte: cada figura y tabla lleva dos bloques obligatorios.

* **Como leerlo** (interpretable): que representan los ejes, colores y filas.
* **Que explica** (explicable): que conclusion se sigue y por que, con las
  cifras de esta corrida, y que NO se puede concluir.

`Report.figure` y `Report.table` exigen ambos textos: una figura sin ellos no
compila, que es la forma de que la regla no dependa de la memoria de nadie.
"""

from __future__ import annotations

import base64
import html
import io
import numbers
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#5d5d58;--line:#e2e1dc;
--accent:#b8461b;--read:#eef4fb;--read-b:#2f6fb0;--why:#fdf3e7;--why-b:#b8661b;
--ok:#2e7d4f;--bad:#b3261e;--code:#f0efe9}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--card:#1f1f1d;
--ink:#ecebe6;--muted:#a3a29b;--line:#34332f;--accent:#e0794e;--read:#1b2633;--read-b:#6aa3dd;
--why:#2e2518;--why-b:#e0a060;--ok:#6cc08b;--bad:#f08a80;--code:#2a2926}}
:root[data-theme="dark"]{--bg:#161615;--card:#1f1f1d;--ink:#ecebe6;--muted:#a3a29b;--line:#34332f;
--accent:#e0794e;--read:#1b2633;--read-b:#6aa3dd;--why:#2e2518;--why-b:#e0a060;--ok:#6cc08b;
--bad:#f08a80;--code:#2a2926}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 "Source Serif 4",Georgia,serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 80px}
h1{font:700 2rem/1.2 "Inter",system-ui,sans-serif;margin:0 0 8px}
h2{font:700 1.45rem/1.3 "Inter",system-ui,sans-serif;margin:56px 0 8px;padding-top:12px;
border-top:3px solid var(--accent)}
h3{font:600 1.1rem/1.35 "Inter",system-ui,sans-serif;margin:32px 0 8px}
.sub{color:var(--muted);font-family:"Inter",system-ui,sans-serif;font-size:.92rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;
margin:18px 0}
.card img{width:100%;height:auto;background:#fff;border-radius:6px}
.box{border-left:4px solid;border-radius:6px;padding:10px 14px;margin:10px 0;font-size:.95rem}
.box .lbl{font-family:"Inter",system-ui,sans-serif;font-size:.8rem;letter-spacing:.04em;
text-transform:uppercase;display:block;margin-bottom:2px}
.read{background:var(--read);border-color:var(--read-b)}.why{background:var(--why);border-color:var(--why-b)}
.tbl{overflow-x:auto}
table{border-collapse:collapse;width:100%;font:13.5px/1.4 "Inter",system-ui,sans-serif}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:right;vertical-align:top}
th{background:var(--code);position:sticky;top:0}
td:first-child,th:first-child{text-align:left}
td.txt,th.txt{text-align:left}
nav{font-family:"Inter",system-ui,sans-serif;font-size:.9rem;columns:2;column-gap:32px}
nav a{color:var(--ink)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.kpi .v{font:700 1.6rem/1.1 "Inter",system-ui,sans-serif}
.kpi .l{color:var(--muted);font:13px/1.3 "Inter",system-ui,sans-serif}
.verdict{border:2px solid var(--ok);border-radius:10px;padding:14px 18px;background:var(--card)}
code{background:var(--code);padding:1px 5px;border-radius:4px;font-size:.88em}
@media (max-width:640px){nav{columns:1}h1{font-size:1.5rem}}
"""


@dataclass
class Report:
    title: str
    subtitle: str
    parts: list[str] = field(default_factory=list)
    toc: list[tuple[str, str]] = field(default_factory=list)
    _n_fig: int = 0
    _n_tab: int = 0

    def section(self, title: str, intro: str = "") -> None:
        anchor = f"s{len(self.toc) + 1}"
        self.toc.append((anchor, title))
        self.parts.append(f'<h2 id="{anchor}">{html.escape(title)}</h2>')
        if intro:
            self.parts.append(f"<p>{intro}</p>")

    def h3(self, title: str) -> None:
        self.parts.append(f"<h3>{html.escape(title)}</h3>")

    def p(self, text_html: str) -> None:
        self.parts.append(f"<p>{text_html}</p>")

    def raw(self, fragment: str) -> None:
        self.parts.append(fragment)

    def kpis(self, items: list[tuple[str, str]]) -> None:
        cells = "".join(
            f'<div class="kpi"><div class="v">{html.escape(v)}</div>'
            f'<div class="l">{html.escape(label)}</div></div>'
            for v, label in items
        )
        self.parts.append(f'<div class="kpis">{cells}</div>')

    @staticmethod
    def _boxes(como_leer: str, que_explica: str) -> str:
        if not como_leer.strip() or not que_explica.strip():
            raise ValueError("toda figura/tabla necesita 'como leerlo' y 'que explica'")
        return (f'<div class="box read"><span class="lbl">Cómo leerlo</span>{como_leer}</div>'
                f'<div class="box why"><span class="lbl">Qué explica</span>{que_explica}</div>')

    def figure(self, fig, title: str, *, como_leer: str, que_explica: str) -> None:
        import matplotlib.pyplot as plt

        self._n_fig += 1
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode()
        self.parts.append(
            f'<div class="card"><h3>Figura {self._n_fig}. {html.escape(title)}</h3>'
            f'<img alt="{html.escape(title)}" src="data:image/png;base64,{b64}">'
            f"{self._boxes(como_leer, que_explica)}</div>"
        )

    def table(self, df: pd.DataFrame, title: str, *, como_leer: str, que_explica: str,
              floatfmt: str = "{:.3f}", index: bool = False) -> None:
        self._n_tab += 1
        d = df.reset_index() if index else df
        numeric = {c: pd.api.types.is_numeric_dtype(d[c]) for c in d.columns}
        head = "".join(
            f"<th{'' if numeric[c] else ' class=txt'}>{html.escape(str(c))}</th>"
            for c in d.columns
        )
        rows = []
        for _, r in d.iterrows():
            cells = []
            for v in r:
                if isinstance(v, (bool, np.bool_)):
                    cells.append(f"<td>{'sí' if v else 'no'}</td>")
                elif isinstance(v, numbers.Integral):
                    cells.append(f"<td>{int(v):,}</td>".replace(",", "."))
                elif isinstance(v, numbers.Real):
                    cells.append(f"<td>{'—' if pd.isna(v) else floatfmt.format(v)}</td>")
                else:
                    cells.append(f'<td class="txt">{html.escape(str(v))}</td>')
            rows.append("<tr>" + "".join(cells) + "</tr>")
        self.parts.append(
            f'<div class="card"><h3>Tabla {self._n_tab}. {html.escape(title)}</h3>'
            f'<div class="tbl"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
            f"{self._boxes(como_leer, que_explica)}</div>"
        )

    def render(self) -> str:
        nav = "".join(f'<div><a href="#{a}">{html.escape(t)}</a></div>' for a, t in self.toc)
        return (
            '<!doctype html><html lang="es"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{html.escape(self.title)}</title>"
            '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700'
            '&family=Source+Serif+4:wght@400;600&display=swap" rel="stylesheet">'
            f"<style>{CSS}</style></head><body><main>"
            f"<h1>{html.escape(self.title)}</h1><p class='sub'>{self.subtitle}</p>"
            f'<div class="card"><b>Contenido</b><nav>{nav}</nav></div>'
            + "".join(self.parts)
            + "</main></body></html>"
        )
