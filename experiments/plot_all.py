"""Publication figures for experiments E1-E5 (RGCV paper).

Style: English labels, serif (STIX/Times), Okabe-Ito colorblind-safe palette,
PNG (300 dpi) + PDF vector output.  Run with base anaconda python (matplotlib 3.7).
"""
import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]   # repo root
ROOT = BASE / "figures"                      # all figure output lands here
RGCV_RES = BASE / "results"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlepad": 9,
    "axes.labelsize": 11, "axes.labelweight": "bold", "axes.labelcolor": "#222222",
    "legend.fontsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.color": "#333333", "ytick.color": "#333333",
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#C9C9C9", "grid.alpha": 0.4, "grid.linestyle": "-", "grid.linewidth": 0.55,
    "hatch.linewidth": 0.5,
    "svg.fonttype": "none",
    "figure.dpi": 110, "figure.facecolor": "white",
    "savefig.bbox": "tight",
})

EDGE = "#333333"          # unified bar/patch edge
BAR_LW = 0.5              # unified bar outline width
LBL_FS = 6.8              # rotated in-chart value labels
LBL_C = "#222222"

# Okabe-Ito colorblind-safe
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verm": "#D55E00",
      "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "black": "#000000"}


PAPER_FIGS = BASE / "figures" / "paper_figs"
HTML_PATH = BASE / "figures" / "interactive_figures.html"


def _gid(s: str) -> str:
    """XML-safe element id from an arbitrary label."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s)


def _inject_title(svg: str, gid: str, tip: str) -> str:
    """Insert a native SVG <title> tooltip as the first child of the element
    carrying `gid`.  Handles self-closing (<path/>) and open (<g>) forms."""
    t = tip.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for tag in ("g", "path", "use", "rect", "circle"):
        pat_self = re.compile(rf'(<{tag}\b[^>]*?id="{re.escape(gid)}"[^>]*?)/>')
        if pat_self.search(svg):
            return pat_self.sub(rf'\1><title>{t}</title></{tag}>', svg, count=1)
        pat_open = re.compile(rf'(<{tag}\b[^>]*?id="{re.escape(gid)}"[^>]*?>)')
        if pat_open.search(svg):
            return pat_open.sub(rf'\1<title>{t}</title>', svg, count=1)
    return svg


def save(fig, out_dir: Path, name: str, tips: dict | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300)
    fig.savefig(out_dir / f"{name}.pdf")
    svg_path = out_dir / f"{name}.svg"
    fig.savefig(svg_path)
    if tips:  # embed per-element hover tooltips into the SVG
        svg = svg_path.read_text(encoding="utf-8")
        for gid, tip in tips.items():
            svg = _inject_title(svg, gid, tip)
        svg_path.write_text(svg, encoding="utf-8")
    plt.close(fig)
    print(f"saved {out_dir / name}.png/.pdf/.svg")


def load(p):
    return json.load(open(p, encoding="utf-8"))


# ============================================================ E1 retrieval
def fig_e1():
    d = load(ROOT / "E1_retrieval" / "e1_retrieval.json")
    summary, order = d["summary"], ["full", "sg_rag", "no_rerank", "no_2stage", "no_vsig", "flat_bm25"]
    labels = {"full": "Full RGCV-SR", "sg_rag": "SG-RAG (ours)", "no_rerank": "w/o Rerank",
              "no_2stage": "w/o 2-stage", "no_vsig": "w/o Value-Sig", "flat_bm25": "Flat BM25"}
    colors = [OI["black"], OI["blue"], OI["orange"], OI["green"], OI["purple"], OI["verm"]]
    datasets = [("bird", "BIRD (n=80)"), ("spider2lite", "Spider2.0-Lite (n=69)"), ("spider2lite_grast", "GRAST (n=233)")]

    hatches = ["", "//", ".."]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 3.6))
    x = np.arange(len(order))
    w = 0.26
    tips = {}
    for ax, metric, ylab, logy in [
        (axes[0], "table_F1", "Table-level F1", False),
        (axes[1], "tokens", "Avg. schema tokens", True),
    ]:
        hi_all = 0.0
        for i, (ds, dsname) in enumerate(datasets):
            means = [summary[ds][c][metric]["mean"] for c in order]
            cis = [summary[ds][c][metric]["ci95"] for c in order]
            hi_all = max(hi_all, max(ci[1] for ci in cis))
            err = [[m - c[0] for m, c in zip(means, cis)], [c[1] - m for m, c in zip(means, cis)]]
            bb = ax.bar(x + (i - 1) * w, means, w, yerr=err, capsize=2.5,
                        color=colors[i], hatch=hatches[i], edgecolor=EDGE, linewidth=BAR_LW,
                        label=dsname if metric == "table_F1" else None,
                        error_kw=dict(lw=0.8, ecolor="#444444"))
            for rect, c, m, ci in zip(bb, order, means, cis):
                gid = _gid(f"e1-{ds}-{c}-{metric}")
                rect.set_gid(gid)
                val = f"{m:,.3f}" if metric == "table_F1" else f"{m:,.0f}"
                tips[gid] = (f"{labels[c]} — {dsname}<br>{ylab}: {val}"
                             f"<br>95% CI [{ci[0]:,.3f}, {ci[1]:,.3f}]")
            # rotated value labels for anchor configurations only
            for c in ("sg_rag", "flat_bm25"):
                j = order.index(c)
                m, cu = means[j], cis[j][1]
                ytop = cu * 1.10 if logy else cu + 0.012
                txt = f"{m:.3f}" if metric == "table_F1" else f"{m:,.0f}"
                ax.text(x[j] + (i - 1) * w, ytop, txt, ha="center", va="bottom",
                        rotation=90, fontsize=LBL_FS, color=LBL_C)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[c].replace(" ", "\n") for c in order], fontsize=8.5)
        ax.set_ylabel(ylab)
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(top=hi_all * 1.5)
        else:
            ax.set_ylim(top=hi_all + 0.09)
    axes[0].set_title("(a) Schema retrieval quality")
    axes[1].set_title("(b) Schema token cost")
    axes[0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.suptitle("")

    # (c) E-G edge-type ablation: table recall of SG variants vs flat baseline
    dG = load(RGCV_RES / "eG_edge_ablation.json")["summary"]
    ax = axes[2]
    g_order = ["flat_bm25", "no_hetero", "no_cooccur", "no_sim", "sg_rag"]
    g_labels = {"flat_bm25": "Flat\nBM25", "no_hetero": "SG w/o\nhetero",
                "no_cooccur": "SG w/o\nco-occur", "no_sim": "SG w/o\nsim",
                "sg_rag": "Full\nSG-RAG"}
    gx = np.arange(len(g_order))
    for i, (ds, dsname) in enumerate(datasets):
        means = [dG[ds][c]["table_R"]["mean"] for c in g_order]
        cis = [dG[ds][c]["table_R"]["ci95"] for c in g_order]
        err = [[m - c[0] for m, c in zip(means, cis)], [c[1] - m for m, c in zip(means, cis)]]
        bb = ax.bar(gx + (i - 1) * w, means, w, yerr=err, capsize=2.5,
                    color=colors[i], hatch=hatches[i], edgecolor=EDGE, linewidth=BAR_LW,
                    error_kw=dict(lw=0.8, ecolor="#444444"))
        for rect, c, m, ci in zip(bb, g_order, means, cis):
            gid = _gid(f"e1eg-{ds}-{c}-table_R"); rect.set_gid(gid)
            tips[gid] = (f"{g_labels[c].replace(chr(10), ' ')} — {dsname}"
                         f"<br>Table recall: {m:.3f}"
                         f"<br>95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
        for c in ("sg_rag", "flat_bm25"):
            j = g_order.index(c)
            ax.text(gx[j] + (i - 1) * w, cis[j][1] + 0.010, f"{means[j]:.3f}",
                    ha="center", va="bottom", rotation=90, fontsize=LBL_FS, color=LBL_C)
    ax.axhline(1.0, color="#888888", lw=0.7, ls=":")
    ax.set_xticks(gx)
    ax.set_xticklabels([g_labels[c] for c in g_order], fontsize=8)
    ax.set_ylabel("Table-level recall")
    ax.set_ylim(0.70, 1.12)
    ax.set_title("(c) Edge-type ablation (E-G)")
    ax.text(0.02, 0.995, "all SG variants: identical coverage ($\\Delta$ = 0.0)\n"
                         "mined join paths/q: 9.64 (SG) vs 2.36 (w/o hetero)",
            transform=ax.transAxes, fontsize=7.2, style="italic", va="top", color="#444444")
    save(fig, ROOT / "E1_retrieval" / "figs", "fig_e1_retrieval_metrics", tips=tips)


# ============================================================ E2 repair
def fig_e2():
    d = load(ROOT / "E2_repair" / "e2_correction_bird.json")
    st = d["summary_total"]
    strat_order = ["none", "regen", "S1", "S1+S3", "S1+S3+S2"]
    strat_labels = ["No repair", "Naive regen", "S1 (schema)", "S1+S3 (exec)", "S1+S3+S2 (full RGCV)"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    tips = {}
    # (a) strategy ladder
    ax = axes[0]
    ex = [st[s]["EX_strict"] * 100 for s in strat_order]
    exk = [st[s]["EX_known"] * 100 for s in strat_order]
    execr = [st[s]["exec_ok_rate"] * 100 for s in strat_order]
    x = np.arange(len(strat_order))
    b1 = ax.bar(x - 0.2, ex, 0.38, color=OI["blue"], edgecolor=EDGE, linewidth=BAR_LW,
                label="EX$_{strict}$ (all)")
    b2 = ax.bar(x + 0.2, exk, 0.38, color=OI["sky"], hatch="//", edgecolor=EDGE, linewidth=BAR_LW,
                label="EX (known-answer subset)")
    for b, sname, v in zip(b1, strat_order, ex):
        gid = _gid(f"e2a-{sname}-strict"); b.set_gid(gid)
        tips[gid] = f"{sname}: EX_strict {v:.1f}%"
    for b, sname, v in zip(b2, strat_order, exk):
        gid = _gid(f"e2a-{sname}-known"); b.set_gid(gid)
        tips[gid] = f"{sname}: EX_known {v:.1f}%"
    for xi, v in zip(x, ex):
        ax.text(xi - 0.2, v + 0.8, f"{v:.1f}", ha="center", fontsize=8)
    for xi, v in zip(x, exk):
        ax.text(xi + 0.2, v + 0.8, f"{v:.1f}", ha="center", fontsize=8)
    ax.plot(x, execr, "o--", color=OI["verm"], lw=1.2, ms=5, label="Executability")
    for xi, v in zip(x, execr):
        ax.text(xi, v + 2.2, f"{v:.0f}%", ha="center", fontsize=8, color=OI["verm"])
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" ", "\n") for l in strat_labels], fontsize=8)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(a) Repair strategy ladder (304 injected cases)")
    ax.legend(frameon=False, fontsize=8)

    # (b) per-corruption: none vs best strategy
    ax = axes[1]
    corr_order = ["attribute", "relation", "value", "function", "join", "predicate_extra", "predicate_missing", "group_missing"]
    corr_labels = ["Attribute", "Relation", "Value", "Function", "Join", "Pred-extra", "Pred-missing", "Group-missing"]
    bc = d["summary_by_corruption"]
    series = [("none", "No repair", OI["verm"], ""), ("S1", "S1 only", OI["orange"], "//"),
              ("S1+S3+S2", "Full RGCV repair", OI["blue"], "xx")]
    x = np.arange(len(corr_order))
    for i, (key, lbl, color, ht) in enumerate(series):
        vals = [bc[c][key]["EX"] * 100 for c in corr_order]
        bb = ax.bar(x + (i - 1) * 0.27, vals, 0.26, color=color, hatch=ht,
                    edgecolor=EDGE, linewidth=BAR_LW, label=lbl)
        for rect, c, v in zip(bb, corr_order, vals):
            gid = _gid(f"e2b-{c}-{key}"); rect.set_gid(gid)
            tips[gid] = f"{c} — {key}: EX {v:.1f}%"
        for xi, v in zip(x + (i - 1) * 0.27, vals):
            ax.text(xi, v + 1.2, f"{v:.0f}", ha="center", va="bottom", rotation=90,
                    fontsize=6.5, color=LBL_C)
    ax.set_xticks(x)
    ax.set_xticklabels(corr_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("EX (%)")
    ax.set_ylim(0, 108)
    ax.set_title("(b) Repair gain by corruption type")
    ax.legend(frameon=False, fontsize=8)
    save(fig, ROOT / "E2_repair" / "figs", "fig_e2_repair", tips=tips)


# ============================================================ E3 verification
def fig_e3():
    d = load(ROOT / "E3_validation" / "e3_verification_bird.json")
    s = d["summary"]
    kinds = [("correct", "Correct SQL (false-alarm rate)", OI["green"], ""),
             ("silent", "Silent errors (detection rate)", OI["verm"], "//"),
             ("hard", "Hard errors (detection rate)", OI["purple"], "xx")]
    layers = ["V1", "V1+V2", "V1+V2+V3"]
    layer_labels = ["V1\n(exec)", "V1+V2\n(+consistency)", "V1+V2+V3\n(+value audit)"]

    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    x = np.arange(len(layers))
    w = 0.25
    tips = {}
    for i, (k, klabel, color, ht) in enumerate(kinds):
        vals = [s[k][l]["flag_rate"] * 100 for l in layers]
        bb = ax.bar(x + (i - 1) * w, vals, w, color=color, hatch=ht,
                    edgecolor=EDGE, linewidth=BAR_LW, label=klabel)
        for rect, l, v in zip(bb, layers, vals):
            gid = _gid(f"e3-{k}-{l}"); rect.set_gid(gid)
            tips[gid] = f"{k} @ {l}: flag rate {v:.1f}%"
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, fontsize=8)
    ax.set_ylabel("Flag rate (%)")
    ax.set_title("Semantic verification: alarm behavior by layer (n=319)", pad=24)
    ax.legend(frameon=False, fontsize=6.2, loc="lower left", bbox_to_anchor=(0, 1.01),
              ncol=3, columnspacing=0.8, handletextpad=0.4)
    ax.set_ylim(0, 112)
    save(fig, ROOT / "E3_validation" / "figs", "fig_e3_verification_layers", tips=tips)


# ============================================================ E4 pipeline
def fig_e4():
    d = load(ROOT / "E4_pipeline" / "e4_pipeline_bird.json")
    s = d["summary"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    tips = {}
    # (a) budget ladder
    ax = axes[0]
    budgets = ["B1000_full", "B2000_full", "B4000_full", "B8000_full", "B32000_full"]
    blabels = ["1K", "2K", "4K", "8K", "32K"]
    ex = [s[b]["EX_strict"] * 100 for b in budgets]
    tok = [s[b]["tokens_mean"] for b in budgets]
    x = np.arange(len(budgets))
    bb = ax.bar(x, ex, 0.55, color=OI["blue"], edgecolor=EDGE, linewidth=BAR_LW, label="EX$_{strict}$")
    for rect, bl, v in zip(bb, blabels, ex):
        gid = _gid(f"e4a-{bl}-ex"); rect.set_gid(gid)
        tips[gid] = f"budget {bl}: EX {v:.1f}%"
    for xi, v in zip(x, ex):
        ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=8.5)
    ax2 = ax.twinx()
    ax2.plot(x, tok, "--", color=OI["verm"], lw=1.0, zorder=2)
    ax2.plot([], [], "o", color=OI["verm"], ms=5.5, label="Actual tokens used")
    for xi, t, bl, e in zip(x, tok, blabels, ex):
        ln, = ax2.plot([xi], [t], "o", color=OI["verm"], ms=5.5, mec="white", mew=0.7, zorder=3)
        gid = _gid(f"e4a-{bl}-tok"); ln.set_gid(gid)
        tips[gid] = f"budget {bl}: tokens {t:,.0f} (EX {e:.1f}%)"
    for xi, v in zip(x, tok):
        ax2.text(xi, v - 210, f"{v:,.0f}", fontsize=8, color=OI["verm"], ha="center")
    ax2.set_ylabel("Actual schema tokens used", color=OI["verm"])
    ax2.tick_params(axis="y", colors=OI["verm"])
    ax2.set_ylim(0, max(tok) * 1.45)
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax.set_xticks(x)
    ax.set_xticklabels(blabels)
    ax.set_xlabel("Schema token budget")
    ax.set_ylabel("EX$_{strict}$ (%)")
    ax.set_ylim(0, 40)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.97))
    ax.set_title("(a) EX across schema budgets (full pipeline)")

    # (b) ablation + Pareto inset
    ax = axes[1]
    abl = ["B8000_open", "B8000_rgc", "B8000_full"]
    alb_labels = ["Open-loop\n(no repair/verify)", "RGCV repair\n(verify off)", "RGCV full\n(repair+verify)"]
    exa = [s[b]["EX_strict"] * 100 for b in abl]
    execa = [s[b]["exec_ok_rate"] * 100 for b in abl]
    x = np.arange(3)
    bb1 = ax.bar(x - 0.2, exa, 0.38, color=OI["blue"], edgecolor=EDGE, linewidth=BAR_LW, label="EX$_{strict}$")
    bb2 = ax.bar(x + 0.2, execa, 0.38, color=OI["sky"], hatch="//", edgecolor=EDGE, linewidth=BAR_LW, label="Executability")
    for rect, bl, v in zip(bb1, abl, exa):
        gid = _gid(f"e4b-{bl}-ex"); rect.set_gid(gid)
        tips[gid] = f"{bl}: EX {v:.1f}%"
    for rect, bl, v in zip(bb2, abl, execa):
        gid = _gid(f"e4b-{bl}-exec"); rect.set_gid(gid)
        tips[gid] = f"{bl}: executability {v:.1f}%"
    for xi, v in zip(x, exa):
        ax.text(xi - 0.2, v + 1, f"{v:.1f}", ha="center", fontsize=8.5)
    for xi, v in zip(x, execa):
        ax.text(xi + 0.2, v + 1, f"{v:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(alb_labels, fontsize=8.5)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(b) Closed-loop ablation @ 8K budget")
    ax.legend(frameon=False, fontsize=8)
    save(fig, ROOT / "E4_pipeline" / "figs", "fig_e4_pipeline", tips=tips)


# ============================================================ E5 baselines
E5 = {
    # method: (EX v4pro %, EX kimi %, prompt tok, comp tok, exec v4pro, exec kimi)
    "DAIL-SQL": (44.7, 40.0, 964_617, 971_382, 14_137, 12_613, 185, 164),
    "ReFoRCE": (58.3, 59.3, 254_269, 254_156, 15_638, 44_453, 293, 293),
    "MAC-SQL": (58.0, 57.7, 1_733_624, 1_583_540, 3_375_236, 3_567_721, 300, 295),
    "DIN-SQL": (57.0, None, 6_148_543, None, 219_260, None, 293, None),
    "CHESS": (61.0, 62.0, 19_037_440, 15_208_244, 7_053_165, 2_530_350, 298, 299),
    "SafeQL (PG17)": (47.3, 46.3, 308_848, 278_238, 626_891, 702_326, 298, 296),
    "SQLCoder-7B-2": (28.0, None, 389_287, None, 19_018, None, 194, None),
    "RGCV (Ours)": (54.7, 56.0, 310_502, 308_823, 15_934, 17_055, 290, 291),
}
E5_61 = {"MAC-SQL": 57.4, "CHESS": 52.5, "ReFoRCE": 52.5, "SafeQL (PG17)": 47.5,
         "DAIL-SQL": 47.5, "SQLCoder-7B-2": 18.0}


def fig_e5_main():
    methods = ["DAIL-SQL", "SafeQL (PG17)", "MAC-SQL", "ReFoRCE", "CHESS", "RGCV (Ours)"]
    x = np.arange(len(methods))
    v4 = [E5[m][0] for m in methods]
    km = [E5[m][1] for m in methods]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    b1 = ax.bar(x - 0.2, v4, 0.38, color=OI["blue"], edgecolor=EDGE, linewidth=BAR_LW,
                label="DeepSeek-V4-Pro (t=0)")
    b2 = ax.bar(x + 0.2, km, 0.38, color=OI["orange"], edgecolor=EDGE, linewidth=BAR_LW,
                label="Kimi-K2.6 (thinking on)")
    for b in (b1[-1], b2[-1]):          # highlight ours (RGCV, thinking-off both)
        b.set_edgecolor(OI["black"]); b.set_linewidth(1.4); b.set_hatch("//")
    tips = {}
    for b, m, v in zip(b1, methods, v4):
        gid = _gid(f"e5m-{m}-v4"); b.set_gid(gid)
        tips[gid] = (f"{m} — DeepSeek-V4-Pro: EX {v:.1f}%"
                     f"<br>prompt {E5[m][2]:,} tok / completion {E5[m][4]:,} tok")
    for b, m, v in zip(b2, methods, km):
        gid = _gid(f"e5m-{m}-km"); b.set_gid(gid)
        tips[gid] = (f"{m} — Kimi-K2.6: EX {v:.1f}%"
                     f"<br>prompt {E5[m][3]:,} tok / completion {E5[m][5]:,} tok")
    for xi, v in zip(x, v4):
        ax.text(xi - 0.2, v + 0.8, f"{v:.1f}", ha="center", fontsize=8.5)
    for xi, v in zip(x, km):
        ax.text(xi + 0.2, v + 0.8, f"{v:.1f}", ha="center", fontsize=8.5)
    # SQLCoder local reference line
    ax.axhline(E5["SQLCoder-7B-2"][0], color=OI["black"], lw=1.0, ls=":",)
    ax.text(4.42, E5["SQLCoder-7B-2"][0] + 0.7, "SQLCoder-7B-2 (local SFT): 28.0",
            ha="right", fontsize=8, style="italic")
    # DAIL-SQL on full BIRD dev (1,534 q), same backbone — subset-calibration reference (E-E)
    ax.axhline(66.1, color=OI["green"], lw=1.0, ls="--")
    ax.text(4.42, 66.8, "DAIL-SQL, full BIRD dev (1,534 q), same backbone: 66.1",
            ha="right", fontsize=8, style="italic", color=OI["green"])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("Execution accuracy EX$_{strict}$ (%)")
    ax.set_ylim(0, 72)
    ax.set_title("End-to-end baseline comparison on BIRD (300-question stratified subset)")
    from matplotlib.patches import Patch
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="white", edgecolor=OI["black"], hatch="//",
                         label="Ours"))
    ax.legend(frameon=False, loc="upper left", handles=handles)
    save(fig, ROOT / "E5_baselines" / "figs", "fig_e5_main_comparison", tips=tips)


def fig_e5_pareto():
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    bcolors = {"DeepSeek-V4-Pro": OI["blue"], "Kimi-K2.6": OI["orange"]}
    mstyle = {"DAIL-SQL": "v", "SafeQL (PG17)": "D", "MAC-SQL": "^", "DIN-SQL": "o",
              "ReFoRCE": "P", "CHESS": "X", "RGCV (Ours)": "*"}
    base_off = {"DeepSeek-V4-Pro": (7, -12), "Kimi-K2.6": (7, 6)}
    over_off = {("DAIL-SQL", "Kimi-K2.6"): (-14, -13), ("RGCV (Ours)", "Kimi-K2.6"): (10, -4),
                ("DIN-SQL", "DeepSeek-V4-Pro"): (-14, -19),
                ("MAC-SQL", "DeepSeek-V4-Pro"): (-16, -15)}
    tips = {}
    for method in ["DAIL-SQL", "SafeQL (PG17)", "MAC-SQL", "DIN-SQL", "ReFoRCE", "CHESS", "RGCV (Ours)"]:
        ex4, exk, p4, pk, c4, ck, _, _ = E5[method]
        for backbone, exv, p, c in [("DeepSeek-V4-Pro", ex4, p4, c4), ("Kimi-K2.6", exk, pk, ck)]:
            if exv is None:
                continue
            total = (p + c) / 1e6
            ours = method == "RGCV (Ours)"
            ln, = ax.plot(total, exv, marker=mstyle[method], ls="none",
                          markersize=15 if ours else 7.5, color=bcolors[backbone],
                          mec=OI["black"] if ours else "white",
                          mew=1.1 if ours else 0.8, zorder=4 if ours else 3)
            gid = _gid(f"e5p-{method}-{backbone}")
            ln.set_gid(gid)
            tips[gid] = f"{method} ({backbone})<br>EX {exv:.1f}% — {total:.2f}M tokens (300 q)"
            name = method.split(" ")[0]
            dx, dy = over_off.get((method, backbone), base_off[backbone])
            ax.annotate(f"{name} {exv:.1f}", (total, exv), textcoords="offset points",
                        xytext=(dx, dy), fontsize=8,
                        fontweight="bold" if ours else "normal")
    ax.set_xscale("log")
    ax.set_ylim(bottom=37.5)
    ax.set_xlabel("Total tokens per run, 300 questions (millions, log scale)")
    ax.set_ylabel("EX$_{strict}$ (%)")
    ax.set_title("Accuracy-cost Pareto (marker = method, color = backbone)")
    from matplotlib.lines import Line2D
    h_color = [Line2D([0], [0], marker="o", ls="none", color=bc, markersize=7, label=bb)
               for bb, bc in bcolors.items()]
    h_shape = [Line2D([0], [0], marker=ms, ls="none", color="#555555", markersize=6.5,
                      label=mm.split(" ")[0]) for mm, ms in mstyle.items()]
    leg1 = ax.legend(handles=h_color, frameon=False, loc="lower right", fontsize=8,
                     title="Backbone", title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=h_shape, frameon=False, loc="center right", ncol=2, fontsize=8)
    save(fig, ROOT / "E5_baselines" / "figs", "fig_e5_pareto", tips=tips)


def fig_e5_exec():
    methods = ["DAIL-SQL", "ReFoRCE", "MAC-SQL", "CHESS", "SafeQL (PG17)", "RGCV (Ours)", "SQLCoder-7B-2"]
    x = np.arange(len(methods))
    exec_v4 = [E5[m][6] / 3.0 for m in methods]
    exec_km = [E5[m][7] / 3.0 if E5[m][7] is not None else np.nan for m in methods]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x - 0.2, exec_v4, 0.38, color=OI["blue"], edgecolor=EDGE, linewidth=BAR_LW, label="DeepSeek-V4-Pro")
    ax.bar(x + 0.2, exec_km, 0.38, color=OI["orange"], edgecolor=EDGE, linewidth=BAR_LW, label="Kimi-K2.6")
    for xi, v in zip(x, exec_v4):
        ax.text(xi - 0.2, v + 1.2, f"{v:.1f}", ha="center", fontsize=8.5)
    for xi, v in zip(x, exec_km):
        if not np.isnan(v):
            ax.text(xi + 0.2, v + 1.2, f"{v:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8.5, rotation=12)
    ax.set_ylabel("Pred. executability (%)")
    ax.set_ylim(0, 108)
    ax.text(len(methods) - 1, 70, "backbone-agnostic\n(local SFT model)", ha="center",
            fontsize=7.5, style="italic", color="#444444")
    ax.set_title("Prediction executability (share of 300 questions producing a runnable SQL)")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.02, 0.02))
    save(fig, ROOT / "E5_baselines" / "figs", "fig_e5_executability")


def fig_e5_rank():
    methods = ["MAC-SQL", "CHESS", "ReFoRCE", "SafeQL (PG17)", "DAIL-SQL", "SQLCoder-7B-2"]
    short = {"MAC-SQL": "MAC-SQL", "CHESS": "CHESS", "ReFoRCE": "ReFoRCE",
             "SafeQL (PG17)": "SafeQL", "DAIL-SQL": "DAIL-SQL", "SQLCoder-7B-2": "SQLCoder"}
    old = {m: E5_61[m] for m in methods}
    new = {m: (E5[m][0] + E5[m][1]) / 2 if E5[m][1] is not None else E5[m][0] for m in methods}

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    # bump chart: rank at 61-question deepseek-chat vs rank at 300-question dual-backbone mean
    order_old = sorted(methods, key=lambda m: -old[m])
    order_new = sorted(methods, key=lambda m: -new[m])
    rank_old = {m: i + 1 for i, m in enumerate(order_old)}
    rank_new = {m: i + 1 for i, m in enumerate(order_new)}
    palette = [OI["blue"], OI["orange"], OI["green"], OI["purple"], OI["verm"], OI["sky"]]
    for i, m in enumerate(methods):
        ax.plot([0, 1], [rank_old[m], rank_new[m]], "-o", color=palette[i], lw=1.6, ms=7)
        ax.text(-0.04, rank_old[m], f"{short[m]} ({old[m]:.1f})", ha="right", va="center", fontsize=9, color=palette[i])
        ax.text(1.04, rank_new[m], f"{short[m]} ({new[m]:.1f})", ha="left", va="center", fontsize=9, color=palette[i])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["61-question subset\n(DeepSeek-chat)", "300-question subset\n(dual-backbone mean)"])
    ax.set_ylabel("Rank (by EX)")
    ax.set_yticks(range(1, 7))
    ax.set_xlim(-0.65, 1.6)
    ax.invert_yaxis()
    ax.set_title("Rank shift: small-sample probe vs. full stratified subset")
    ax.grid(axis="y", alpha=0.15)
    save(fig, ROOT / "E5_baselines" / "figs", "fig_e5_rank_shift")


def fig_e5_gating():
    """门控收益:E5 修复变异净伤害 → 保守门控恢复生成器质量。

    (a) EX_strict 无门控 vs 门控 (双骨干), norepair 上限参考线;
    (b) V3 judge 触发次数下降 (变异 SQL 连锁告警消失);
    (c) 无门控修复变异结局归因 (deepseek 侧 vs norepair 逐题比对)。
    数据: installation.txt 章八门控重跑 + 消融归因分析。
    """
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.7))
    x = np.arange(2)
    labels = ["DeepSeek-V4-Pro", "Kimi-K2.6"]
    tips = {}

    # (a) EX_strict: ungated vs gated, norepair reference
    ax = axes[0]
    ungated, gated = [40.3, 38.7], [51.7, 52.0]
    b1 = ax.bar(x - 0.18, ungated, 0.34, color=OI["sky"], edgecolor=EDGE, linewidth=BAR_LW)
    b2 = ax.bar(x + 0.18, gated, 0.34, color=OI["blue"], hatch="//", edgecolor=EDGE, linewidth=BAR_LW)
    for b, ds, v in zip(b1, ["v4pro", "kimi"], ungated):
        gid = _gid(f"e5ga-{ds}-ungated"); b.set_gid(gid)
        tips[gid] = f"Ungated loop ({ds}): EX {v:.1f}%"
    for b, ds, v in zip(b2, ["v4pro", "kimi"], gated):
        gid = _gid(f"e5ga-{ds}-gated"); b.set_gid(gid)
        tips[gid] = f"Gated main ({ds}): EX {v:.1f}%"
    for xi, v in zip(x - 0.18, ungated):
        ax.text(xi, v + 0.9, f"{v:.1f}", ha="center", fontsize=8.5)
    for xi, v in zip(x + 0.18, gated):
        ax.text(xi, v + 1.7, f"{v:.1f}", ha="center", fontsize=8.5)
    for xi, (u, gain) in enumerate(zip(ungated, gated)):
        ax.text(xi + 0.18, gain + 5.8, f"+{gain - u:.1f}", ha="center", fontsize=9,
                color=OI["verm"], fontweight="bold")
    ax.axhline(53.0, color=OI["black"], lw=1.0, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("EX$_{strict}$ (%)")
    ax.set_ylim(0, 62)
    ax.set_title("(a) Repair-gating benefit")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=OI["sky"], edgecolor=EDGE, linewidth=BAR_LW),
                       plt.Rectangle((0, 0), 1, 1, facecolor=OI["blue"], edgecolor=EDGE, linewidth=BAR_LW, hatch="//"),
                       plt.Line2D([0], [0], color=OI["black"], lw=1.0, ls=":")],
              labels=["Ungated loop", "Gated loop",
                      "norepair 53.0 (ceiling ref.)"],
              frameon=False, fontsize=7.5, loc="lower right")

    # (b) V3 judge trigger count
    ax = axes[1]
    trig_un, trig_gd = [104, 113], [39, 40]
    b1 = ax.bar(x - 0.18, trig_un, 0.34, color=OI["sky"], edgecolor=EDGE, linewidth=BAR_LW)
    b2 = ax.bar(x + 0.18, trig_gd, 0.34, color=OI["blue"], hatch="//", edgecolor=EDGE, linewidth=BAR_LW)
    for b, ds, v in zip(b1, ["v4pro", "kimi"], trig_un):
        gid = _gid(f"e5gb-{ds}-ungated"); b.set_gid(gid)
        tips[gid] = f"Ungated loop ({ds}): {v} V3 triggers"
    for b, ds, v in zip(b2, ["v4pro", "kimi"], trig_gd):
        gid = _gid(f"e5gb-{ds}-gated"); b.set_gid(gid)
        tips[gid] = f"Gated main ({ds}): {v} V3 triggers"
    for xi, v in zip(x - 0.18, trig_un):
        ax.text(xi, v + 2.5, str(v), ha="center", fontsize=8.5)
    for xi, v in zip(x + 0.18, trig_gd):
        ax.text(xi, v + 2.5, str(v), ha="center", fontsize=8.5)
    for xi, (u, gv) in enumerate(zip(trig_un, trig_gd)):
        ax.text(xi, max(u, gv) + 8, f"{(gv - u) / u * 100:+.0f}%", ha="center",
                fontsize=9, color=OI["green"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("V3 judge triggers (/300 q)")
    ax.set_ylim(0, 140)
    ax.set_title("(b) Semantic-verification triggers\n(cascade alarms eliminated)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=OI["sky"], edgecolor=EDGE, linewidth=BAR_LW),
                       plt.Rectangle((0, 0), 1, 1, facecolor=OI["blue"], edgecolor=EDGE, linewidth=BAR_LW, hatch="//")],
              labels=["Ungated loop", "Gated loop"],
              frameon=False, fontsize=8, loc="upper left")

    # (c) mutation outcomes: ungated (orig. run) vs verify-then-replace control
    #     (offline replay on the compressed-injection run, E-F)
    ax = axes[2]
    cats = ["Correct\nto broken", "Wrong\nto fixed", "Net effect"]
    un_vals = [47, 9, -38]
    vtr_vals = [0, 3, 3]
    xb = np.arange(3)
    b1 = ax.bar(xb - 0.18, un_vals, 0.34, color=OI["verm"], edgecolor=EDGE, linewidth=BAR_LW,
                label="Ungated (orig. run)")
    b2 = ax.bar(xb + 0.18, vtr_vals, 0.34, color=OI["blue"], hatch="//", edgecolor=EDGE,
                linewidth=BAR_LW, label="Verify-then-replace")
    un_tips = ["Ungated repair mutations broke 47 previously correct SQLs",
               "Ungated repair mutations fixed 9 wrong SQLs",
               "Net effect of ungated repair: -38 questions (net harmful)"]
    vtr_tips = ["VtR control (compressed-injection replay): mutations applied, 0 broken",
                "VtR control: 3 wrong SQLs rescued",
                "VtR net effect: +3 questions; a mass the control eliminates"]
    for rect, cat, tip in zip(b1, ["broken", "fixed", "net"], un_tips):
        gid = _gid(f"e5gc-{cat}"); rect.set_gid(gid); tips[gid] = tip
    for rect, cat, tip in zip(b2, ["broken", "fixed", "net"], vtr_tips):
        gid = _gid(f"e5gcv-{cat}"); rect.set_gid(gid); tips[gid] = tip
    for xi, v in zip(xb - 0.18, un_vals):
        ax.text(xi, v + (2.5 if v >= 0 else -6.5), f"{v:+d}" if v < 0 else str(v),
                ha="center", fontsize=9, fontweight="bold")
    for xi, v in zip(xb + 0.18, vtr_vals):
        ax.text(xi, v + 2.5, f"{v:+d}" if v > 0 else "0",
                ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0, color=OI["black"], lw=0.9)
    ax.set_xticks(xb)
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel("Questions (of 300)")
    ax.set_ylim(-52, 60)
    ax.set_title("(c) Ungated outcomes vs. VtR control\n(DeepSeek-V4-Pro, compressed injection)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.subplots_adjust(wspace=0.34)
    save(fig, ROOT / "E5_baselines" / "figs", "fig_e5_gating_benefit", tips=tips)


def fig_ed_tradeoff():
    """E-D (T1-2d):conditional-clear 折中曲线(误报率 vs 拦截率)。

    每骨干一条 Pareto 前沿:V3(LLM)清除权按告警类别权限子集 S 扫描
    (cleared iff V3 pass 且全部告警类别 ∈ S);锚点 = 无条件清除(S=全集,
    被前沿支配)与不清除(S=∅);参考点 = SC-vote / CHESS-UT(E-C)。
    数据: results/eD_utility_analysis.json (t12d)。
    """
    d = json.loads((RGCV_RES / "eD_utility_analysis.json").read_text(encoding="utf-8"))
    t12d = d["t12d_conditional_clear"]
    fig, ax = plt.subplots(figsize=(4.7, 3.5))
    tips = {}

    eng_style = {"deepseek-v4-pro": ("DeepSeek-V4-Pro", OI["blue"]),
                 "kimi-k2.6": ("Kimi-K2.6", OI["verm"])}
    for eng, (label, color) in eng_style.items():
        dd = t12d[eng]
        fr = dd["frontier"]
        fx = [p["false_alarm"] * 100 for p in fr]
        fy = [p["interception"] * 100 for p in fr]
        line, = ax.plot(fx, fy, "-o", color=color, lw=1.6, ms=4.5,
                        markerfacecolor="white", markeredgewidth=1.2,
                        zorder=3, label=f"{label} conditional-clear frontier")
        gid = _gid(f"ed-{eng}-frontier"); line.set_gid(gid)
        tips[gid] = (f"{label}: {len(fr)} Pareto-optimal permission sets; "
                     f"FA {fx[0]:.1f}-{fx[-1]:.1f}%, interception "
                     f"{fy[0]:.1f}-{fy[-1]:.1f}%")
        # 锚点:无条件清除(S=全集)被前沿支配
        anchors = {
            "unconditional": {"deepseek-v4-pro": (3.64, 11.36),
                              "kimi-k2.6": (0.59, 12.5)}[eng],
            "no-clear": {"deepseek-v4-pro": (8.48, 16.67),
                         "kimi-k2.6": (8.88, 21.88)}[eng],
        }
        ua = ax.scatter(*anchors["unconditional"], marker="x", s=55,
                        color=color, linewidths=2.0, zorder=4)
        gid = _gid(f"ed-{eng}-uncond"); ua.set_gid(gid)
        tips[gid] = (f"{label} unconditional V3 clearing "
                     f"(dominated by the frontier)")
        nc = ax.scatter(*anchors["no-clear"], marker="D", s=40,
                        facecolor="white", edgecolor=color,
                        linewidth=1.4, zorder=4)
        gid = _gid(f"ed-{eng}-noclear"); nc.set_gid(gid)
        tips[gid] = f"{label} no-clear (V1+V2 one-way alarms)"
    # 机制参考点(E-C)
    refs = {
        "SC-vote": {"deepseek-v4-pro": (13.3, 50.0), "kimi-k2.6": (14.8, 35.2)},
        "CHESS-UT": {"deepseek-v4-pro": (3.6, 16.7), "kimi-k2.6": (2.4, 15.6)},
    }
    for name, pts in refs.items():
        for eng, (x, y) in pts.items():
            color = eng_style[eng][1]
            fill = OI["orange"] if name == "SC-vote" else OI["green"]
            sc = ax.scatter(x, y, marker="s", s=34, facecolor=fill,
                            edgecolor=color, linewidth=1.1, zorder=4,
                            alpha=0.9)
            gid = _gid(f"ed-ref-{name}-{eng}"); sc.set_gid(gid)
            tips[gid] = f"{name} ({eng_style[eng][0]}): FA {x}%, interception {y}%"
    ax.scatter([], [], marker="s", facecolor=OI["orange"], edgecolor=EDGE,
               linewidth=BAR_LW, label="SC-vote (E-C)")
    ax.scatter([], [], marker="s", facecolor=OI["green"], edgecolor=EDGE,
               linewidth=BAR_LW, label="CHESS-UT (E-C)")
    ax.plot([], [], "x", color="gray", markersize=7, markeredgewidth=2.0,
            label="Unconditional V3 clear (dominated)")
    ax.plot([], [], "D", color="gray", markersize=5.5, markerfacecolor="white",
            markeredgecolor=EDGE, linewidth=BAR_LW, label="No-clear (V1+V2)")
    ax.set_xlabel("False-alarm rate on correct queries (%)")
    ax.set_ylabel("Interception of silent errors (%)")
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 55)
    ax.set_title("V3 clearing-permission trade-off\n(category-restricted policies, E-A replay)")
    ax.legend(frameon=False, fontsize=7.0, loc="upper left")
    fig.tight_layout()
    save(fig, ROOT / "E3_validation" / "figs", "fig_ed_tradeoff", tips=tips)


def build_html(cards):
    """Assemble interactive_figures.html embedding each figure's SVG
    (native <title> hover tooltips are already injected into the SVGs)."""
    css = ("body{font-family:Georgia,'Times New Roman',serif;max-width:1080px;margin:24px auto;"
           "padding:0 16px;background:#fafafa;color:#222}"
           "h1{font-size:22px}h1 small{font-size:13px;color:#666;font-weight:normal}"
           ".fig{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px 18px;"
           "margin:18px 0;box-shadow:0 1px 4px rgba(0,0,0,.07)}"
           ".fig h2{font-size:15px;margin:2px 0 8px}"
           ".fig .note{font-size:12px;color:#777;margin-top:6px}"
           ".fig svg{max-width:100%;height:auto;display:block}")
    parts = ["<!DOCTYPE html>", "<html><head><meta charset='utf-8'>",
             "<title>RGCV figures - interactive previews</title>",
             f"<style>{css}</style></head><body>",
             "<h1>RGCV - Interactive Figure Previews "
             "<small>hover any bar/point for exact values (native SVG tooltips); "
             "the static PDFs under figs/ are the submission artifacts</small></h1>"]
    for title, svg_path, note in cards:
        svg = svg_path.read_text(encoding="utf-8")
        svg = re.sub(r"<\?xml[^>]*\?>\s*", "", svg)
        svg = re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg)
        parts.append(f"<div class='fig'><h2>{title}</h2>{svg}"
                     + (f"<div class='note'>{note}</div>" if note else "") + "</div>")
    parts.append("</body></html>")
    HTML_PATH.write_text("\n".join(parts), encoding="utf-8")
    print(f"saved {HTML_PATH}")


if __name__ == "__main__":
    fig_e1()
    fig_e2()
    fig_e3()
    fig_e4()
    fig_e5_main()
    fig_e5_pareto()
    fig_e5_exec()
    fig_e5_rank()
    fig_e5_gating()
    fig_ed_tradeoff()

    # copy the paper figures (PDF) into the submission directory
    paper_map = {
        "E1_retrieval/figs/fig_e1_retrieval_metrics.pdf": "fig_e1_retrieval.pdf",
        "E2_repair/figs/fig_e2_repair.pdf": "fig_e2_repair.pdf",
        "E3_validation/figs/fig_e3_verification_layers.pdf": "fig_e3_verification.pdf",
        "E4_pipeline/figs/fig_e4_pipeline.pdf": "fig_e4_pipeline.pdf",
        "E5_baselines/figs/fig_e5_main_comparison.pdf": "fig_e5_main.pdf",
        "E5_baselines/figs/fig_e5_gating_benefit.pdf": "fig_e5_gating.pdf",
        "E5_baselines/figs/fig_e5_pareto.pdf": "fig_e5_pareto.pdf",
        "E3_validation/figs/fig_ed_tradeoff.pdf": "fig_ed_tradeoff.pdf",
    }
    for src, dst in paper_map.items():
        shutil.copyfile(ROOT / src, PAPER_FIGS / dst)
    print(f"copied {len(paper_map)} PDF figures to {PAPER_FIGS}")

    cards = [
        ("E1 - Schema retrieval quality and token cost", ROOT / "E1_retrieval/figs/fig_e1_retrieval_metrics.svg",
         "Anchor configurations (SG-RAG, Flat BM25) carry rotated value labels; hover bars for means and 95% CI."),
        ("E2 - Repair strategy ladder and per-corruption gains", ROOT / "E2_repair/figs/fig_e2_repair.svg",
         "Panel (a): hatched bars = known-answer EX. Panel (b): cross-hatched = S1-only; hover any bar for exact EX."),
        ("E3 - Verification alarm behavior by layer (n=319)", ROOT / "E3_validation/figs/fig_e3_verification_layers.svg",
         "Hover bars for per-layer flag rates."),
        ("E4 - Budget ladder and closed-loop ablation", ROOT / "E4_pipeline/figs/fig_e4_pipeline.svg",
         "Hover bars for EX; hover token points for actual schema tokens used."),
        ("E5 - End-to-end baseline comparison", ROOT / "E5_baselines/figs/fig_e5_main_comparison.svg",
         "Black-outlined hatched bars = ours; hover bars for EX and per-phase token totals."),
        ("E5 - Accuracy-cost Pareto", ROOT / "E5_baselines/figs/fig_e5_pareto.svg",
         "Marker shape = method, color = backbone; hover points for exact EX and total tokens."),
        ("E5 - Repair-gating benefit", ROOT / "E5_baselines/figs/fig_e5_gating_benefit.svg",
         "Hatched bars = gated (main configuration); hover bars for exact counts."),
        ("Supplementary - E5 prediction executability", ROOT / "E5_baselines/figs/fig_e5_executability.svg", None),
        ("Supplementary - E5 rank shift (61-q probe vs 300-q)", ROOT / "E5_baselines/figs/fig_e5_rank_shift.svg", None),
    ]
    build_html(cards)
    print("all figures done")
