#!/usr/bin/env python3
"""Build the editable B/C/D survey decks in five-slide batches.

The Markdown outlines are the single source of truth.  This script deliberately
keeps text, cards, arrows and charts editable; only paper figures and equations
are raster elements.  Every slide uses a pure-white background.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


ROOT = Path("/mnt/data/zhangshuai/LLMs")
BASE = ROOT / "PPT/李老师组会/ABCD-Agent整体Survey"
OUTLINES = BASE / "outlines"
ASSETS = BASE / "assets"
DECKS = BASE / "decks"
WORK = DECKS / "_work"
LOGO = ROOT / "PPT/李老师组会/Agent动作投机Survey/assets/whu-cse-logo.png"

B_SRC = ROOT / "paper/Agent系列latex论文/端云协同Agent"
C_SRC = ROOT / "paper/Agent系列latex论文/GUI Agent执行"
D_SRC = ROOT / "paper/Agent系列latex论文/Agent幻觉"
AGENT_X = ROOT / "paper/Agent系列latex论文/Agent加速/Agent-X Full Pipeline Acceleration of On-device AI Agents"

HERA = B_SRC / "Hera- Learning Long-Horizon Coordination for Device-Cloud Collaborative LLM Agents/figures"
CORE_B = B_SRC / "CORE- Reducing UI Exposure in Mobile Agents via Collaboration Between Cloud and Local LLMs/pics"
OSWORLD = C_SRC / "OSWorld- Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments/images"
UFO2 = C_SRC / "UFO2- The Desktop AgentOSUFO2- The Desktop AgentOS/figure"
CORE_C = C_SRC / "CORE- Reducing UI Exposure in Mobile Agents via Collaboration Between Cloud and Local LLMs/pics"

FONT_CN = "Microsoft YaHei"
FONT_EQ = "Times New Roman"
RED = RGBColor(166, 0, 0)
RED2 = RGBColor(205, 45, 45)
TEAL = RGBColor(0, 128, 123)
TEAL2 = RGBColor(36, 163, 156)
BLUE = RGBColor(34, 93, 168)
PURPLE = RGBColor(105, 76, 166)
GOLD = RGBColor(184, 126, 0)
DARK = RGBColor(28, 37, 51)
MID = RGBColor(78, 88, 103)
LIGHT = RGBColor(222, 228, 235)
PALE = RGBColor(248, 250, 252)
PALE_RED = RGBColor(255, 246, 246)
PALE_TEAL = RGBColor(242, 251, 249)
PALE_BLUE = RGBColor(244, 248, 255)
WHITE = RGBColor(255, 255, 255)

OUTLINE_FILES = {
    "B": OUTLINES / "B-端云协同Agent-PPT文字大纲.md",
    "C": OUTLINES / "C-GUI-Agent执行-PPT文字大纲.md",
    "D": OUTLINES / "D-可靠性与Agent幻觉-PPT文字大纲.md",
    "M": OUTLINES / "ABCD-总览与桥接-PPT文字大纲.md",
}
FINAL_NAMES = {
    "B": "26.8-B-端云协同Agent-Survey-张帅.pptx",
    "C": "26.8-C-GUI-Agent执行-Survey-张帅.pptx",
    "D": "26.8-D-可靠性与Agent幻觉-Survey-张帅.pptx",
    "M": "26.8-ABCD-Agent系统Survey-总览与桥接-张帅.pptx",
}
SECTIONS = {
    "B": ("B · WHERE", "端云协同 Agent", TEAL),
    "C": ("C · WHAT", "GUI Agent 执行", BLUE),
    "D": ("D · TRUST", "可靠性与 Agent 幻觉", PURPLE),
    "M": ("A–D · SYSTEM", "Agent 系统 Survey", RED),
}
TOTALS = {"B": 24, "C": 27, "D": 25, "M": 18}


def fill(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color


def line(shape, color, width=1.0):
    shape.line.color.rgb = color
    shape.line.width = Pt(width)


def rect(slide, x, y, w, h, fill_color=WHITE, line_color=LIGHT,
         rounded=True, width=1.0):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    fill(shp, fill_color)
    line(shp, line_color, width)
    return shp


def rule(slide, x, y, w, color, h=0.035):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    fill(shp, color)
    shp.line.fill.background()
    return shp


def add_text(slide, text, x, y, w, h, size=16, color=DARK, bold=False,
             align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP, font=FONT_CN,
             margin=0.04, italic=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = align
    p.space_after = Pt(0)
    p.line_spacing = 1.04
    for run in p.runs:
        run.font.name = font
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = color
    return box


def add_lines(slide, items, x, y, w, h, size=16, accent=TEAL,
              bullet=True, spacing=7):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.03)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    for i, raw in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        label, body = split_label(raw)
        if bullet:
            p.text = "• "
        if label:
            r = p.add_run()
            r.text = label + "："
            r.font.name = FONT_CN
            r.font.size = Pt(size)
            r.font.bold = True
            r.font.color.rgb = accent
            r = p.add_run()
            r.text = body
            r.font.name = FONT_CN
            r.font.size = Pt(size)
            r.font.color.rgb = DARK
        else:
            if bullet:
                p.add_run().text = raw
            else:
                p.text = raw
            for r in p.runs:
                r.font.name = FONT_CN
                r.font.size = Pt(size)
                r.font.color.rgb = DARK
        p.space_after = Pt(spacing)
        p.line_spacing = 1.07
    return box


def split_label(text):
    text = text.strip().lstrip("•").strip()
    for sep in ("：", ":"):
        if sep in text:
            a, b = text.split(sep, 1)
            if 0 < len(a) <= 22:
                return a.strip(), b.strip()
    return "", text


def connector(slide, x1, y1, x2, y2, color=MID, width=1.5):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    line(c, color, width)
    return c


def add_picture_contain(slide, path, x, y, w, h):
    path = Path(path)
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    return slide.shapes.add_picture(str(path), Inches(x + (w-pw)/2), Inches(y + (h-ph)/2),
                                    width=Inches(pw), height=Inches(ph))


def raster_pdf(path, key, zoom=2.15):
    if Path(path).suffix.lower() != ".pdf":
        return Path(path)
    out = ASSETS / "paper-figures" / f"{key}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return out
    doc = fitz.open(path)
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    pix.save(out)
    doc.close()
    return out


def formula_png(key, formula, fontsize=29, width=9.2, height=0.9):
    out = ASSETS / "formulas" / f"{key}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return out
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["font.family"] = "Liberation Serif"
    fig = plt.figure(figsize=(width, height), dpi=220, facecolor=(1, 1, 1, 0))
    fig.text(0.5, 0.5, formula, ha="center", va="center", fontsize=fontsize, color="#1c2533")
    plt.axis("off")
    fig.savefig(out, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out


def set_notes(slide, text):
    tf = slide.notes_slide.notes_text_frame
    tf.clear()
    tf.paragraphs[0].text = text


def add_header(slide, deck, title, page):
    _, section_name, accent = SECTIONS[deck]
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE
    if LOGO.exists():
        slide.shapes.add_picture(str(LOGO), Inches(0.25), Inches(0.12), width=Inches(2.55))
    add_text(slide, title, 3.03, 0.16, 9.55, 0.60, size=22, bold=True,
             valign=MSO_ANCHOR.MIDDLE)
    rule(slide, 0.26, 0.91, 9.80, RED)
    rule(slide, 10.18, 0.91, 2.88, accent)
    add_text(slide, section_name, 10.25, 0.54, 2.50, 0.24, size=11, color=accent,
             bold=True, align=PP_ALIGN.RIGHT)
    add_text(slide, f"{page}/{TOTALS[deck]}", 12.47, 7.13, 0.54, 0.18,
             size=10, color=MID, align=PP_ALIGN.RIGHT)


def add_cover(slide, deck, title, subtitle):
    tag, section_name, accent = SECTIONS[deck]
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE
    if LOGO.exists():
        slide.shapes.add_picture(str(LOGO), Inches(0.62), Inches(0.42), width=Inches(4.55))
    rule(slide, 0, 1.38, 10.48, RED, 0.12)
    rule(slide, 10.62, 1.38, 2.25, accent, 0.12)
    add_text(slide, tag, 0.90, 1.86, 2.4, 0.38, size=17, color=accent, bold=True)
    add_text(slide, title, 0.90, 2.28, 7.35, 0.85, size=32, color=RED, bold=True,
             valign=MSO_ANCHOR.MIDDLE)
    add_text(slide, subtitle, 0.93, 3.20, 7.10, 0.86, size=19, color=DARK)
    # Editable concept graphic.
    labels = {
        "B": [("DEVICE", "本地模型\n隐私 · 低延迟"), ("ROUTER", "step / block\n动态协作"), ("CLOUD", "强模型\n规划 · 验证")],
        "C": [("OBSERVE", "Screen / UIA"), ("ACT", "GUI / API"), ("VERIFY", "Effect / Goal")],
        "D": [("DETECT", "证据是否支持"), ("ATTRIBUTE", "根因与传播"), ("RECOVER", "拦截 · 重试")],
        "M": [("WHEN", "A · 动作投机"), ("WHERE / WHAT", "B · 端云　C · GUI"), ("TRUST", "D · 验证恢复")],
    }[deck]
    y0 = 2.02
    for i, (lab, body) in enumerate(labels):
        y = y0 + i*1.35
        rect(slide, 8.65, y, 3.54, 1.02, WHITE, accent, True, 1.4)
        add_text(slide, lab, 8.88, y+0.12, 1.18, 0.25, size=13, color=accent, bold=True)
        add_text(slide, body, 10.08, y+0.11, 1.86, 0.60, size=15, color=DARK,
                 align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
        if i < 2:
            connector(slide, 10.42, y+1.02, 10.42, y+1.31, accent, 1.8)
    rect(slide, 0.93, 4.63, 6.90, 1.12, PALE, LIGHT, True, 1.0)
    add_text(slide, "A 决定何时并行　B 决定在哪里执行　C 决定执行什么　D 决定能否信任",
             1.23, 4.88, 6.30, 0.55, size=17, color=accent, bold=True,
             align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
    add_text(slide, "Agent 系统 Survey｜武汉大学国家网络安全学院｜张帅｜2026.08",
             0.90, 6.22, 11.55, 0.35, size=15, color=RED, bold=True,
             align=PP_ALIGN.CENTER)
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(6.95), Inches(13.333), Inches(0.55))
    fill(band, RED)
    band.line.fill.background()
    add_text(slide, "自强 · 弘毅 · 求是 · 拓新", 0.5, 7.04, 12.3, 0.25,
             size=12, color=WHITE, bold=True, align=PP_ALIGN.CENTER)


def add_card(slide, title, body, x, y, w, h, accent, number=None, size=16):
    rect(slide, x, y, w, h, WHITE, LIGHT, True, 1.0)
    tx = x + 0.20
    if number is not None:
        bubble = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x+0.17), Inches(y+0.18), Inches(0.43), Inches(0.43))
        fill(bubble, accent)
        bubble.line.fill.background()
        add_text(slide, str(number), x+0.17, y+0.20, 0.43, 0.35, size=13,
                 color=WHITE, bold=True, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
        tx = x + 0.72
    add_text(slide, title, tx, y+0.16, w-(tx-x)-0.18, 0.34, size=size,
             color=accent, bold=True)
    add_text(slide, body, x+0.20, y+0.60, w-0.40, h-0.72, size=16, color=DARK)


def add_metric(slide, value, label, x, y, w, accent):
    rect(slide, x, y, w, 0.86, PALE, accent, True, 1.0)
    add_text(slide, value, x+0.06, y+0.06, w-0.12, 0.40, size=22, color=accent,
             bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, label, x+0.06, y+0.50, w-0.12, 0.24, size=11, color=MID,
             align=PP_ALIGN.CENTER)


def parse_outline(path):
    pages = {}
    current = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        ln = raw.strip()
        m = re.match(r"第\s*(\d+)\s*页｜(.+)", ln)
        if m:
            current = int(m.group(1))
            pages[current] = {"label": m.group(2).strip(), "title": "", "subtitle": "", "items": [], "layout": "", "image_note": ""}
            continue
        if current is None or not ln or ln.startswith("```") or ln.startswith("# "):
            continue
        if ln.startswith("标题："):
            pages[current]["title"] = ln.split("：", 1)[1].strip()
        elif ln.startswith("副标题："):
            pages[current]["subtitle"] = ln.split("：", 1)[1].strip()
        elif ln.startswith("页脚："):
            continue
        elif ln.startswith("排版："):
            pages[current]["layout"] = ln.split("：", 1)[1].strip()
        elif ln.startswith("配图："):
            pages[current]["image_note"] = ln.split("：", 1)[1].strip()
        else:
            pages[current]["items"].append(ln.lstrip("•").strip())
    return pages


FIGURES = {
    "B": {
        8: (HERA / "motivating_example.pdf", "Hera motivation"),
        10: (HERA / "teaser.pdf", "Hera IL stage"),
        11: (HERA / "rl_training.pdf", "Hera RL training"),
        12: (HERA / "exps/clouassist_appworld.pdf", "Hera Pareto"),
        13: (CORE_B / "intro_v3.pdf", "CORE motivation"),
        14: (CORE_B / "method_pipeline_v3.pdf", "CORE pipeline"),
        15: (CORE_B / "method_block_partitioin_v2.pdf", "CORE block partition"),
        16: (CORE_B / "method_pipeline_v3.pdf", "CORE collaboration"),
        17: (CORE_B / "radar_chart_sr_upload_colored_labels.pdf", "CORE results"),
        18: (AGENT_X / "fig/proposed_overview.pdf", "Agent-X overview"),
        19: (AGENT_X / "fig/eval_end_to_end_speedup.pdf", "Agent-X results"),
    },
    "C": {
        5: (OSWORLD / "arch_figure_w_config_new.pdf", "OSWorld architecture"),
        6: (OSWORLD / "task_demostration_new.pdf", "OSWorld tasks"),
        8: (UFO2 / "comparison.pdf", "UFO2 comparison"),
        9: (UFO2 / "framework.pdf", "UFO2 framework"),
        10: (UFO2 / "hostagent.pdf", "UFO2 HostAgent"),
        11: (UFO2 / "api_gui_case.pdf", "UFO2 GUI/API hybrid"),
        12: (UFO2 / "knowledge.pdf", "UFO2 knowledge substrate"),
        13: (UFO2 / "multiaction.pdf", "UFO2 speculative multi-action"),
        14: (UFO2 / "pip.pdf", "UFO2 picture-in-picture"),
        15: (UFO2 / "safeguard.pdf", "UFO2 safeguard"),
        16: (UFO2 / "controls.pdf", "UFO2 control results"),
        17: (CORE_C / "method_pipeline_v3.pdf", "CORE pipeline"),
    },
    "D": {
        5: (D_SRC / "3.MIRAGE-Bench/assets/overview.pdf", "MIRAGE overview"),
        6: (D_SRC / "3.MIRAGE-Bench/assets/pipeline.pdf", "MIRAGE pipeline"),
        7: (D_SRC / "1.AgentHallu/Fig/figure1.pdf", "AgentHallu teaser"),
        8: (D_SRC / "1.AgentHallu/Fig/score_across_model.pdf", "AgentHallu results"),
        9: (D_SRC / "4.Where LLM Agents Fail and How They can Learn From Failures/figures/method.pdf", "AgentDebug method"),
        10: (D_SRC / "4.Where LLM Agents Fail and How They can Learn From Failures/figures/task_success_rates_qwen_grouped.pdf", "AgentDebug repair"),
        12: (D_SRC / "5.The Reasoning Trap/imgs/Reasoning_trap.png", "Reasoning Trap"),
        13: (D_SRC / "5.The Reasoning Trap/imgs/GRPO_HR.pdf", "Reasoning Trap results"),
        14: (D_SRC / "7.From Proof to Program/figures/motivation_new.pdf", "Tool-Induced Myopia"),
        15: (D_SRC / "10.Reducing Tool Hallucination via Reliability Alignment/figs/hallucination_type.pdf", "Relign taxonomy"),
        16: (D_SRC / "10.Reducing Tool Hallucination via Reliability Alignment/figs/system.pdf", "Relign pipeline"),
        17: (D_SRC / "10.Reducing Tool Hallucination via Reliability Alignment/figs/metrics_grouped.pdf", "Relign results"),
        18: (D_SRC / "2.Evaluating Tool-Using Language Agents/figures/fig_judge_agreement.png", "AgentProp judge agreement"),
        19: (D_SRC / "2.Evaluating Tool-Using Language Agents/figures/fig_interceptor_effect.png", "AgentProp interceptor"),
        20: (D_SRC / "11.Beyond Final Answers/Architecture.png", "Trajel architecture"),
    },
    "M": {
        6: (ROOT / "paper/Agent系列latex论文/Agent动作投机/Speculative Actions/figures/Speculative_figure_1.pdf", "Speculative Actions"),
        7: (HERA / "teaser.pdf", "Hera device-cloud coordination"),
        8: (UFO2 / "framework.pdf", "UFO2 AgentOS"),
        9: (D_SRC / "3.MIRAGE-Bench/assets/overview.pdf", "MIRAGE overview"),
        10: (CORE_B / "method_pipeline_v3.pdf", "CORE pipeline"),
        11: (UFO2 / "multiaction.pdf", "UFO2 speculative multi-action"),
        12: (AGENT_X / "fig/proposed_overview.pdf", "Agent-X overview"),
    },
}

METRICS = {
    "B": {
        7: [("5.7×", "远程成本降低"), ("97.9%", "cloud-only 性能")],
        12: [("92.5%", "云端成功率"), ("53.6%", "云端成本"), ("69.3%", "E2E latency"), ("61 ms", "router / step")],
        17: [("55.6%", "UI exposure 最高降幅")],
        19: [("1.97×", "prefill"), ("1.73×", "decode"), ("1.61×", "end-to-end")],
        20: [("46.1%", "per-token latency↓"), ("67.4%", "downlink↓")],
        21: [("56.5", "OSWorld"), ("71.6", "AndroidWorld"), ("48.4", "WebArena"), ("80.3", "ScreenSpotPro")],
        22: [("<5.15%", "runtime overhead")], 23: [("ASR <1%", "1,000 adversarial samples")],
    },
    "C": {
        6: [("369", "tasks"), ("134", "evaluators"), ("72.36%", "human"), ("12.24%", "then-best model")],
        16: [(">10%", "vs dedicated CUA")],
        18: [("2B→235B", "统一模型谱系"), ("6", "代表 benchmarks")],
        20: [("22.4→27.6", "AndroidWorld success")],
        22: [("51.1/52.5%", "RSR 3B/7B"), ("23.5%", "Sim-TSR"), ("1.09", "ASO")],
        24: [("92.9/93.4%", "verification accuracy"), ("34.3%", "OSWorld success"), ("+3.8pp", "vs baseline")],
        25: [("86.9%", "GUI-RewardBench"), ("34.0%", "OSWorld success"), ("34.9%", "script reward")],
    },
    "D": {
        8: [("693", "trajectories"), ("41.1%", "best localization"), ("11.6%", "tool-use attribution")],
        10: [("+26%", "relative task success")],
        14: [("1,679", "PyMath tasks"), ("+19.3pp", "final accuracy"), ("≈55%", "high-risk TIM")],
        17: [("56.5→18.3", "ToolLLaMA THR"), ("50.8→14.6", "Llama-3.1 THR"), ("49.2→17.6", "Qwen2.5 THR")],
        18: [("κ=.049", "substring judge"), ("κ=.432", "3-LLM ensemble")],
        19: [("≈0.62", "error propagation"), ("55.8→32.8%", "GPT-4o-mini hallu")],
        21: [("68.3%", "human hallu"), ("F1=.855", "binary judge"), ("AUC=.908", "execution signal"), ("97.1%", "no justify+verify")],
        23: [("ASR <1%", "Cognitive Firewall")],
    },
    "M": {
        7: [("92.5%", "Hera cloud success"), ("55.6%", "CORE exposure↓"), ("1.61×", "Agent-X E2E")],
        10: [("55.6%", "UI exposure↓")],
        12: [("1.97×", "prefill"), ("1.73×", "decode"), ("1.61×", "end-to-end")],
        13: [("ASR <1%", "1,000 adversarial samples")],
    },
}

FORMULAS = {
    ("B", 9): [
        ("hera-mix", r"$\hat a(s,x)=(1-r_{\phi}(s,x))\,\pi_d(s,x)+r_{\phi}(s,x)\,\pi_c(s,x)$"),
        ("hera-obj", r"$\min_{\phi}\;\mathbb{E}[\sum_t r_{\phi}(s_t,x)]\quad \mathrm{s.t.}\quad \mathbb{E}[R(\tau_{mix})]\geq\mathbb{E}[R(\tau_{cloud})]$"),
    ],
    ("D", 16): [
        ("relign-pref", r"$a_{correct}\;\succ\;a_{indecisive}\;\succ\;a_{hallucinated}$"),
    ],
}


def figure_for(deck, page):
    spec = FIGURES.get(deck, {}).get(page)
    if not spec:
        return None, ""
    src, label = spec
    if not src.exists():
        return None, ""
    return raster_pdf(src, f"{deck.lower()}-{page:02d}-{slug(label)}"), label


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def normalise_items(items):
    out = []
    for item in items:
        if item.startswith(("排版", "配图", "页脚")):
            continue
        out.append(item)
    return out


def step_items(items):
    return [x for x in items if re.match(r"^(Step\s*\d+|组件\s*\d+|Stage\s*[IⅡ1-9]+|层次\s*\d+|Gate\s*\d+)", x, re.I)]


def metric_strip(slide, deck, page, accent, y=6.05):
    metrics = METRICS.get(deck, {}).get(page, [])
    if not metrics:
        return 0
    n = len(metrics)
    width = min(2.65, (12.05 - 0.18*(n-1))/n)
    total = n*width + (n-1)*0.18
    x = (13.333-total)/2
    colors = [accent, RED, BLUE, TEAL2]
    for i, (v, lab) in enumerate(metrics):
        add_metric(slide, v, lab, x+i*(width+0.18), y, width, colors[i % len(colors)])
    return 1


def build_formula_slide(slide, deck, page, page_data):
    _, _, accent = SECTIONS[deck]
    add_header(slide, deck, page_data["title"], page)
    formulas = FORMULAS[(deck, page)]
    items = normalise_items(page_data["items"])
    if deck == "B":
        add_lines(slide, items[:3], 0.55, 1.35, 4.05, 2.10, 16, accent)
        for i, (key, formula) in enumerate(formulas):
            img = formula_png(key, formula, 27 if i == 0 else 24)
            rect(slide, 4.90, 1.40+i*1.48, 7.90, 1.17, PALE_BLUE if i else PALE_TEAL, accent, True, 1.0)
            add_picture_contain(slide, img, 5.15, 1.62+i*1.48, 7.40, 0.63)
        add_card(slide, "轨迹级约束", "router 管理的不是单步答案质量，而是未来成功率与累计云成本。",
                 0.57, 3.83, 5.86, 1.47, accent, 1)
        add_card(slide, "二元执行语义", "rφ=0 走设备，rφ=1 升级云端；云端 policy 是质量锚点。",
                 6.72, 3.83, 6.08, 1.47, RED, 2)
        add_lines(slide, items[3:], 0.75, 5.62, 11.85, 0.80, 16, accent)
    else:
        add_lines(slide, items[:3], 0.55, 1.36, 5.35, 2.55, 16, accent)
        img = formula_png(*formulas[0], fontsize=31)
        rect(slide, 6.25, 1.45, 6.55, 1.35, PALE_TEAL, accent, True, 1.0)
        add_picture_contain(slide, img, 6.65, 1.74, 5.75, 0.65)
        add_lines(slide, items[3:], 6.35, 3.13, 6.25, 2.55, 16, accent)
        add_card(slide, "核心变化", "拒绝、澄清与换工具是合法 action，不再被当作消极失败。",
                 0.58, 4.28, 5.32, 1.52, accent, None)
    add_text(slide, "公式采用 Times New Roman / STIX，并作为独立透明 PNG 元素插入。",
             7.2, 6.68, 5.35, 0.20, size=9, color=MID, italic=True, align=PP_ALIGN.RIGHT)


def build_steps_slide(slide, deck, page, data, steps, figure, fig_label):
    _, _, accent = SECTIONS[deck]
    add_header(slide, deck, data["title"], page)
    others = [x for x in normalise_items(data["items"]) if x not in steps]
    if figure:
        rect(slide, 0.45, 1.28, 6.25, 4.90, WHITE, LIGHT, True, 1.0)
        add_picture_contain(slide, figure, 0.62, 1.47, 5.91, 4.35)
        add_text(slide, f"论文原图：{fig_label}", 2.25, 5.90, 4.15, 0.20,
                 size=9, color=MID, italic=True, align=PP_ALIGN.RIGHT)
        # Two-column grids preserve 16 pt body text on dense 4–5 step pages.
        count = min(5, len(steps))
        if count >= 4:
            cols = 2
            card_w, card_h = 2.80, 1.34 if count == 5 else 1.66
            gap_x, gap_y = 0.18, 0.20
            x0, y0 = 6.98, 1.28
            for i, st in enumerate(steps[:5]):
                lab, body = split_label(st)
                col, row = i % cols, i // cols
                x = x0 + col*(card_w+gap_x)
                y = y0 + row*(card_h+gap_y)
                add_card(slide, lab or f"Step {i+1}", body if lab else st,
                         x, y, card_w, card_h,
                         accent if i % 2 == 0 else BLUE, i+1, 14)
        else:
            x0, y0, w, h = 6.98, 1.28, 5.86, 1.34
            for i, st in enumerate(steps[:3]):
                lab, body = split_label(st)
                add_card(slide, lab or f"Step {i+1}", body if lab else st,
                         x0, y0+i*1.54, w, h,
                         accent if i % 2 == 0 else BLUE, i+1, 15)
        if others:
            add_text(slide, others[-1], 7.18, 6.40, 5.43, 0.42, size=13,
                     color=RED, bold=True, align=PP_ALIGN.CENTER)
    else:
        n = min(5, len(steps))
        card_w = (12.12 - 0.20*(n-1))/n
        for i, st in enumerate(steps[:5]):
            lab, body = split_label(st)
            x = 0.58+i*(card_w+0.20)
            add_card(slide, lab or f"Step {i+1}", body if lab else st, x, 1.52, card_w, 2.35,
                     [accent, BLUE, TEAL2, RED, GOLD][i], i+1, 15)
            if i < n-1:
                connector(slide, x+card_w, 2.69, x+card_w+0.18, 2.69, MID, 1.4)
        if others:
            rect(slide, 0.58, 4.30, 12.10, 2.05, PALE, LIGHT, True, 1.0)
            add_lines(slide, others[:5], 0.82, 4.58, 11.60, 1.50, 16, accent)
    metric_strip(slide, deck, page, accent, 6.16)


def build_generic_slide(slide, deck, page, data):
    _, _, accent = SECTIONS[deck]
    add_header(slide, deck, data["title"], page)
    items = normalise_items(data["items"])
    figure, fig_label = figure_for(deck, page)
    metrics = METRICS.get(deck, {}).get(page, [])
    usable_bottom = 5.74 if metrics else 6.58
    if figure:
        img_x = 6.20
        text_x = 0.48
        if page % 2 == 0:
            img_x, text_x = 0.48, 7.05
        rect(slide, img_x, 1.28, 5.82, usable_bottom-1.28, WHITE, LIGHT, True, 1.0)
        add_picture_contain(slide, figure, img_x+0.15, 1.46, 5.52, usable_bottom-1.83)
        add_text(slide, f"论文原图：{fig_label}", img_x+1.25, usable_bottom-0.30, 4.28, 0.18,
                 size=9, color=MID, italic=True, align=PP_ALIGN.RIGHT)
        rect(slide, text_x, 1.28, 5.82, usable_bottom-1.28, PALE, LIGHT, True, 1.0)
        add_lines(slide, items[:8], text_x+0.24, 1.52, 5.34, usable_bottom-1.72,
                  16, accent, True, 6)
    else:
        # Give timeline/map/summary pages a compact editable visual backbone.
        if any(k in data["title"] for k in ("地图", "分类", "层次", "统一", "闭环", "对比", "小结", "过渡")):
            n = min(6, max(3, len(items)))
            selected = items[:n]
            if len(selected) < 3:
                selected += ["研究对象：状态、动作与证据", "比较维度：质量、成本与风险"][:3-len(selected)]
            card_w = (12.08 - 0.18*(n-1))/n
            for i, raw in enumerate(selected):
                lab, body = split_label(raw)
                x = 0.60+i*(card_w+0.18)
                add_card(slide, lab or f"0{i+1}", body if lab else raw, x, 1.58, card_w, 2.32,
                         [accent, BLUE, TEAL2, RED, GOLD, PURPLE][i], i+1, 14)
                if i < n-1:
                    connector(slide, x+card_w, 2.73, x+card_w+0.16, 2.73, MID, 1.3)
            rest = items[n:]
            if rest:
                rect(slide, 0.60, 4.30, 12.08, 1.86, PALE, LIGHT, True, 1.0)
                add_lines(slide, rest[:6], 0.87, 4.57, 11.54, 1.35, 16, accent)
        else:
            mid = (len(items)+1)//2
            left, right = items[:mid], items[mid:]
            for x, group, col in ((0.48, left, accent), (6.73, right, BLUE)):
                rect(slide, x, 1.30, 6.10, usable_bottom-1.30, PALE if x < 1 else WHITE, LIGHT, True, 1.0)
                add_lines(slide, group, x+0.28, 1.58, 5.54, usable_bottom-1.80,
                          16, col, True, 7)
    metric_strip(slide, deck, page, accent, 6.02)


def note_for(deck, page, data):
    items = normalise_items(data["items"])
    key = "；".join(items[:3])
    bridge = ""
    if any(x in data["title"] for x in ("桥梁", "A–", "B–", "C–", "D–", "过渡", "合流")):
        bridge = " 这一页承担跨方向衔接：明确它同时改变了执行位置、动作语义或可靠性边界。"
    return (f"本页目的：{data['title']}。讲解时先给出结论，再解释证据和方法边界。"
            f"核心内容：{key}。{bridge} 不要把相关性或单步指标等同于端到端任务成功；"
            "对结果数字说明比较对象，对原图按从左到右或从上到下的顺序讲。")


def build_slide(prs, deck, page, data):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    if page == 1:
        add_cover(slide, deck, data["title"], data["subtitle"])
    elif (deck, page) in FORMULAS:
        build_formula_slide(slide, deck, page, data)
    else:
        steps = step_items(data["items"])
        fig, fig_label = figure_for(deck, page)
        if len(steps) >= 3:
            build_steps_slide(slide, deck, page, data, steps, fig, fig_label)
        else:
            build_generic_slide(slide, deck, page, data)
    set_notes(slide, note_for(deck, page, data))
    return slide


def new_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    return prs


def build_batch(deck, batch):
    ASSETS.mkdir(parents=True, exist_ok=True)
    DECKS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    pages = parse_outline(OUTLINE_FILES[deck])
    start = 1 + (batch-1)*5
    end = min(batch*5, TOTALS[deck])
    if start > TOTALS[deck]:
        raise ValueError(f"{deck} has only {TOTALS[deck]} slides")
    if batch == 1:
        prs = new_presentation()
    else:
        prev = WORK / f"{deck}-batch-{batch-1:02d}.pptx"
        if not prev.exists():
            raise FileNotFoundError(f"Build batch {batch-1} first: {prev}")
        prs = Presentation(prev)
    for page in range(start, end+1):
        build_slide(prs, deck, page, pages[page])
    prs.core_properties.title = f"{SECTIONS[deck][1]} Survey"
    prs.core_properties.subject = "Agent 系统 A–D 整体 Survey"
    prs.core_properties.author = "张帅"
    work_out = WORK / f"{deck}-batch-{batch:02d}.pptx"
    prs.save(work_out)
    if end == TOTALS[deck]:
        final = DECKS / FINAL_NAMES[deck]
        prs.save(final)
        return final
    return work_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", choices=["B", "C", "D", "M"], required=True)
    ap.add_argument("--batch", type=int, required=True)
    args = ap.parse_args()
    print(build_batch(args.deck, args.batch))


if __name__ == "__main__":
    main()
