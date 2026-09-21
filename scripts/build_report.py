import html
import json
import re
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_SOURCE = ROOT / "report" / "report.md"
REPORT_OUTPUT = ROOT / "report" / "final_report.pdf"
ARTIFACTS_DIR = ROOT / "artifacts"
FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def _register_fonts() -> None:
    if not FONT_REGULAR.exists() or not FONT_BOLD.exists():
        raise FileNotFoundError("DejaVu Sans fonts are required to build the Russian PDF.")
    pdfmetrics.registerFont(TTFont("DejaVu", FONT_REGULAR))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", FONT_BOLD))


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleRu",
            parent=base["Title"],
            fontName="DejaVu-Bold",
            fontSize=21,
            leading=26,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#17365D"),
            spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "Heading1Ru",
            parent=base["Heading1"],
            fontName="DejaVu-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#17365D"),
            spaceBefore=14,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Heading2Ru",
            parent=base["Heading2"],
            fontName="DejaVu-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#244A73"),
            spaceBefore=10,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "BodyRu",
            parent=base["BodyText"],
            fontName="DejaVu",
            fontSize=9.2,
            leading=13,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "SmallRu",
            parent=base["BodyText"],
            fontName="DejaVu",
            fontSize=7.4,
            leading=9.5,
        ),
        "tiny": ParagraphStyle(
            "TinyRu",
            parent=base["BodyText"],
            fontName="DejaVu",
            fontSize=6.5,
            leading=8.2,
        ),
        "caption": ParagraphStyle(
            "CaptionRu",
            parent=base["BodyText"],
            fontName="DejaVu",
            fontSize=7.5,
            leading=10,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
    }


def _paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    escaped = html.escape(text)
    escaped = escaped.replace("\\(y \\in \\{0, 1\\}\\)", "y ∈ {0, 1}")
    escaped = escaped.replace("`", "")
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(
        r"(https://[^\s<]+)",
        r'<link href="\1" color="#175CD3">\1</link>',
        escaped,
    )
    return Paragraph(escaped, style)


def _table(
    data: list[list[str]],
    widths: list[float],
    styles: dict[str, ParagraphStyle],
    style_name: str = "small",
) -> Table:
    cells = [[_paragraph(value, styles[style_name]) for value in row] for row in data]
    table = Table(cells, colWidths=widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
                ("FONTNAME", (0, 0), (-1, 0), "DejaVu-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#667085")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _related_work_table(styles: dict[str, ParagraphStyle]) -> Table:
    rows = [
        ["Model", "Protocol", "AUROC", "Macro F1", "Recall"],
        ["DistilBERT", "multilabel, 310k train", "0.9804", "0.3879", "0.9001"],
        ["RoBERTa BCE", "multilabel, 310k train", "0.9813", "0.4749", "0.8891"],
        ["RoBERTa focal loss", "multilabel, 310k train", "0.9818", "0.4648", "0.8839"],
        ["BiLSTM", "multilabel, 310k train", "0.9754", "0.3638", "0.8761"],
    ]
    return _table(rows, [4 * cm, 5 * cm, 2 * cm, 2.2 * cm, 2 * cm], styles)


def _dataset_table(styles: dict[str, ParagraphStyle]) -> Table:
    path = ARTIFACTS_DIR / "dataset_summary.json"
    if not path.exists():
        return _table(
            [["Split", "Rows", "Toxic", "Toxic share"], ["Нет данных", "-", "-", "-"]],
            [4 * cm, 2.5 * cm, 2.5 * cm, 3 * cm],
            styles,
        )
    summary = json.loads(path.read_text(encoding="utf-8"))
    rows = [["Split", "Rows", "Toxic", "Toxic share", "Median chars"]]
    for split in ("train", "validation", "test"):
        values = summary[split]
        rows.append(
            [
                split,
                f"{values['rows']:,}",
                f"{values['toxic_rows']:,}",
                f"{100 * values['toxic_share']:.2f}%",
                f"{values['median_characters']:.0f}",
            ]
        )
    return _table(rows, [3 * cm, 2.5 * cm, 2.5 * cm, 3 * cm, 3 * cm], styles)


def _metrics_table(styles: dict[str, ParagraphStyle]) -> Table:
    path = ARTIFACTS_DIR / "metrics.csv"
    if not path.exists():
        return _table(
            [["Model", "Split", "F1", "Macro-F1", "PR-AUC", "ROC-AUC"], ["Нет данных"] * 6],
            [5 * cm, 1.8 * cm, 1.4 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm],
            styles,
        )
    metrics = pd.read_csv(path)
    rows = [["Model", "Split", "Prec.", "Recall", "F1", "Macro F1", "PR AUC", "ROC AUC"]]
    model_aliases = {
        "Dummy prior": "Dummy",
        "TF-IDF + Logistic Regression": "TF-IDF + LogReg",
        "Word TF-IDF + Linear SVM": "Word TF-IDF + SVM",
        "Word+char TF-IDF + Linear SVM": "Word+char TF-IDF + SVM",
        "DistilBERT (fine-tuned)": "DistilBERT",
    }
    for record in metrics.to_dict(orient="records"):
        rows.append(
            [
                model_aliases.get(str(record["model"]), str(record["model"])),
                "val" if str(record["split"]) == "validation" else str(record["split"]),
                f"{record['precision']:.3f}",
                f"{record['recall']:.3f}",
                f"{record['f1_toxic']:.3f}",
                f"{record['f1_macro']:.3f}",
                f"{record['pr_auc']:.3f}",
                f"{record['roc_auc']:.3f}",
            ]
        )
    return _table(
        rows,
        [4 * cm, 1.2 * cm, 1.5 * cm, 1.5 * cm, 1.2 * cm, 1.7 * cm, 1.6 * cm, 1.6 * cm],
        styles,
        style_name="tiny",
    )


def _threshold_table(styles: dict[str, ParagraphStyle]) -> Table:
    path = ARTIFACTS_DIR / "threshold_comparison.csv"
    frame = pd.read_csv(path)
    rows = [["Policy", "Threshold", "Accuracy", "Precision", "Recall", "F1 toxic"]]
    for record in frame.to_dict(orient="records"):
        rows.append(
            [
                str(record["threshold_policy"]),
                f"{record['threshold']:.3f}",
                f"{record['accuracy']:.3f}",
                f"{record['precision']:.3f}",
                f"{record['recall']:.3f}",
                f"{record['f1_toxic']:.3f}",
            ]
        )
    return _table(rows, [4.2 * cm, 2.1 * cm, 2.1 * cm, 2.2 * cm, 2.1 * cm, 2.2 * cm], styles)


def _uncertainty_table(styles: dict[str, ParagraphStyle]) -> Table:
    path = ARTIFACTS_DIR / "bootstrap_ci.csv"
    frame = pd.read_csv(path)
    rows = [["Metric", "Point estimate", "95% CI lower", "95% CI upper"]]
    for record in frame.to_dict(orient="records"):
        rows.append(
            [
                str(record["metric"]),
                f"{record['point_estimate']:.3f}",
                f"{record['ci_lower_95']:.3f}",
                f"{record['ci_upper_95']:.3f}",
            ]
        )
    return _table(rows, [4.5 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm], styles)


def _error_analysis_table(styles: dict[str, ParagraphStyle]) -> Table:
    path = ARTIFACTS_DIR / "error_analysis.csv"
    frame = pd.read_csv(path)
    sample = pd.concat(
        [
            frame[frame["error_type"] == "False positive"].head(2),
            frame[frame["error_type"] == "False negative"].head(2),
        ],
        ignore_index=True,
    )
    rows = [["Error", "Score", "Test comment excerpt"]]
    for record in sample.to_dict(orient="records"):
        rows.append(
            [
                "FP" if record["error_type"] == "False positive" else "FN",
                f"{record['score']:.3f}",
                str(record["text"]),
            ]
        )
    return _table(rows, [1.4 * cm, 1.6 * cm, 12.4 * cm], styles, style_name="tiny")


def _best_model_summary(styles: dict[str, ParagraphStyle]) -> Paragraph:
    metrics_path = ARTIFACTS_DIR / "metrics.csv"
    if not metrics_path.exists():
        return _paragraph("Полный эксперимент ещё не выполнен.", styles["body"])
    metrics = pd.read_csv(metrics_path)
    validation = metrics[metrics["split"] == "validation"]
    best_validation = validation.loc[validation["f1_toxic"].idxmax()]
    model_name = str(best_validation["model"])
    test_row = metrics[(metrics["model"] == model_name) & (metrics["split"] == "test")].iloc[0]
    text = (
        f"По validation выбран подход «{model_name}» с порогом "
        f"{best_validation['threshold']:.3f}. На независимом test он получил "
        f"F1={test_row['f1_toxic']:.3f}, macro-F1={test_row['f1_macro']:.3f}, "
        f"PR-AUC={test_row['pr_auc']:.3f} и ROC-AUC={test_row['roc_auc']:.3f}."
    )
    return _paragraph(text, styles["body"])


def _abstract_results(styles: dict[str, ParagraphStyle]) -> Paragraph:
    metrics = pd.read_csv(ARTIFACTS_DIR / "metrics.csv")
    test = metrics[metrics["split"] == "test"].set_index("model")
    transformer = test.loc["DistilBERT (fine-tuned)"]
    classical = test.loc["Word+char TF-IDF + Linear SVM"]
    text = (
        f"На test DistilBERT получил F1={transformer['f1_toxic']:.3f} и "
        f"PR-AUC={transformer['pr_auc']:.3f}, классический SVM — "
        f"F1={classical['f1_toxic']:.3f} и PR-AUC={classical['pr_auc']:.3f}."
    )
    return _paragraph(text, styles["body"])


def _comparison_summary(styles: dict[str, ParagraphStyle]) -> Paragraph:
    metrics = pd.read_csv(ARTIFACTS_DIR / "metrics.csv")
    test = metrics[metrics["split"] == "test"].set_index("model")
    classical = test.loc["Word+char TF-IDF + Linear SVM"]
    transformer = test.loc["DistilBERT (fine-tuned)"]
    f1_gain = transformer["f1_toxic"] - classical["f1_toxic"]
    text = (
        "Современный Transformer-baseline сравнивается с классическими моделями "
        "на тех же train/validation/test split. DistilBERT получил на test "
        f"F1={transformer['f1_toxic']:.3f}, PR-AUC={transformer['pr_auc']:.3f} и "
        f"ROC-AUC={transformer['roc_auc']:.3f}; прирост F1 относительно лучшего "
        f"классического SVM составляет {f1_gain:+.3f}."
    )
    return _paragraph(text, styles["body"])


def _ablation_summary(styles: dict[str, ParagraphStyle]) -> Paragraph:
    metrics = pd.read_csv(ARTIFACTS_DIR / "metrics.csv")
    test = metrics[metrics["split"] == "test"].set_index("model")
    word = test.loc["Word TF-IDF + Linear SVM"]
    word_char = test.loc["Word+char TF-IDF + Linear SVM"]
    text = (
        "Контролируемая абляция сохраняет классификатор, калибровку и протокол "
        "неизменными. Добавление character n-грамм изменяет test F1 "
        f"с {word['f1_toxic']:.3f} до {word_char['f1_toxic']:.3f} "
        f"({word_char['f1_toxic'] - word['f1_toxic']:+.3f}) и PR-AUC "
        f"с {word['pr_auc']:.3f} до {word_char['pr_auc']:.3f} "
        f"({word_char['pr_auc'] - word['pr_auc']:+.3f})."
    )
    return _paragraph(text, styles["body"])


def _transformer_setup(styles: dict[str, ParagraphStyle]) -> Paragraph:
    metadata = json.loads((ARTIFACTS_DIR / "transformer_metadata.json").read_text(encoding="utf-8"))
    text = (
        f"В эксперименте число эпох равно {metadata['epochs']:g}, максимальная "
        f"длина — {metadata['max_length']} tokens, train batch size — "
        f"{metadata['train_batch_size']} и eval batch size "
        f"— {metadata['eval_batch_size']}, learning rate — "
        f"{metadata.get('learning_rate', 2e-5):g}, weight decay — "
        f"{metadata.get('weight_decay', 0.01):g}. Обучение выполнено на "
        f"{str(metadata['device']).upper()}."
    )
    return _paragraph(text, styles["body"])


def _threshold_summary(styles: dict[str, ParagraphStyle]) -> Paragraph:
    frame = pd.read_csv(ARTIFACTS_DIR / "threshold_comparison.csv").set_index("threshold_policy")
    default = frame.loc["Default"]
    selected = frame.loc["Validation-selected"]
    text = (
        f"Validation-selected threshold {selected['threshold']:.3f} изменяет recall "
        f"с {default['recall']:.3f} до {selected['recall']:.3f} и F1 "
        f"с {default['f1_toxic']:.3f} до {selected['f1_toxic']:.3f}. "
        "Порог не настраивается на test."
    )
    return _paragraph(text, styles["body"])


def _error_counts_summary(styles: dict[str, ParagraphStyle]) -> Paragraph:
    summary = json.loads(
        (ARTIFACTS_DIR / "error_analysis_summary.json").read_text(encoding="utf-8")
    )
    text = (
        f"Матрица ошибок содержит {summary['true_negative']:,} true negative, "
        f"{summary['false_positive']:,} false positive, "
        f"{summary['false_negative']:,} false negative и "
        f"{summary['true_positive']:,} true positive. False positive rate равен "
        f"{100 * summary['false_positive_rate']:.2f}%, а false negative rate — "
        f"{100 * summary['false_negative_rate']:.2f}%."
    )
    return _paragraph(text, styles["body"])


def _figures(styles: dict[str, ParagraphStyle]) -> list[object]:
    result: list[object] = []
    for filename, caption, width, height in [
        (
            "precision_recall_curves.png",
            "Рисунок 1. Precision-recall curves на test.",
            11 * cm,
            8.25 * cm,
        ),
        (
            "confusion_matrix.png",
            "Рисунок 2. Confusion matrix выбранной модели на test.",
            9.5 * cm,
            7.2 * cm,
        ),
    ]:
        path = ARTIFACTS_DIR / filename
        if path.exists():
            result.extend(
                [
                    KeepTogether(
                        [
                            Image(path, width=width, height=height),
                            _paragraph(caption, styles["caption"]),
                        ]
                    ),
                    Spacer(1, 4),
                ]
            )
    return result


def _render_markdown(styles: dict[str, ParagraphStyle]) -> list[object]:
    lines = REPORT_SOURCE.read_text(encoding="utf-8").splitlines()
    story: list[object] = []
    paragraph_lines: list[str] = []
    bullets: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_lines:
            story.append(_paragraph(" ".join(paragraph_lines), styles["body"]))
            paragraph_lines.clear()

    def flush_bullets() -> None:
        if bullets:
            items = [ListItem(_paragraph(item, styles["body"])) for item in bullets]
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=16))
            story.append(Spacer(1, 4))
            bullets.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            flush_bullets()
        elif stripped == "{{DATASET_TABLE}}":
            flush_paragraph()
            flush_bullets()
            story.append(KeepTogether([_dataset_table(styles), Spacer(1, 8)]))
        elif stripped == "{{METRICS_TABLE}}":
            flush_paragraph()
            flush_bullets()
            story.extend([_metrics_table(styles), Spacer(1, 8)])
        elif stripped == "{{RELATED_WORK_TABLE}}":
            flush_paragraph()
            flush_bullets()
            story.extend([_related_work_table(styles), Spacer(1, 8)])
        elif stripped == "{{THRESHOLD_TABLE}}":
            flush_paragraph()
            flush_bullets()
            story.extend([_threshold_table(styles), Spacer(1, 8)])
        elif stripped == "{{UNCERTAINTY_TABLE}}":
            flush_paragraph()
            flush_bullets()
            story.extend([_uncertainty_table(styles), Spacer(1, 8)])
        elif stripped == "{{ERROR_ANALYSIS_TABLE}}":
            flush_paragraph()
            flush_bullets()
            story.extend([_error_analysis_table(styles), Spacer(1, 8)])
        elif stripped == "{{BEST_MODEL_SUMMARY}}":
            flush_paragraph()
            flush_bullets()
            story.append(_best_model_summary(styles))
        elif stripped == "{{ABSTRACT_RESULTS}}":
            flush_paragraph()
            flush_bullets()
            story.append(_abstract_results(styles))
        elif stripped == "{{COMPARISON_SUMMARY}}":
            flush_paragraph()
            flush_bullets()
            story.append(_comparison_summary(styles))
        elif stripped == "{{ABLATION_SUMMARY}}":
            flush_paragraph()
            flush_bullets()
            story.append(_ablation_summary(styles))
        elif stripped == "{{TRANSFORMER_SETUP}}":
            flush_paragraph()
            flush_bullets()
            story.append(_transformer_setup(styles))
        elif stripped == "{{THRESHOLD_SUMMARY}}":
            flush_paragraph()
            flush_bullets()
            story.append(_threshold_summary(styles))
        elif stripped == "{{ERROR_COUNTS_SUMMARY}}":
            flush_paragraph()
            flush_bullets()
            story.append(_error_counts_summary(styles))
        elif stripped == "{{FIGURES}}":
            flush_paragraph()
            flush_bullets()
            story.extend(_figures(styles))
        elif stripped.startswith("# "):
            flush_paragraph()
            flush_bullets()
            story.append(_paragraph(stripped[2:], styles["title"]))
        elif stripped.startswith("## "):
            flush_paragraph()
            flush_bullets()
            if stripped == "## Литература":
                story.append(PageBreak())
            story.append(_paragraph(stripped[3:], styles["h1"]))
        elif stripped.startswith("### "):
            flush_paragraph()
            flush_bullets()
            story.append(_paragraph(stripped[4:], styles["h2"]))
        elif stripped.startswith("- "):
            flush_paragraph()
            bullets.append(stripped[2:])
        else:
            flush_bullets()
            paragraph_lines.append(stripped)

    flush_paragraph()
    flush_bullets()
    return story


def _add_page_number(canvas: object, document: object) -> None:
    canvas.saveState()
    canvas.setFont("DejaVu", 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    if document.page > 1:
        canvas.drawString(1.7 * cm, A4[1] - 1.0 * cm, "Определение токсичности комментариев")
        canvas.setStrokeColor(colors.HexColor("#D0D5DD"))
        canvas.line(1.7 * cm, A4[1] - 1.15 * cm, A4[0] - 1.7 * cm, A4[1] - 1.15 * cm)
    canvas.drawCentredString(A4[0] / 2, 1.1 * cm, str(document.page))
    canvas.restoreState()


def main() -> None:
    _register_fonts()
    styles = _styles()
    document = SimpleDocTemplate(
        str(REPORT_OUTPUT),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.7 * cm,
        title="Определение токсичности комментариев",
        author="Екатерина Бойкова",
    )
    document.build(
        _render_markdown(styles),
        onFirstPage=_add_page_number,
        onLaterPages=_add_page_number,
    )
    print(f"Saved report: {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
