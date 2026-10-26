from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


SOURCE = Path(r"C:\Year 3 Sem 3\Thesis\Report\Docs\written doc.docx")
OUTPUT = Path(
    r"C:\Year 3 Sem 3\Thesis\Report\Docs"
    r"\written doc - model architectures revised.docx"
)
FIGURE_DIR = Path(r"C:\Year 3 Sem 3\Thesis\Denoise + Whisper\chapter3_figures")
FIGURE_PATH = FIGURE_DIR / "figure_3_6_model_architectures.png"


def font(size, bold=False):
    candidates = [
        r"C:\Windows\Fonts\timesbd.ttf" if bold else r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_centered(draw, xy, text, fnt, fill=(0, 0, 0), max_width=None, line_gap=4):
    x1, y1, x2, y2 = xy
    if max_width is None:
        max_width = x2 - x1 - 24
    words = text.split()
    lines = []
    line = ""
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textbbox((0, 0), trial, font=fnt)[2] <= max_width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    line_heights = [
        draw.textbbox((0, 0), current, font=fnt)[3]
        - draw.textbbox((0, 0), current, font=fnt)[1]
        for current in lines
    ]
    total_height = sum(line_heights) + line_gap * max(0, len(lines) - 1)
    y = y1 + (y2 - y1 - total_height) / 2
    for current, height in zip(lines, line_heights):
        bbox = draw.textbbox((0, 0), current, font=fnt)
        width = bbox[2] - bbox[0]
        draw.text((x1 + (x2 - x1 - width) / 2, y), current, font=fnt, fill=fill)
        y += height + line_gap


def rounded_box(draw, xy, fill, outline=(40, 40, 40), radius=18):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=3)


def arrow(draw, start, end, fill=(70, 70, 70)):
    draw.line([start, end], fill=fill, width=5)
    x1, y1 = start
    x2, y2 = end
    if x2 >= x1:
        points = [(x2, y2), (x2 - 22, y2 - 12), (x2 - 22, y2 + 12)]
    else:
        points = [(x2, y2), (x2 + 22, y2 - 12), (x2 + 22, y2 + 12)]
    draw.polygon(points, fill=fill)


def create_architecture_figure():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 2200, 1420
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    title_font = font(48, bold=True)
    label_font = font(31, bold=True)
    text_font = font(27)
    note_font = font(26)

    draw.text(
        (width / 2, 55),
        "Architecture of the Three Disease-Classification Models",
        font=title_font,
        anchor="ma",
        fill=(0, 0, 0),
    )

    rows = [
        (
            "TF-IDF + LR",
            [
                "Cleaned symptom text",
                "TF-IDF vectorizer",
                "Sparse feature vector",
                "Logistic Regression",
                "24 disease classes",
            ],
            (219, 234, 254),
            (31, 78, 121),
        ),
        (
            "Fine-tuned BERT",
            [
                "Symptom text",
                "BERT tokenizer",
                "BERT encoder + [CLS]",
                "Linear classifier + softmax",
                "24 disease classes",
            ],
            (226, 239, 218),
            (49, 105, 48),
        ),
        (
            "Fine-tuned BioBERT",
            [
                "Symptom text",
                "BioBERT tokenizer",
                "BioBERT encoder + [CLS]",
                "Linear classifier + softmax",
                "24 disease classes",
            ],
            (252, 229, 214),
            (143, 75, 42),
        ),
    ]

    label_x1, label_x2 = 95, 360
    box_w, box_h = 300, 120
    gap = 66
    start_x = 410
    y_positions = [245, 565, 885]

    for row_index, (name, steps, fill, accent) in enumerate(rows):
        y = y_positions[row_index]
        rounded_box(draw, (label_x1, y + 18, label_x2, y + box_h - 18), (245, 245, 245), accent)
        draw_centered(draw, (label_x1, y + 18, label_x2, y + box_h - 18), name, label_font, accent)

        previous = None
        for step_index, step in enumerate(steps):
            x1 = start_x + step_index * (box_w + gap)
            x2 = x1 + box_w
            y1, y2 = y, y + box_h
            rounded_box(draw, (x1, y1, x2, y2), fill, accent)
            draw_centered(draw, (x1 + 10, y1 + 5, x2 - 10, y2 - 5), step, text_font)
            if previous is not None:
                arrow(draw, (previous[2] + 10, (previous[1] + previous[3]) / 2), (x1 - 15, (y1 + y2) / 2))
            previous = (x1, y1, x2, y2)

    legend_y = 1235
    notes = [
        ("Shared evaluation:", "same Symptom2Disease dataset, same train-test split, same metrics"),
        ("Main difference:", "lexical TF-IDF features vs. general BERT transformer vs. biomedical BioBERT transformer"),
    ]
    for i, (prefix, body) in enumerate(notes):
        y = legend_y + i * 58
        draw.text((150, y), prefix, font=font(28, bold=True), fill=(0, 0, 0))
        draw.text((410, y), body, font=note_font, fill=(0, 0, 0))

    img.save(FIGURE_PATH, quality=95)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=85, start=85, bottom=85, end=85):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def prevent_row_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def set_table_geometry(table, widths_inches):
    widths_dxa = [round(width * 1440) for width in widths_inches]
    total_width = 6 * 1440
    widths_dxa[-1] += total_width - sum(widths_dxa)

    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(total_width))
    tbl_w.set(qn("w:type"), "dxa")

    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)

    for row in table.rows:
        prevent_row_split(row)
        for index, cell in enumerate(row.cells):
            cell.width = Inches(widths_inches[index])
            set_cell_margins(cell)
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(widths_dxa[index]))
            tc_w.set(qn("w:type"), "dxa")


def format_cell(cell, size=8.2, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT):
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    for paragraph in cell.paragraphs:
        paragraph.alignment = align
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.line_spacing = 1.0
        for run in paragraph.runs:
            run.font.name = "Times New Roman"
            run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Times New Roman")
            run.font.size = Pt(size)
            run.bold = bold


def find_paragraph(document, prefix):
    for paragraph in document.paragraphs:
        if " ".join(paragraph.text.split()).startswith(prefix):
            return paragraph
    raise ValueError(f"Could not find paragraph starting with {prefix!r}")


def add_before(anchor, item):
    anchor.addprevious(item._element)
    return item


def add_body(document, anchor, text):
    paragraph = document.add_paragraph(text, style="Normal")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.first_line_indent = Inches(0.5)
    paragraph.paragraph_format.keep_together = True
    return add_before(anchor, paragraph)


def add_caption(document, anchor, text, table_caption=False):
    paragraph = document.add_paragraph(style="Normal")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.first_line_indent = Inches(0)
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(6)
    if table_caption:
        paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run(text)
    run.italic = True
    run.font.name = "Times New Roman"
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(11)
    return add_before(anchor, paragraph)


def add_figure(document, anchor):
    paragraph = document.add_paragraph(style="Normal")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.first_line_indent = Inches(0)
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.keep_together = True
    paragraph.add_run().add_picture(str(FIGURE_PATH), width=Inches(6.0))
    return add_before(anchor, paragraph)


def add_architecture_table(document, anchor):
    rows = [
        [
            "TF-IDF + LR",
            "Cleaned symptom text is converted into TF-IDF sparse vectors.",
            "Logistic Regression learns class weights from term-frequency features.",
            "Traditional machine-learning baseline for the same 24-class task.",
        ],
        [
            "Fine-tuned BERT",
            "Symptom text is tokenized using WordPiece tokens.",
            "Pre-trained BERT encoder produces the [CLS] representation, followed by a classification layer.",
            "General transformer comparison model.",
        ],
        [
            "Fine-tuned BioBERT",
            "Symptom text is tokenized using the BioBERT tokenizer.",
            "BioBERT encoder uses biomedical pretraining before fine-tuning with the classification layer.",
            "Biomedical-domain transformer comparison model.",
        ],
    ]
    headers = ["Model", "Input representation", "Main architecture", "Purpose"]
    table = document.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    for i, header in enumerate(headers):
        table.rows[0].cells[i].text = header
    for row_data in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row_data):
            cells[i].text = value

    set_table_geometry(table, [1.05, 1.65, 2.1, 1.2])
    repeat_header(table.rows[0])
    for row_index, row in enumerate(table.rows):
        for column_index, cell in enumerate(row.cells):
            if row_index == 0:
                set_cell_shading(cell, "D9E2F3")
                format_cell(cell, size=8.2, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
            else:
                format_cell(cell, size=7.8, align=WD_ALIGN_PARAGRAPH.LEFT)
    return add_before(anchor, table)


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    create_architecture_figure()

    document = Document(SOURCE)
    anchor = find_paragraph(document, "3.5 Evaluation Metrics")._element

    heading = document.add_paragraph("3.4.1 Architecture of Classification Models", style="Heading 3")
    heading.paragraph_format.keep_with_next = True
    add_before(anchor, heading)

    add_body(
        document,
        anchor,
        "The three classification models share the same input and output structure: "
        "each model receives a cleaned symptom description and predicts one disease "
        "class from the 24-class Symptom2Disease dataset. The main difference is "
        "the way each model represents and learns from the text. TF-IDF with "
        "Logistic Regression uses sparse lexical features, while BERT and BioBERT "
        "use transformer encoders with a classification layer for fine-tuning.",
    )
    add_figure(document, anchor)
    add_caption(
        document,
        anchor,
        "Figure 3.6, Architecture of the three text-classification models",
    )
    add_caption(
        document,
        anchor,
        "Table 3.4, Architecture summary of the three classification models",
        table_caption=True,
    )
    add_architecture_table(document, anchor)
    add_body(
        document,
        anchor,
        "This architecture comparison allows the study to evaluate whether a "
        "traditional lexical model is sufficient for the selected dataset or "
        "whether transformer-based models provide better disease-classification "
        "performance when the same training and testing conditions are used.",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
