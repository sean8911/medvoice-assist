from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


SOURCE = Path(r"C:\Year 3 Sem 3\Thesis\Report\Docs\written doc.docx")
OUTPUT = Path(
    r"C:\Year 3 Sem 3\Thesis\Report\Docs"
    r"\written doc - Gantt and milestones revised.docx"
)

GANTT_WIDTHS = [4.5] + [4.5 / 14] * 14
MILESTONE_WIDTHS = [2.35, 4.55, 2.10]

SEMESTER_202604 = [
    ("Proposal scope and research planning", 1, 2),
    ("Literature review and research gap identification", 1, 4),
    ("Dataset collection, inspection, and preprocessing", 3, 5),
    ("Audio preprocessing and Whisper evaluation", 4, 7),
    ("TF-IDF + LR, BERT, and BioBERT experiments", 6, 9),
    ("Results analysis and three-model comparison", 9, 10),
    ("Thesis proposal writing and revision", 2, 12),
    ("Proposal presentation and submission", 13, 14),
]

SEMESTER_202609 = [
    ("System requirements analysis and architecture design", 1, 2),
    ("Dataset expansion and data preparation", 1, 4),
    ("Audio preprocessing and Whisper module refinement", 3, 5),
    ("Classifier training, selection, and integration", 4, 7),
    ("Chatbot backend and safety response development", 6, 9),
    ("User interface and speech-input integration", 8, 10),
    ("End-to-end testing and system evaluation", 10, 12),
    ("Final thesis writing, revision, and demonstration", 11, 14),
]

MILESTONES = [
    (
        "Research foundation established",
        "Research scope, objectives, literature review, and research gap completed",
        "Semester 202604, W4",
    ),
    (
        "Preliminary speech pipeline completed",
        "Audio preprocessing and Whisper WER/CER evaluation completed",
        "Semester 202604, W7",
    ),
    (
        "Classification comparison completed",
        "TF-IDF + LR, fine-tuned BERT, and BioBERT results compared",
        "Semester 202604, W10",
    ),
    (
        "Thesis proposal completed",
        "Revised proposal, presentation slides, and proposal submission completed",
        "Semester 202604, W14",
    ),
    (
        "Medical chatbot prototype completed",
        "Speech input, classifier, chatbot backend, safety responses, and interface integrated",
        "Semester 202609, W9",
    ),
    (
        "Final evaluation and thesis completed",
        "End-to-end testing, final thesis, system demonstration, and submission completed",
        "Semester 202609, W14",
    ),
]


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=75, start=75, bottom=75, end=75):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for side, value in (
        ("top", top),
        ("start", start),
        ("bottom", bottom),
        ("end", end),
    ):
        element = tc_mar.find(qn(f"w:{side}"))
        if element is None:
            element = OxmlElement(f"w:{side}")
            tc_mar.append(element)
        element.set(qn("w:w"), str(value))
        element.set(qn("w:type"), "dxa")


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
    total_width = 9 * 1440
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

    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "0")
    tbl_ind.set(qn("w:type"), "dxa")

    tbl_layout = tbl_pr.find(qn("w:tblLayout"))
    if tbl_layout is None:
        tbl_layout = OxmlElement("w:tblLayout")
        tbl_pr.append(tbl_layout)
    tbl_layout.set(qn("w:type"), "fixed")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        grid_col = OxmlElement("w:gridCol")
        grid_col.set(qn("w:w"), str(width))
        grid.append(grid_col)

    for row in table.rows:
        prevent_row_split(row)
        for index, cell in enumerate(row.cells):
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(widths_dxa[index]))
            tc_w.set(qn("w:type"), "dxa")
            cell.width = Inches(widths_inches[index])
            set_cell_margins(cell)


def set_cell_width_dxa(cell, width):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width))
    tc_w.set(qn("w:type"), "dxa")


def format_paragraph(paragraph, size, bold=False, alignment=None):
    if alignment is not None:
        paragraph.alignment = alignment
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.0
    for run in paragraph.runs:
        run.font.name = "Times New Roman"
        run._element.get_or_add_rPr().rFonts.set(
            qn("w:eastAsia"), "Times New Roman"
        )
        run.font.size = Pt(size)
        run.bold = bold


def format_cell(cell, size=8, bold=False, alignment=WD_ALIGN_PARAGRAPH.CENTER):
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    for paragraph in cell.paragraphs:
        format_paragraph(paragraph, size, bold=bold, alignment=alignment)


def create_gantt_table(document, semester, activities):
    table = document.add_table(rows=10, cols=15)
    table.style = "Table Grid"

    table.cell(0, 0).text = "Activity"
    semester_cell = table.cell(0, 1).merge(table.cell(0, 14))
    semester_cell.text = ""
    semester_paragraph = semester_cell.paragraphs[0]
    semester_run = semester_paragraph.add_run(f"Semester {semester}")
    semester_run.font.highlight_color = WD_COLOR_INDEX.YELLOW

    table.cell(1, 0).text = ""
    for week in range(1, 15):
        table.cell(1, week).text = f"W{week}"

    for row_index, (activity, start_week, end_week) in enumerate(
        activities, start=2
    ):
        table.cell(row_index, 0).text = activity
        for week in range(1, 15):
            table.cell(row_index, week).text = ""
            if start_week <= week <= end_week:
                set_cell_shading(table.cell(row_index, week), "000000")

    set_table_geometry(table, GANTT_WIDTHS)
    set_cell_width_dxa(table.cell(0, 0), 6480)
    set_cell_width_dxa(semester_cell, 6480)
    repeat_header(table.rows[0])
    repeat_header(table.rows[1])

    for row_index, row in enumerate(table.rows):
        for column_index, cell in enumerate(row.cells):
            if row_index == 0:
                format_cell(cell, size=9, bold=True)
            elif row_index == 1:
                set_cell_shading(cell, "E7E6E6")
                format_cell(cell, size=7.5, bold=True)
            elif column_index == 0:
                format_cell(
                    cell,
                    size=8,
                    bold=False,
                    alignment=WD_ALIGN_PARAGRAPH.LEFT,
                )
            else:
                format_cell(cell, size=7.5)

    document.add_paragraph()
    return table


def create_milestones_table(document):
    table = document.add_table(rows=8, cols=3)
    table.style = "Table Grid"

    title_cell = table.cell(0, 0).merge(table.cell(0, 2))
    title_cell.text = "MILESTONES"
    set_cell_shading(title_cell, "D9D9D9")

    headers = ["Milestone", "Key deliverable(s)", "Completion date"]
    for index, header in enumerate(headers):
        table.cell(1, index).text = header
        set_cell_shading(table.cell(1, index), "E7E6E6")

    for row_index, milestone in enumerate(MILESTONES, start=2):
        for column_index, value in enumerate(milestone):
            table.cell(row_index, column_index).text = value

    set_table_geometry(table, MILESTONE_WIDTHS)
    set_cell_width_dxa(title_cell, 12960)
    repeat_header(table.rows[0])
    repeat_header(table.rows[1])

    for row_index, row in enumerate(table.rows):
        for column_index, cell in enumerate(row.cells):
            if row_index <= 1:
                format_cell(cell, size=9, bold=True)
            else:
                alignment = (
                    WD_ALIGN_PARAGRAPH.CENTER
                    if column_index == 2
                    else WD_ALIGN_PARAGRAPH.LEFT
                )
                format_cell(cell, size=8.5, alignment=alignment)

    return table


def add_page_break(document):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.add_run().add_break(WD_BREAK.PAGE)


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    document = Document(SOURCE)
    appendix_heading = None
    for paragraph in document.paragraphs:
        if "GANTT CHART AND MILESTONES" in " ".join(paragraph.text.split()):
            appendix_heading = paragraph
            break
    if appendix_heading is None:
        raise ValueError("Appendix Gantt heading was not found.")

    body = document.element.body
    current = appendix_heading._element.getnext()
    while current is not None and current.tag != qn("w:sectPr"):
        following = current.getnext()
        body.remove(current)
        current = following

    create_gantt_table(document, "202604", SEMESTER_202604)
    add_page_break(document)
    create_gantt_table(document, "202609", SEMESTER_202609)
    add_page_break(document)
    create_milestones_table(document)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
