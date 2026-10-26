from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


WORKSPACE = Path(r"C:\Year 3 Sem 3\Thesis\Denoise + Whisper")
SOURCE = Path(r"C:\Year 3 Sem 3\Thesis\Report\Docs\written doc.docx")
OUTPUT = Path(
    r"C:\Year 3 Sem 3\Thesis\Report\Docs"
    r"\written doc - Chapter 4 revised.docx"
)

FIGURE_4_1 = WORKSPACE / "chapter4_rewrite_figures" / "figure_4_1_pipeline.png"
FIGURE_4_2 = WORKSPACE / "chapter4_rewrite_figures" / "figure_4_2_real_audio.png"
FIGURE_4_3 = (
    WORKSPACE / "chapter4_rewrite_figures" / "figure_4_3_synthetic_noise.png"
)
FIGURE_4_4 = WORKSPACE / "three_model_comparison.png"

TOTAL_TABLE_WIDTH_DXA = 8640  # 6.0 inches


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=100, bottom=90, end=100):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (
        ("top", top),
        ("start", start),
        ("bottom", bottom),
        ("end", end),
    ):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def prevent_row_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def set_table_geometry(table, widths_inches):
    widths_dxa = [round(width * 1440) for width in widths_inches]
    if sum(widths_dxa) != TOTAL_TABLE_WIDTH_DXA:
        widths_dxa[-1] += TOTAL_TABLE_WIDTH_DXA - sum(widths_dxa)

    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(TOTAL_TABLE_WIDTH_DXA))
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

    tbl_grid = table._tbl.tblGrid
    for child in list(tbl_grid):
        tbl_grid.remove(child)
    for width in widths_dxa:
        grid_col = OxmlElement("w:gridCol")
        grid_col.set(qn("w:w"), str(width))
        tbl_grid.append(grid_col)

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


def style_table(table, font_size=9, first_column_left=False):
    set_repeat_table_header(table.rows[0])
    for row_index, row in enumerate(table.rows):
        for col_index, cell in enumerate(row.cells):
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if row_index == 0:
                set_cell_shading(cell, "D9E2F3")
            for paragraph in cell.paragraphs:
                paragraph.alignment = (
                    WD_ALIGN_PARAGRAPH.LEFT
                    if first_column_left and col_index == 0 and row_index > 0
                    else WD_ALIGN_PARAGRAPH.CENTER
                )
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.0
                for run in paragraph.runs:
                    run.font.name = "Times New Roman"
                    run._element.rPr.rFonts.set(
                        qn("w:eastAsia"), "Times New Roman"
                    )
                    run.font.size = Pt(font_size)
                    if row_index == 0:
                        run.bold = True


def set_keep_with_next(paragraph, value=True):
    paragraph.paragraph_format.keep_with_next = value


def set_keep_together(paragraph, value=True):
    paragraph.paragraph_format.keep_together = value


def find_paragraph(document, prefix):
    for paragraph in document.paragraphs:
        if paragraph.text.strip().startswith(prefix):
            return paragraph
    raise ValueError(f"Could not find paragraph starting with: {prefix}")


def add_before(document, anchor_element, item):
    element = item._element
    anchor_element.addprevious(element)
    return item


def add_heading(document, anchor, text, style):
    paragraph = document.add_paragraph(text, style=style)
    set_keep_with_next(paragraph)
    return add_before(document, anchor, paragraph)


def add_body(document, anchor, text):
    paragraph = document.add_paragraph(text, style="Normal")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.first_line_indent = Inches(0.5)
    paragraph.paragraph_format.keep_together = True
    return add_before(document, anchor, paragraph)


def add_caption(document, anchor, text, table_caption=False):
    paragraph = document.add_paragraph(style="Normal")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.first_line_indent = Inches(0)
    paragraph.paragraph_format.space_before = Pt(3 if table_caption else 0)
    paragraph.paragraph_format.space_after = Pt(4 if table_caption else 8)
    run = paragraph.add_run(text)
    run.italic = True
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(11)
    if table_caption:
        set_keep_with_next(paragraph)
    return add_before(document, anchor, paragraph)


def add_figure(document, anchor, image_path, width_inches):
    paragraph = document.add_paragraph(style="Normal")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.first_line_indent = Inches(0)
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(2)
    set_keep_with_next(paragraph)
    set_keep_together(paragraph)
    run = paragraph.add_run()
    run.add_picture(str(image_path), width=Inches(width_inches))
    return add_before(document, anchor, paragraph)


def add_table(document, anchor, headers, rows, widths, font_size, first_col_left):
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for index, header in enumerate(headers):
        table.rows[0].cells[index].text = header
    for row_data in rows:
        row_cells = table.add_row().cells
        for index, value in enumerate(row_data):
            row_cells[index].text = str(value)
    set_table_geometry(table, widths)
    style_table(table, font_size=font_size, first_column_left=first_col_left)
    return add_before(document, anchor, table)


def main():
    for path in (SOURCE, FIGURE_4_1, FIGURE_4_2, FIGURE_4_3, FIGURE_4_4):
        if not path.exists():
            raise FileNotFoundError(path)

    document = Document(SOURCE)
    chapter_4 = find_paragraph(document, "CHAPTER 4")
    chapter_5 = find_paragraph(document, "CHAPTER 5")
    chapter_5_element = chapter_5._element
    chapter_5.paragraph_format.page_break_before = True

    current = chapter_4._element.getnext()
    while current is not None and current is not chapter_5_element:
        following = current.getnext()
        current.getparent().remove(current)
        current = following

    add_heading(document, chapter_5_element, "4.0 Preliminary Results", "Heading 2")
    add_body(
        document,
        chapter_5_element,
        "This chapter presents preliminary findings from two experiments. "
        "The first evaluates whether audio preprocessing improves Whisper "
        "transcription under a real recording and synthetic Gaussian-noise "
        "conditions. The second compares TF-IDF with Logistic Regression, "
        "fine-tuned BERT, and BioBERT for single-label classification across "
        "24 disease classes. All three classifiers were evaluated using the "
        "same Symptom2Disease dataset and held-out test set.",
    )

    add_heading(
        document, chapter_5_element, "4.1 Audio Preprocessing Evaluation", "Heading 2"
    )
    add_body(
        document,
        chapter_5_element,
        "Figure 4.1 presents the baseline and improved speech-processing "
        "pipelines. Both configurations use the same Whisper model and "
        "downstream text classifier. The improved pipeline adds DC-offset "
        "removal, silence trimming, stationary noise reduction, and amplitude "
        "normalization before transcription. Therefore, audio preprocessing "
        "is the only difference between the two configurations.",
    )
    add_figure(document, chapter_5_element, FIGURE_4_1, 5.8)
    add_caption(
        document,
        chapter_5_element,
        "Figure 4.1 Baseline and improved speech-processing pipelines",
    )

    add_heading(
        document, chapter_5_element, "Real Recording Comparison", "Heading 3"
    )
    add_body(
        document,
        chapter_5_element,
        'The reference sentence for the real recording was "I have a headache '
        'every day in the morning." When the raw audio was processed directly, '
        'Whisper transcribed "headache" as "haydack", producing a Word Error '
        "Rate (WER) of 11.11% and a Character Error Rate (CER) of 9.52%. After "
        "audio preprocessing, the sentence was transcribed correctly, reducing "
        "both WER and CER to 0.00%, as shown in Table 4.1 and Figure 4.2.",
    )
    add_caption(
        document,
        chapter_5_element,
        "Table 4.1 Whisper transcription results for the real recording",
        table_caption=True,
    )
    add_table(
        document,
        chapter_5_element,
        ["Pipeline", "Whisper transcript", "WER (%)", "CER (%)"],
        [
            [
                "Baseline (raw audio)",
                "I have a haydack every day in the morning.",
                "11.11",
                "9.52",
            ],
            [
                "Improved (preprocessed audio)",
                "I have a headache every day in the morning.",
                "0.00",
                "0.00",
            ],
        ],
        [1.45, 3.25, 0.65, 0.65],
        font_size=9,
        first_col_left=True,
    )
    add_figure(document, chapter_5_element, FIGURE_4_2, 5.55)
    add_caption(
        document,
        chapter_5_element,
        "Figure 4.2 WER and CER comparison for the real recording",
    )
    add_body(
        document,
        chapter_5_element,
        "This recording did not contain a verified disease label. Consequently, "
        "it was used only to evaluate transcription quality and was not used "
        "to calculate disease-classification accuracy.",
    )

    add_heading(
        document,
        chapter_5_element,
        "Synthetic-Noise Evaluation",
        "Heading 3",
    )
    add_body(
        document,
        chapter_5_element,
        "Robustness was further evaluated by adding synthetic Gaussian noise at "
        "signal-to-noise ratios (SNRs) of 20, 10, 5, and 0 dB. Both pipelines "
        "produced the correct transcript from 20 to 5 dB. At 0 dB, where the "
        "noise level was most severe, preprocessing reduced WER from 33.33% to "
        "22.22% and CER from 35.71% to 19.05%.",
    )
    add_caption(
        document,
        chapter_5_element,
        "Table 4.2 Whisper performance under synthetic Gaussian noise",
        table_caption=True,
    )
    add_table(
        document,
        chapter_5_element,
        [
            "SNR (dB)",
            "Baseline WER (%)",
            "Improved WER (%)",
            "Baseline CER (%)",
            "Improved CER (%)",
        ],
        [
            ["20", "0.00", "0.00", "0.00", "0.00"],
            ["10", "0.00", "0.00", "0.00", "0.00"],
            ["5", "0.00", "0.00", "0.00", "0.00"],
            ["0", "33.33", "22.22", "35.71", "19.05"],
            ["Mean", "8.33", "5.56", "8.93", "4.76"],
        ],
        [0.85, 1.25, 1.30, 1.25, 1.35],
        font_size=8.7,
        first_col_left=False,
    )
    add_body(
        document,
        chapter_5_element,
        "Across all four noise levels, the mean WER decreased from 8.33% to "
        "5.56%, an improvement of 2.77 percentage points. Mean CER decreased "
        "from 8.93% to 4.76%, an improvement of 4.17 percentage points. These "
        "results indicate that preprocessing was most beneficial under the "
        "most challenging noise condition.",
    )
    add_figure(document, chapter_5_element, FIGURE_4_3, 5.55)
    add_caption(
        document,
        chapter_5_element,
        "Figure 4.3 Mean WER and CER under synthetic Gaussian noise",
    )

    add_heading(
        document,
        chapter_5_element,
        "4.2 Three-Model Text Classification Comparison",
        "Heading 2",
    )
    add_body(
        document,
        chapter_5_element,
        "The text-classification experiment used the Symptom2Disease dataset, "
        "which contains 1,200 symptom descriptions distributed across 24 "
        "disease classes. A stratified split with a fixed random seed of 42 "
        "produced 960 training samples and 240 test samples. TF-IDF with "
        "Logistic Regression provides a sparse lexical baseline, fine-tuned "
        "BERT represents a general-purpose transformer, and BioBERT represents "
        "a biomedical-domain transformer. The same held-out test set and "
        "evaluation metrics were used for all three models.",
    )
    add_caption(
        document,
        chapter_5_element,
        "Table 4.3 Classification performance of the three models (%)",
        table_caption=True,
    )
    add_table(
        document,
        chapter_5_element,
        [
            "Model",
            "Acc.",
            "Macro P",
            "Macro R",
            "Macro F1",
            "Weighted P",
            "Weighted R",
            "Weighted F1",
        ],
        [
            [
                "TF-IDF + LR",
                "96.67",
                "97.15",
                "96.78",
                "96.70",
                "96.99",
                "96.67",
                "96.54",
            ],
            [
                "BioBERT",
                "88.75",
                "90.66",
                "89.00",
                "87.57",
                "91.80",
                "88.75",
                "88.01",
            ],
            [
                "Fine-tuned BERT",
                "87.92",
                "88.27",
                "87.62",
                "86.94",
                "88.58",
                "87.92",
                "87.25",
            ],
        ],
        [1.25, 0.58, 0.68, 0.68, 0.67, 0.72, 0.72, 0.70],
        font_size=8.1,
        first_col_left=True,
    )
    add_body(
        document,
        chapter_5_element,
        "TF-IDF with Logistic Regression achieved the highest result for every "
        "reported metric, including 96.67% accuracy and 96.54% weighted F1. "
        "BioBERT ranked second with 88.75% accuracy and 88.01% weighted F1, "
        "followed by fine-tuned BERT with 87.92% accuracy and 87.25% weighted "
        "F1. The strongest transformer result was therefore 8.53 percentage "
        "points below the TF-IDF model in weighted F1.",
    )
    add_figure(document, chapter_5_element, FIGURE_4_4, 5.85)
    add_caption(
        document,
        chapter_5_element,
        "Figure 4.4 Performance comparison of TF-IDF + LR, BioBERT, and fine-tuned BERT",
    )
    add_body(
        document,
        chapter_5_element,
        "The result does not establish that TF-IDF with Logistic Regression is "
        "universally superior to transformer models. Its stronger performance "
        "is likely related to the small, balanced, and lexically structured "
        "dataset, where disease classes can be distinguished by recurring "
        "symptom terms. Transformer models may benefit more from larger and "
        "more diverse training data, domain-matched pretraining, and further "
        "hyperparameter optimisation. These possibilities should be verified "
        "experimentally in future work.",
    )

    add_heading(
        document,
        chapter_5_element,
        "4.3 Summary of Preliminary Findings",
        "Heading 2",
    )
    add_body(
        document,
        chapter_5_element,
        "The preliminary audio experiment showed that preprocessing improved "
        "Whisper transcription for the real recording and reduced average "
        "error under synthetic noise, particularly at 0 dB. However, the audio "
        "findings are based on one source recording and should be validated "
        "with a larger set of speakers, symptoms, and recording conditions in "
        "Thesis II.",
    )
    add_body(
        document,
        chapter_5_element,
        "For single-label, 24-class disease classification, TF-IDF with "
        "Logistic Regression produced the highest scores on the present "
        "dataset, while BioBERT slightly outperformed fine-tuned BERT. The "
        "three-model comparison provides a consistent basis for evaluating "
        "accuracy, computational cost, latency, and scalability during the "
        "development of the speech-enabled medical chatbot. Full pipeline "
        "integration and broader end-to-end evaluation will be conducted in "
        "Thesis II.",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
