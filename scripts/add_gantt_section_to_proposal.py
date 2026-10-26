from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches


SOURCE = Path(
    r"C:\Year 3 Sem 3\Thesis\Report\Docs"
    r"\written doc - model architectures revised.docx"
)
OUTPUT = Path(
    r"C:\Year 3 Sem 3\Thesis\Report\Docs"
    r"\written doc - model architectures and gantt section revised.docx"
)


def find_paragraph(document, prefix):
    for paragraph in document.paragraphs:
        text = " ".join(paragraph.text.split())
        if text.startswith(prefix):
            return paragraph
    raise ValueError(f"Could not find paragraph starting with {prefix!r}")


def add_before(anchor, paragraph):
    anchor.addprevious(paragraph._element)
    return paragraph


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    document = Document(SOURCE)

    # Remove an existing inserted 3.7 block if this script is run more than once.
    existing = None
    chapter_4 = find_paragraph(document, "CHAPTER 4")
    for paragraph in document.paragraphs:
        if " ".join(paragraph.text.split()).startswith("3.7 Gantt Chart"):
            existing = paragraph
            break
    if existing is not None:
        current = existing._element
        stop = chapter_4._element
        while current is not None and current is not stop:
            following = current.getnext()
            current.getparent().remove(current)
            current = following
        chapter_4 = find_paragraph(document, "CHAPTER 4")

    anchor = chapter_4._element

    heading = document.add_paragraph("3.7 Gantt Chart and Milestones", style="Heading 2")
    heading.paragraph_format.keep_with_next = True
    add_before(anchor, heading)

    body = document.add_paragraph(
        "The project schedule and major milestones are presented in the appendix "
        "under the section titled \"Gantt Chart and Milestones\". The Gantt charts "
        "outline the planned activities for Semester 202604 and Semester 202609, "
        "including proposal development, audio preprocessing, model comparison, "
        "chatbot prototype development, system testing, thesis writing, and final "
        "submission. The milestones table summarizes the key deliverables and "
        "expected completion period for each major stage of the project."
    )
    body.style = "Normal"
    body.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    body.paragraph_format.first_line_indent = Inches(0.5)
    body.paragraph_format.keep_together = True
    add_before(anchor, body)

    spacer = document.add_paragraph("")
    spacer.style = "Normal"
    add_before(anchor, spacer)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
