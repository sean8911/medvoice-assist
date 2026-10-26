# Gantt Chart and Milestones Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the incomplete appendix schedule with readable Gantt charts
for Semesters 202604 and 202609 and a six-row milestones table.

**Architecture:** Use `python-docx` and direct OOXML formatting to remove only
the appendix elements after the existing heading, then insert three
page-separated native Word tables. Preserve the original proposal and verify
the revised copy through Microsoft Word PDF rendering.

**Tech Stack:** Bundled Python, `python-docx`, OOXML, Microsoft Word COM

---

### Task 1: Build the Appendix Tables

**Files:**
- Create: `scripts/rewrite_gantt_and_milestones.py`
- Read: `C:\Year 3 Sem 3\Thesis\Report\Docs\written doc.docx`
- Create: `C:\Year 3 Sem 3\Thesis\Report\Docs\written doc - Gantt and milestones revised.docx`

- [ ] Locate the appendix heading and delete only the following appendix
  tables and blank paragraphs before the final section properties.
- [ ] Insert the Semester 202604 Gantt table with W1-W14 and eight activities.
- [ ] Insert a page break and the Semester 202609 Gantt table with W1-W14 and
  eight activities.
- [ ] Insert a page break and the six-row milestones table.
- [ ] Apply explicit table geometry, Times New Roman fonts, header shading,
  black active-week cells, cell margins, and non-splitting rows.

### Task 2: Verify Structure and Content

**Files:**
- Verify: `C:\Year 3 Sem 3\Thesis\Report\Docs\written doc - Gantt and milestones revised.docx`

- [ ] Confirm both semester labels exist.
- [ ] Confirm all week labels W1-W14 exist in each Gantt table.
- [ ] Confirm each Gantt table contains eight activities.
- [ ] Confirm the milestones table contains six completed entries.
- [ ] Confirm text before the appendix remains unchanged.

### Task 3: Render and Review

**Files:**
- Render: revised DOCX to an internal PDF
- Inspect: all appendix page PNGs

- [ ] Export the revised document to PDF using Microsoft Word.
- [ ] Determine the appendix page range.
- [ ] Rasterize every appendix page.
- [ ] Inspect table widths, row wrapping, headers, active-week cells, page
  breaks, and margins.
- [ ] Correct and repeat rendering if any layout defect is found.
