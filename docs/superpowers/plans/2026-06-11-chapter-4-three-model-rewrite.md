# Chapter 4 Three-Model Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Chapter 4 in a copy of the latest proposal with verified audio results and a fair three-model text-classification comparison.

**Architecture:** Use deterministic Word/OOXML editing to preserve all content outside Chapter 4. Insert styled headings, prose, native Word tables, and verified PNG figures, then render through Microsoft Word and inspect the affected pages.

**Tech Stack:** Python, python-docx, OOXML, Microsoft Word PDF export, PNG visual QA

---

### Task 1: Build the revised Chapter 4 copy

**Files:**
- Read: `C:/Year 3 Sem 3/Thesis/Report/Docs/written doc.docx`
- Create: `C:/Year 3 Sem 3/Thesis/Report/Docs/written doc - Chapter 4 revised.docx`

- [ ] Locate the Chapter 4 and Chapter 5 boundary paragraphs.
- [ ] Preserve Chapter 4's chapter-title paragraph and replace all following Chapter 4 content.
- [ ] Add Sections 4.0 through 4.3 with correct heading styles.
- [ ] Add proper Word tables for real audio, synthetic noise, and three-model results.
- [ ] Insert the pipeline, real-audio, synthetic-noise, and three-model figures.
- [ ] Save the revised copy without modifying the original.

### Task 2: Verify content and layout

**Files:**
- Verify: `C:/Year 3 Sem 3/Thesis/Report/Docs/written doc - Chapter 4 revised.docx`

- [ ] Confirm Chapter 4 contains all three model names and verified metrics.
- [ ] Confirm Chapter 5 and all content outside Chapter 4 remain present.
- [ ] Export the revised document to PDF using Microsoft Word.
- [ ] Render affected pages to PNG.
- [ ] Inspect headings, tables, captions, figure legibility, clipping, and page breaks.
- [ ] Correct any layout defects and repeat rendering.
