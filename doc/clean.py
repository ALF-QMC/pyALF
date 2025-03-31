#!/usr/bin/env python3
"""Script for cleaning up the LaTeX source."""
# pylint: disable=line-too-long

import sys

infile = sys.argv[1]
outfile = sys.argv[2]

def replace_lines(orig_lines, search_str, insert_lines, N_lines_before=0, N_lines_after=0):
    for i, line in enumerate(orig_lines):
        if search_str in line:
            i_line = i
            break
    return orig_lines[:i_line-N_lines_before] + [insert_lines] + orig_lines[i_line+1+N_lines_after:]

with open(infile, encoding="UTF-8") as f:
    lines = f.readlines()

# Protect hyperliks, such that they work (within captions?)
lines = [s.replace(r'\hyperlink', r'\protect\hyperlink') for s in lines]

# Remove pagestyle defs and TOC. We will add them ourselves
lines = [s.replace(r'\pagestyle{plain}', r'%\pagestyle{plain}') for s in lines]
lines = [s.replace(r'\sphinxtableofcontents', r'%\sphinxtableofcontents') for s in lines]
lines = [s.replace(r'\pagestyle{normal}', r'%\pagestyle{normal}') for s in lines]

lines = replace_lines(
    lines,
    r'\chapter{Prerequisites and installation}',
    r"""
\tableofcontents \addcontentsline{toc}{chapter}{Contents}
\pagestyle{normal}
\cleardoublepage
\pagenumbering{arabic}
\chapter{Prerequisites and installation}
""")

# # Remove awkward numbering of footnotes
# lines = [re.sub(r'begin{footnote}\[.*\]', r'begin{footnote}', s) for s in lines]

# Replace widgets with images
for name in ['{warmup_widget}.png', '{rebin_widget}.png']:
    lines = replace_lines(lines, "VBox", r"\noindent\sphinxincludegraphics{"+name+"}"+"\n", 1, 1)

# Set pagestyle back to normal for bibliography
lines = replace_lines(
    lines, r'\begin{sphinxthebibliography}', r"""\pagestyle{normal}
""", -1, 0)

# Remove chapter "Bibliography"
lines = replace_lines(lines, r"\chapter{Bibliography}", '', 0, 1)

# Remove footnotes in bibliography
for i, line in enumerate(lines):
    if r'\begin{sphinxthebibliography}' in line:
        i_start = i
        break
for i in range(i_start, len(lines)):
    if r'\begin{footnote}\sphinxAtStartFootnote' in lines[i]:
        lines[i] = '%\n'
        lines[i+1] = '%\n'
        lines[i+2] = '%\n'
        lines[i+3] = lines[i+3].replace(r'\end{footnote}', '')

with open(outfile, 'w', encoding="UTF-8") as f:
    for line in lines:
        f.write(line)
