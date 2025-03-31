#!/bin/sh

if [ -z "$(git status --porcelain)" ]; then 
    # Working directory clean
    commit_hash="$(git log -1 --format=%h)"
    sed "s/xxxxxxx/$commit_hash/" _templates/sidebar_pdf_commit_.html > _templates/sidebar_pdf_commit.html
    echo "Git commit: $commit_hash" > git_commit.tex
else 
    # Uncommitted changes
    cp _templates/sidebar_pdf_commit_.html _templates/sidebar_pdf_commit.html
    echo "Git commit: xxxxxxx" > git_commit.tex
fi
