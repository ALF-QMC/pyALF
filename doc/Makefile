builddir := _build/latex
name := doc
pdf := $(builddir)/$(name).pdf

.PHONY : $(pdf)

$(pdf):
	./set_commit.sh
	jb build --all --config ./_config.yml --builder html .
	jb build --all --config ./_config.yml --builder latex .
	(cd $(builddir) && ../../clean.py $(name).tex $(name).tex)
	make --directory $(builddir) $(name).pdf
	cp $(builddir)/$(name).pdf _build/html
