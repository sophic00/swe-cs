PDF := report.pdf
TEX := report.tex

.PHONY: all pdf open clean

all: pdf

pdf: $(PDF)

$(PDF): $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)

open: pdf
	firefox $(PDF)

clean:
	rm -f report.aux report.log report.out report.toc $(PDF)
