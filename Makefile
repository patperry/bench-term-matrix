RMDFILE=term-matrix

all: $(RMDFILE).html

$(RMDFILE).html: $(RMDFILE).Rmd
	Rscript -e 'rmarkdown::render("$(RMDFILE).Rmd", output_format=c("html_document", "md_document"))'

clean:
	rm -rf $(RMDFILE)_cache $(RMDFILE)_files $(RMDFILE).html $(RMDFILE).md

.PHONY: all clean
