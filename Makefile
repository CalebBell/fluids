PYTHON_FILES:= $(wildcard fluids/*.py)

COVERAGE:= coverage

path_to_pymodule = $(subst /,.,$(basename $(1)))

unittests: clean run html annotate

run: $(PYTHON_FILES)
	@for f in $^; do \
		echo "$(COVERAGE) run -a --include="fluids/*" $$f"; \
		$(COVERAGE) run -a --include="fluids/*" $$f; \
	done

HTML_DIR:= html_report

html: $(HTML_DIR) run
	$(COVERAGE) html -d $<

ANNOTATED_DIR:= code_coverage
ANNOTATED_FILES:= $(foreach file,$(PYTHON_FILES),$(ANNOTATED_DIR)/$(file),cover)

annotate: $(ANNOTATED_FILES)

$(ANNOTATED_DIR)/%,cover: $(ANNOTATED_DIR) %
	$(COVERAGE) annotate -d $< $(word 2,$^)

clean:
	$(COVERAGE) erase
	rm -rf $(HTML_DIR)
	rm -rf $(ANNOTATED_DIR)

$(ANNOTATED_DIR) $(HTML_DIR):
	mkdir -p $@
