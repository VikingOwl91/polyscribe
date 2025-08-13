# Simple helper to build and link the plugin into the user's XDG data dir
# Usage:
#   make build
#   make link

PLUGIN := polyscribe-plugin-tubescribe
BIN := ../../target/release/$(PLUGIN)

.PHONY: build link

build:
	cargo build -p $(PLUGIN) --release

link: build
	@DATA_DIR=$${XDG_DATA_HOME:-$$HOME/.local/share}; \
	mkdir -p $$DATA_DIR/polyscribe/plugins; \
	ln -sf "$(CURDIR)/$(BIN)" $$DATA_DIR/polyscribe/plugins/$(PLUGIN); \
	echo "Linked: $$DATA_DIR/polyscribe/plugins/$(PLUGIN) -> $(CURDIR)/$(BIN)"
