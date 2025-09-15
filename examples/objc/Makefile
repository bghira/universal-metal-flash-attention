# Simple Makefile for Objective-C → Swift bridge example

.PHONY: build clean run

build:
	@echo "Building Universal Metal Flash Attention..."
	cd ../.. && swift build --configuration debug
	@echo "Compiling Objective-C → Swift example..."
	clang -O3 \
		-framework Foundation \
		-framework Metal \
		-L../../.build/debug \
		-lMFAFFI \
		-o simple_objc_example \
		simple_main.m

run: build
	@echo "Running Objective-C → Swift benchmark..."
	DYLD_LIBRARY_PATH=../../.build/debug ./simple_objc_example

clean:
	rm -f simple_objc_example
