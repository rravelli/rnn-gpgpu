OUTPUT := build/main

.PHONY: build
build: 
	nvcc *.cu -lm -pg -o $(OUTPUT)

run: 
	./$(OUTPUT)