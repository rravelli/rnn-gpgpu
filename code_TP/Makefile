OUTPUT := build/main
epoch := 3

.PHONY: build
build: 
	nvcc *.cu -lm -pg -o $(OUTPUT)

run: 
	./$(OUTPUT) $(epoch):=1