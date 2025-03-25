build:
	nvcc --std=c++11 image_grayscale.cu -Iinclude -o grayscale

run: build
	./grayscale
	cmp --silent result_cpu.png result_gpu.png && echo '### SUCCESS: Files Are Identical! ###' || echo '### WARNING: Files Are Different! ###'