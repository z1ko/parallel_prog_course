build:
	nvcc --std=c++11 -O3 image_blur.cu -Iinclude -o blur

run: build
	./blur
	cmp --silent result_cpu.png result_gpu.png && echo '### SUCCESS: Files Are Identical! ###' || echo '### WARNING: Files Are Different! ###'