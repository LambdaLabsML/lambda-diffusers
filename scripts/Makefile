bench:
	docker build -t sd-bench .
	docker run \
		--rm -it \
		--gpus all \
		--shm-size=128g \
		--net=host \
		-v $(PWD):/workspace/results \
		sd-bench \
		--steps 30 \
		--samples 1,2,4,8,16,32,64,128 \
		--autocast no \
		--xformers yes \
		--output_file /workspace/results/results.csv

clean:
	rm results.csv