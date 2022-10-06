## Install requirements (ie docker engine)

```
/bin/bash setup.sh
```

## Build docker image (optional if `benchmark.tar` is already present)

```
/bin/bash build_image.sh
```

## Run the benchmark

Set your hugging face access token as environment variable:
```
export ACCESS_TOKEN=<your-hugging-face-access-token-here>
```

Run the benchmark:
```
/bin/bash run_benchmark.sh
```