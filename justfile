set shell := ["bash", "-cu"]

sync-cpu:
    uv sync --extra runtime-cpu --extra dev

sync-gpu:
    uv sync --extra runtime-gpu --extra notebook --extra dev

test:
    uv run pytest

lint:
    uv run ruff check src tests

infer model image trimap output="artifacts/alpha.png":
    uv run diffmatte-infer-onnx --model "{{model}}" --image "{{image}}" --trimap "{{trimap}}" --output "{{output}}" --provider cpu --seed 0

notebook-gpu:
    uv run --extra runtime-gpu --extra notebook jupyter lab notebooks/onnxruntime_gpu_retriever_visualization.ipynb
