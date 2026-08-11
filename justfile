set shell := ["bash", "-cu"]

sync:
    uv sync --extra dev

test:
    uv run pytest

lint:
    uv run ruff check src tests

infer model image trimap output="artifacts/alpha.png":
    uv run diffmatte-infer-onnx --model "{{model}}" --image "{{image}}" --trimap "{{trimap}}" --output "{{output}}" --provider cpu --seed 0

