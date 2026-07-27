# Wildlife Camera-Trap Image Classification

Minimal Python project scaffold for a university research assistant / Work Learn
portfolio project focused on wildlife camera-trap image classification.

The first milestone is a ResNet18 transfer-learning baseline using PyTorch and
torchvision. Dataset-specific loading, model training, active learning, user
interfaces, and database work are intentionally out of scope for this initial
scaffold.

## Project Structure

```text
src/data/
src/models/
src/training/
src/utils/
tests/
configs/
notebooks/
data/raw/
data/processed/
data/metadata/
```

## Development

This project targets Python 3.11.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run checks:

```bash
python -m pytest
python -m ruff check .
python -m mypy src tests
```
