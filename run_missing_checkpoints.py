import json
import sys
import traceback
from pathlib import Path


def run_notebook(path):
    path = Path(path)
    print(f"\n===== Running {path} =====", flush=True)
    import matplotlib

    matplotlib.use("Agg")
    notebook = json.loads(path.read_text())
    namespace = {
        "__name__": "__main__",
        "__file__": str(path),
    }

    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        print(f"\n--- {path.name} cell {index} ---", flush=True)
        try:
            exec(compile(source, f"{path}:cell-{index}", "exec"), namespace)
        except Exception:
            print(f"Failed while running {path.name} cell {index}", flush=True)
            traceback.print_exc()
            raise

    print(f"\n===== Finished {path} =====", flush=True)


def main():
    notebooks = sys.argv[1:] or ["VGG16.ipynb", "ResNet50.ipynb"]
    for notebook in notebooks:
        run_notebook(notebook)


if __name__ == "__main__":
    main()
