"""Roda todas as etapas em ordem."""
import runpy
from pathlib import Path

PASTA = Path(__file__).resolve().parent
for nome in ["01_carregar", "02_verificar", "03_preparar", "04_qp1", "05_qp2", "06_qp3"]:
    print(f"\n{'=' * 20} {nome} {'=' * 20}")
    runpy.run_path(str(PASTA / f"{nome}.py"), run_name="__main__")
