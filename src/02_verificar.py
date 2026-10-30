"""Checagens de consistência da base.

Entrada: data/processed/base.parquet
Saída:   results/verificacao.txt (o mesmo texto é impresso na tela)
"""
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import config  # noqa: E402

CHAVE = ["ID_PESSOA", "CD_PROGRAMA_IES", "DS_GRAU_ACADEMICO_DISCENTE"]


def main():
    base = pd.read_parquet(RAIZ / "data" / "processed" / "base.parquet")
    linhas = []

    def registrar(texto=""):
        print(texto)
        linhas.append(texto)

    registrar("VERIFICAÇÕES DE CONSISTÊNCIA")
    registrar(f"Registros: {len(base)} | Programas: {base['CD_PROGRAMA_IES'].nunique()}")

    # 1. Registros por ano-base
    registrar("\n1. Registros por ano-base")
    registrar(base["AN_BASE"].value_counts().sort_index().to_string())

    # 2. QT_MES_TITULACAO bate com as datas?
    tit = base[base["NM_SITUACAO_DISCENTE"] == "TITULADO"]
    meses = (tit["DT_SITUACAO"].dt.year - tit["DT_MATRICULA"].dt.year) * 12 + (
        tit["DT_SITUACAO"].dt.month - tit["DT_MATRICULA"].dt.month
    )
    dif = (tit["QT_MES_TITULACAO"] - meses).abs()
    registrar("\n2. QT_MES_TITULACAO versus diferença entre as datas (titulados)")
    registrar(f"Registros de titulados: {len(tit)}")
    registrar(f"Diferença de até 1 mês: {100 * (dif <= 1).mean():.1f}%")
    registrar(f"Diferença máxima: {dif.max():.0f} meses")

    # 3. Mais de uma data de matrícula para a mesma matrícula
    variacao = base.groupby(CHAVE)["DT_MATRICULA"].agg(lambda s: (s.max() - s.min()).days)
    multiplas = variacao[variacao > 0]
    limite = config.LIMITE_CORRECAO_DIAS
    registrar("\n3. Discente, programa e nível com mais de uma data de matrícula")
    registrar(f"Total: {len(multiplas)} de {len(variacao)}")
    registrar(f"Até {limite} dias (correção): {(multiplas <= limite).sum()}")
    registrar(f"Mais de {limite} dias (reingresso): {(multiplas > limite).sum()}")

    # 4. Nome da área de avaliação ao longo dos anos
    registrar("\n4. Nome da área de avaliação por ano-base")
    registrar(base.groupby("AN_BASE")["NM_AREA_AVALIACAO"].unique().to_string())

    # 5. Situações encontradas
    registrar("\n5. Situações dos discentes")
    registrar(base["NM_SITUACAO_DISCENTE"].value_counts().to_string())

    (RAIZ / "results" / "verificacao.txt").write_text("\n".join(linhas), encoding="utf-8")


if __name__ == "__main__":
    main()
