"""Lê os CSVs da CAPES, filtra área e modalidade e converte as datas.

Entrada: data/raw/*.csv (um arquivo por ano-base)
Saída:   data/processed/base.parquet
"""
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import config  # noqa: E402

# Só as colunas usadas. Nome, CPF, orientador e título do trabalho
# não são carregados, por serem informações pessoais.
COLUNAS = [
    "AN_BASE",
    "CD_AREA_AVALIACAO",
    "NM_AREA_AVALIACAO",
    "NM_MODALIDADE_PROGRAMA",
    "CD_PROGRAMA_IES",
    "SG_ENTIDADE_ENSINO",
    "DS_DEPENDENCIA_ADMINISTRATIVA",
    "NM_REGIAO",
    "CD_CONCEITO_PROGRAMA",
    "ID_PESSOA",
    "DS_GRAU_ACADEMICO_DISCENTE",
    "NM_SITUACAO_DISCENTE",
    "DT_MATRICULA_DISCENTE",
    "DT_SITUACAO_DISCENTE",
    "QT_MES_TITULACAO",
]

FORMATO_DATA = "%d%b%Y:%H:%M:%S"  # ex.: 19FEB2013:00:00:00


def ler_arquivo(caminho):
    df = pd.read_csv(caminho, sep=";", encoding="latin-1", usecols=COLUNAS, dtype=str)
    # Filtra já na leitura para economizar memória
    df = df[
        (df["CD_AREA_AVALIACAO"] == config.AREA)
        & (df["NM_MODALIDADE_PROGRAMA"] == "ACADÊMICO")
        & (df["DS_GRAU_ACADEMICO_DISCENTE"].isin(["MESTRADO", "DOUTORADO"]))
    ]
    print(f"{caminho.name}: {len(df)} registros no recorte")
    return df


def main():
    arquivos = sorted((RAIZ / "data" / "raw").glob("*.csv"))
    if not arquivos:
        sys.exit("Nenhum CSV encontrado em data/raw/")

    base = pd.concat([ler_arquivo(a) for a in arquivos], ignore_index=True)

    # Conversão de tipos
    base["AN_BASE"] = base["AN_BASE"].astype(int)
    base["DT_MATRICULA"] = pd.to_datetime(base["DT_MATRICULA_DISCENTE"], format=FORMATO_DATA)
    base["DT_SITUACAO"] = pd.to_datetime(base["DT_SITUACAO_DISCENTE"], format=FORMATO_DATA)
    base["QT_MES_TITULACAO"] = pd.to_numeric(base["QT_MES_TITULACAO"], errors="coerce")
    base = base.drop(columns=["DT_MATRICULA_DISCENTE", "DT_SITUACAO_DISCENTE"])

    # A base grafa a mesma situação de dois jeitos
    base["NM_SITUACAO_DISCENTE"] = base["NM_SITUACAO_DISCENTE"].str.replace("MUDANCA", "MUDANÇA")

    saida = RAIZ / "data" / "processed" / "base.parquet"
    base.to_parquet(saida, index=False)

    print(f"\nTotal: {len(base)} registros, {base['CD_PROGRAMA_IES'].nunique()} programas")
    print(f"Anos-base: {sorted(base['AN_BASE'].unique())}")
    print(f"Salvo em {saida}")


if __name__ == "__main__":
    main()
