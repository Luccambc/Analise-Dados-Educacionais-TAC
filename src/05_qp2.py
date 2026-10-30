"""Análises da QP2 (nota do programa).

Correlação de Spearman entre a nota do programa e o indicador principal,
no nível do programa, separadamente por nível e grupo.

Entradas: data/processed/matriculas.parquet, data/processed/base.parquet
Saídas:   results/tabelas/qp2_spearman.csv
          results/tabelas/qp2_indicador_por_nota.csv
          results/tabelas/qp2_programas.csv
"""
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils  # noqa: E402
from utils import config  # noqa: E402


def nota_por_programa_e_ano():
    """Nota de cada programa em cada ano-base (a mais frequente nos registros)."""
    base = pd.read_parquet(utils.PROCESSADOS / "base.parquet", columns=["CD_PROGRAMA_IES", "AN_BASE", "CD_CONCEITO_PROGRAMA"])
    base = base[base["CD_CONCEITO_PROGRAMA"].isin(list("34567"))]  # "A" = programa novo, sem nota
    notas = base.groupby(["CD_PROGRAMA_IES", "AN_BASE"])["CD_CONCEITO_PROGRAMA"].agg(lambda s: s.mode().iat[0])
    return notas.astype(int).rename("NOTA").reset_index()


def main():
    m = utils.carregar_matriculas()
    m["AN_BASE"] = m["PRAZO"].dt.year  # nota vigente no ano do prazo de referência
    m = m.drop(columns=["CD_CONCEITO_PROGRAMA"]).merge(nota_por_programa_e_ano(), on=["CD_PROGRAMA_IES", "AN_BASE"], how="left")
    print(f"Matrículas sem nota no ano do prazo (excluídas da QP2): {m['NOTA'].isna().sum()}")
    m = m[m["NOTA"].notna()]

    programas = (
        m.groupby(["NIVEL", "GRUPO", "CD_PROGRAMA_IES"], observed=True)
        .agg(matriculas=("TITULOU_ATE_CORTE", "size"), indicador=("TITULOU_ATE_CORTE", "mean"),
             nota=("NOTA", lambda s: int(s.mode().iat[0])))
        .reset_index()
    )
    programas = programas[programas["matriculas"] >= config.MIN_POR_PROGRAMA]
    programas["indicador"] = (100 * programas["indicador"]).round(1)

    resultados, por_nota = [], []
    for (nivel, grupo), g in programas.groupby(["NIVEL", "GRUPO"], observed=True):
        rho, p = stats.spearmanr(g["nota"], g["indicador"])
        resultados.append(dict(nivel=nivel, grupo=grupo, programas=len(g), rho_spearman=round(rho, 3),
                               p=p, signif=p < config.ALFA))
        resumo = g.groupby("nota")["indicador"].agg(programas="size", mediana_indicador="median").reset_index()
        resumo.insert(0, "grupo", grupo)
        resumo.insert(0, "nivel", nivel)
        por_nota.append(resumo)

    resultados = pd.DataFrame(resultados)
    por_nota = pd.concat(por_nota, ignore_index=True)
    resultados.to_csv(utils.TABELAS / "qp2_spearman.csv", index=False)
    por_nota.to_csv(utils.TABELAS / "qp2_indicador_por_nota.csv", index=False)
    programas.to_csv(utils.TABELAS / "qp2_programas.csv", index=False)

    print("\nSpearman entre nota e indicador (nível do programa)")
    print(resultados.to_string(index=False))
    print("\nMediana do indicador por nota")
    print(por_nota.to_string(index=False))


if __name__ == "__main__":
    main()
