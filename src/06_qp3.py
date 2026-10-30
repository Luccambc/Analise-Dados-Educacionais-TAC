"""Análises da QP3 (UnB versus demais instituições).

Entrada: data/processed/matriculas.parquet
Saídas:  results/tabelas/qp3_unb_vs_demais.csv
         results/tabelas/qp3_variacao.csv
         results/tabelas/qp3_desfechos_unb.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils  # noqa: E402
from utils import config  # noqa: E402

UNB = "UNB"


def contar(df):
    return int(df["TITULOU_ATE_CORTE"].sum()), len(df)


def ic_diferenca_das_variacoes(unb_exp, unb_ref, dem_exp, dem_ref, rng):
    """IC 95% por bootstrap de (variação UnB) - (variação demais), em p.p."""
    def sorteio(x, n):
        return rng.binomial(n, x / n, config.N_BOOTSTRAP) / n

    amostras = (sorteio(*unb_exp) - sorteio(*unb_ref)) - (sorteio(*dem_exp) - sorteio(*dem_ref))
    lo, hi = np.percentile(amostras, [2.5, 97.5])
    ponto = (unb_exp[0] / unb_exp[1] - unb_ref[0] / unb_ref[1]) - (dem_exp[0] / dem_exp[1] - dem_ref[0] / dem_ref[1])
    return 100 * ponto, 100 * lo, 100 * hi


def main():
    rng = utils.gerador()
    m = utils.carregar_matriculas()
    m["UNB"] = m["SG_ENTIDADE_ENSINO"] == UNB

    # 1. Indicador da UnB e das demais em cada grupo
    linhas = []
    for (nivel, grupo), g in m.groupby(["NIVEL", "GRUPO"], observed=True):
        xu, nu = contar(g[g["UNB"]])
        xd, nd = contar(g[~g["UNB"]])
        if nu < config.MIN_CELULA:
            linhas.append(dict(nivel=nivel, grupo=grupo, n_unb=nu, n_demais=nd, observacao="omitido: menos de 10 matrículas"))
            continue
        _, ulo, uhi = utils.ic_bootstrap_proporcao(xu, nu, rng)
        dif, lo, hi = utils.ic_bootstrap_diferenca(xu, nu, xd, nd, rng)
        if nu < config.MIN_POR_INSTITUICAO:  # poucas matrículas: só descritivo, sem IC
            ulo = uhi = lo = hi = np.nan
        linhas.append(dict(
            nivel=nivel, grupo=grupo, n_unb=nu, n_demais=nd,
            pct_unb=round(100 * xu / nu, 1), ic_unb_inf=round(ulo, 1), ic_unb_sup=round(uhi, 1),
            pct_demais=round(100 * xd / nd, 1),
            diferenca_pp=round(dif, 1), ic_inf=round(lo, 1), ic_sup=round(hi, 1),
            observacao="descritivo" if nu < config.MIN_POR_INSTITUICAO else "",
        ))
    comparacao = pd.DataFrame(linhas)

    # 2. Variação entre grupos: UnB versus demais
    linhas = []
    for nivel in config.GRUPOS:
        x = m[m["NIVEL"] == nivel]
        u_ref, d_ref = contar(x[(x["GRUPO"] == "ref") & x["UNB"]]), contar(x[(x["GRUPO"] == "ref") & ~x["UNB"]])
        for grupo in utils.grupos_expostos(nivel):
            u_exp, d_exp = contar(x[(x["GRUPO"] == grupo) & x["UNB"]]), contar(x[(x["GRUPO"] == grupo) & ~x["UNB"]])
            descritivo = min(u_ref[1], u_exp[1]) < config.MIN_POR_INSTITUICAO
            dd, lo, hi = ic_diferenca_das_variacoes(u_exp, u_ref, d_exp, d_ref, rng)
            linhas.append(dict(
                nivel=nivel, comparacao=f"{grupo} vs ref",
                variacao_unb_pp=round(100 * (u_exp[0] / u_exp[1] - u_ref[0] / u_ref[1]), 1),
                variacao_demais_pp=round(100 * (d_exp[0] / d_exp[1] - d_ref[0] / d_ref[1]), 1),
                diferenca_das_variacoes_pp=round(dd, 1),
                ic_inf=None if descritivo else round(lo, 1), ic_sup=None if descritivo else round(hi, 1),
                observacao="descritivo" if descritivo else "",
            ))
    variacao = pd.DataFrame(linhas)

    # 3. Distribuição dos desfechos na UnB
    u = m[m["UNB"]]
    desfechos = pd.crosstab([u["NIVEL"], u["GRUPO"]], u["DESFECHO"], normalize="index").mul(100).round(1)
    tamanhos = u.groupby(["NIVEL", "GRUPO"], observed=True).size()
    desfechos = desfechos[tamanhos.reindex(desfechos.index) >= config.MIN_CELULA]

    comparacao.to_csv(utils.TABELAS / "qp3_unb_vs_demais.csv", index=False)
    variacao.to_csv(utils.TABELAS / "qp3_variacao.csv", index=False)
    desfechos.to_csv(utils.TABELAS / "qp3_desfechos_unb.csv")

    pd.set_option("display.width", 200)
    print("UnB versus demais: % tituladas até +6 meses")
    print(comparacao.to_string(index=False))
    print("\nVariação em relação à referência")
    print(variacao.to_string(index=False))
    print("\nDesfechos na UnB (%)")
    print(desfechos.to_string())


if __name__ == "__main__":
    main()
