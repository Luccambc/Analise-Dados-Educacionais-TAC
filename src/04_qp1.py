"""Análises da QP1 (país, instituição e região).

Entrada: data/processed/matriculas.parquet
Saídas em results/tabelas/:
    qp1_indicador_nacional.csv   indicador por nível e grupo, com IC 95%
    qp1_desfechos_nacional.csv   distribuição dos desfechos (%)
    qp1_testes_nacional.csv      qui-quadrado, V de Cramér, Mann-Whitney, delta de Cliff
    qp1_robustez.csv             indicador com os cortes 0, 6 e 12 meses
    qp1_instituicoes.csv         diferença por instituição (mestrado), com IC 95%
    qp1_heterogeneidade.csv      Q de Cochran e I²
    qp1_regioes.csv              indicador por região e grupo
Saídas em results/figuras/:
    qp1_curva_acumulada.png
    qp1_forest_instituicoes.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils  # noqa: E402
from utils import config  # noqa: E402

CORES = {"ref": "#52514e", "reta_final": "#2a78d6", "curso_pandemia": "#eb6834"}
ESTILOS = {"ref": "--", "reta_final": "-", "curso_pandemia": "-"}
TEXTO, TEXTO_SEC, GRADE = "#0b0b0b", "#52514e", "#e4e3df"


def titulou_ate(m, corte):
    """Titulou até `corte` meses após o prazo, dentro do horizonte."""
    return m["DESFECHO"].str[0].isin(["1", "2", "3", "4"]) & (m["EXTENSAO_MESES"] <= corte)


def indicador_nacional(m, rng):
    linhas = []
    for (nivel, grupo), g in m.groupby(["NIVEL", "GRUPO"], observed=True):
        x, n = int(g["TITULOU_ATE_CORTE"].sum()), len(g)
        p, lo, hi = utils.ic_bootstrap_proporcao(x, n, rng)
        linhas.append(dict(nivel=nivel, grupo=grupo, matriculas=n, tituladas_ate_corte=x,
                           percentual=round(p, 1), ic_inf=round(lo, 1), ic_sup=round(hi, 1)))
    return utils.suprimir_pequenos(pd.DataFrame(linhas))


def testes_nacionais(m):
    linhas = []
    for nivel in config.GRUPOS:
        x = m[m["NIVEL"] == nivel]
        ref = x[x["GRUPO"] == "ref"]
        for grupo in utils.grupos_expostos(nivel):
            exp = x[x["GRUPO"] == grupo]
            tabela = pd.crosstab(pd.concat([ref, exp])["GRUPO"], pd.concat([ref, exp])["DESFECHO"])
            qui2, gl, p_qui2, v = utils.v_cramer(tabela)

            tit_ref = ref.loc[ref["DESFECHO"].str[0].isin(list("1234")), "QT_MES_TITULACAO"]
            tit_exp = exp.loc[exp["DESFECHO"].str[0].isin(list("1234")), "QT_MES_TITULACAO"]
            _, p_mw, delta = utils.mann_whitney_cliff(tit_exp, tit_ref)

            alfa = utils.alfa_corrigido(nivel)
            linhas.append(dict(
                nivel=nivel, comparacao=f"{grupo} vs ref", alfa_bonferroni=alfa,
                qui2=round(qui2, 1), gl=gl, p_qui2=p_qui2, v_cramer=round(v, 3),
                signif_qui2=p_qui2 < alfa,
                mediana_meses_ref=tit_ref.median(), mediana_meses_exp=tit_exp.median(),
                p_mann_whitney=p_mw, delta_cliff=round(delta, 3), signif_mw=p_mw < alfa,
            ))
    return pd.DataFrame(linhas)


def robustez(m, rng):
    linhas = []
    for corte in sorted(set([config.CORTE_INDICADOR] + config.CORTES_ROBUSTEZ)):
        for nivel in config.GRUPOS:
            x = m[m["NIVEL"] == nivel]
            ok = titulou_ate(x, corte)
            ref = ok[x["GRUPO"] == "ref"]
            for grupo in utils.grupos_expostos(nivel):
                exp = ok[x["GRUPO"] == grupo]
                dif, lo, hi = utils.ic_bootstrap_diferenca(exp.sum(), len(exp), ref.sum(), len(ref), rng)
                _, p, _, _ = stats.chi2_contingency([[exp.sum(), len(exp) - exp.sum()], [ref.sum(), len(ref) - ref.sum()]])
                linhas.append(dict(corte_meses=corte, nivel=nivel, comparacao=f"{grupo} vs ref",
                                   pct_ref=round(100 * ref.mean(), 1), pct_exp=round(100 * exp.mean(), 1),
                                   diferenca_pp=round(dif, 1), ic_inf=round(lo, 1), ic_sup=round(hi, 1),
                                   p=p, signif=p < utils.alfa_corrigido(nivel)))
    return pd.DataFrame(linhas)


def por_instituicao(m, rng):
    """Mestrado: diferença do indicador por instituição, para cada grupo exposto."""
    x = m[m["NIVEL"] == "MESTRADO"]
    contagem = x.groupby(["SG_ENTIDADE_ENSINO", "GRUPO"], observed=True).size().unstack(fill_value=0)
    elegiveis = contagem[(contagem[list(config.GRUPOS["MESTRADO"])] >= config.MIN_POR_INSTITUICAO).all(axis=1)].index

    linhas = []
    for inst in elegiveis:
        g = x[x["SG_ENTIDADE_ENSINO"] == inst]
        ref = g.loc[g["GRUPO"] == "ref", "TITULOU_ATE_CORTE"]
        for grupo in utils.grupos_expostos("MESTRADO"):
            exp = g.loc[g["GRUPO"] == grupo, "TITULOU_ATE_CORTE"]
            dif, lo, hi = utils.ic_bootstrap_diferenca(exp.sum(), len(exp), ref.sum(), len(ref), rng)
            linhas.append(dict(
                instituicao=inst, regiao=g["NM_REGIAO"].iat[0], grupo=grupo,
                n_ref=len(ref), n_exp=len(exp),
                pct_ref=round(100 * ref.mean(), 1), pct_exp=round(100 * exp.mean(), 1),
                diferenca_pp=round(dif, 1), ic_inf=round(lo, 1), ic_sup=round(hi, 1),
                variancia=utils.variancia_diferenca(exp.sum(), len(exp), ref.sum(), len(ref)),
            ))
    inst = pd.DataFrame(linhas)

    het = []
    for grupo, g in inst.groupby("grupo"):
        q, gl, p, i2, media = utils.cochran_q_i2(g["diferenca_pp"] / 100, g["variancia"])
        het.append(dict(grupo=f"{grupo} vs ref", instituicoes=len(g), q_cochran=round(q, 1), gl=gl,
                        p=p, i2_pct=round(i2, 1), diferenca_media_ponderada_pp=round(100 * media, 1)))
    return inst.drop(columns="variancia"), pd.DataFrame(het)


def por_regiao(m):
    t = (m.groupby(["NIVEL", "NM_REGIAO", "GRUPO"], observed=True)["TITULOU_ATE_CORTE"]
         .agg(matriculas="size", percentual="mean").reset_index())
    t["percentual"] = (100 * t["percentual"]).round(1)
    return utils.suprimir_pequenos(t)


def figura_curva(m):
    fig, eixos = plt.subplots(1, 2, figsize=(7.2, 3), sharey=True)
    for eixo, nivel in zip(eixos, ["MESTRADO", "DOUTORADO"]):
        x = m[m["NIVEL"] == nivel]
        meses = np.arange(-12, config.HORIZONTE_MESES[nivel] + 1)
        for grupo in config.GRUPOS[nivel]:
            g = x[x["GRUPO"] == grupo]
            tit = g.loc[g["DESFECHO"].str[0].isin(list("1234")), "EXTENSAO_MESES"].to_numpy()
            acumulada = [100 * (tit <= k).sum() / len(g) for k in meses]
            eixo.plot(meses, acumulada, color=CORES[grupo], ls=ESTILOS[grupo], lw=2,
                      label=utils.ROTULOS_GRUPOS[nivel][grupo])
        eixo.axvline(0, color=TEXTO_SEC, lw=1)
        eixo.axvline(config.CORTE_INDICADOR, color=TEXTO_SEC, lw=1, ls=":")
        eixo.set_title(nivel.capitalize(), fontsize=10, color=TEXTO)
        eixo.set_xlabel("Meses em relação ao prazo de referência", fontsize=8, color=TEXTO_SEC)
        eixo.grid(axis="y", color=GRADE, lw=0.8)
        eixo.legend(fontsize=7, frameon=False, loc="upper left")
        for lado in ["top", "right"]:
            eixo.spines[lado].set_visible(False)
        eixo.tick_params(labelsize=8, colors=TEXTO_SEC)
    eixos[0].set_ylabel("% de matrículas tituladas", fontsize=8, color=TEXTO_SEC)
    eixos[0].set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(utils.FIGURAS / "qp1_curva_acumulada.png", dpi=300)
    plt.close(fig)


def figura_forest(inst, nacional):
    ordem = (inst[inst["grupo"] == "curso_pandemia"].sort_values("diferenca_pp")["instituicao"].tolist())
    y = {nome: i for i, nome in enumerate(ordem)}
    fig, eixo = plt.subplots(figsize=(4.2, 0.22 * len(ordem) + 1))
    deslocamento = {"reta_final": 0.15, "curso_pandemia": -0.15}
    for grupo in utils.grupos_expostos("MESTRADO"):
        g = inst[inst["grupo"] == grupo]
        ys = g["instituicao"].map(y) + deslocamento[grupo]
        eixo.errorbar(g["diferenca_pp"], ys, xerr=[g["diferenca_pp"] - g["ic_inf"], g["ic_sup"] - g["diferenca_pp"]],
                      fmt="o", ms=4, color=CORES[grupo], elinewidth=1, capsize=0,
                      label=utils.ROTULOS_GRUPOS["MESTRADO"][grupo] + " vs 2016-2017")
        media = nacional.loc[nacional["comparacao"] == f"{grupo} vs ref", "diferenca_pp"].iat[0]
        eixo.axvline(media, color=CORES[grupo], lw=1, ls="--")
    eixo.axvline(0, color=TEXTO_SEC, lw=1)
    eixo.set_ylim(-0.7, len(ordem) - 0.3)
    eixo.set_yticks(range(len(ordem)))
    eixo.set_yticklabels(ordem, fontsize=6, color=TEXTO)
    for rotulo in eixo.get_yticklabels():
        if rotulo.get_text() == "UNB":
            rotulo.set_fontweight("bold")
            eixo.axhspan(y["UNB"] - 0.45, y["UNB"] + 0.45, color=GRADE, zorder=0)
    eixo.set_xlabel("Diferença em pontos percentuais\n(titulação até +6 meses, IC 95%)", fontsize=7, color=TEXTO_SEC)
    eixo.tick_params(axis="x", labelsize=7, colors=TEXTO_SEC)
    eixo.legend(fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
    for lado in ["top", "right"]:
        eixo.spines[lado].set_visible(False)
    fig.tight_layout()
    fig.savefig(utils.FIGURAS / "qp1_forest_instituicoes.png", dpi=300)
    plt.close(fig)


def main():
    rng = utils.gerador()
    m = utils.carregar_matriculas()

    indicador = indicador_nacional(m, rng)
    desfechos = pd.crosstab([m["NIVEL"], m["GRUPO"]], m["DESFECHO"], normalize="index").mul(100).round(1)
    testes = testes_nacionais(m)
    rob = robustez(m, rng)
    inst, het = por_instituicao(m, rng)
    regioes = por_regiao(m)

    indicador.to_csv(utils.TABELAS / "qp1_indicador_nacional.csv", index=False)
    desfechos.to_csv(utils.TABELAS / "qp1_desfechos_nacional.csv")
    testes.to_csv(utils.TABELAS / "qp1_testes_nacional.csv", index=False)
    rob.to_csv(utils.TABELAS / "qp1_robustez.csv", index=False)
    inst.to_csv(utils.TABELAS / "qp1_instituicoes.csv", index=False)
    het.to_csv(utils.TABELAS / "qp1_heterogeneidade.csv", index=False)
    regioes.to_csv(utils.TABELAS / "qp1_regioes.csv", index=False)

    figura_curva(m)
    figura_forest(inst, rob[rob["corte_meses"] == config.CORTE_INDICADOR])

    pd.set_option("display.width", 200)
    print(f"Indicador: % tituladas até +{config.CORTE_INDICADOR} meses (IC 95%)")
    print(indicador.to_string(index=False))
    print("\ntestes nacionais")
    print(testes.to_string(index=False))
    print("\nRobustez")
    print(rob.to_string(index=False))
    print(f"\nInstituições elegíveis no mestrado: {inst['instituicao'].nunique()}")
    print(het.to_string(index=False))
    print("\nRegiões")
    print(regioes.to_string(index=False))


if __name__ == "__main__":
    main()
