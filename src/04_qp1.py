"""Análises da QP1 (país, instituição e região).

Entrada: data/processed/matriculas.parquet
Saídas em results/tabelas/:
    qp1_indicador_nacional.csv   indicador por nível e grupo, com IC 95%
    qp1_desfechos_nacional.csv   distribuição das faixas de desfecho (%)
    qp1_testes_nacional.csv      qui-quadrado, V de Cramér, Mann-Whitney, delta de Cliff
    qp1_robustez.csv             indicador com os cortes 0, 6 e 12 meses
    qp1_meses_ate_metade.csv     meses após o prazo até 25% e 50% do grupo titularem
    qp1_instituicoes.csv         diferença por instituição (mestrado), com IC 95%
    qp1_heterogeneidade.csv      Q de Cochran e I²
    qp1_regioes.csv              indicador por região e grupo
Saídas em results/figuras/:
    qp1_faixas.png
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

CORES = {"ref": "#52514e", "reta_final": "#2a78d6", "meio_curso": "#8a5cc2", "curso_pandemia": "#eb6834"}
ESTILOS = {"ref": "--", "reta_final": "-", "meio_curso": "-", "curso_pandemia": "-"}
CORES_FAIXAS = ["#1f5fae", "#6d9fd8", "#b9d1ee", "#e3a72f", "#52514e", "#a8a7a2", "#d6d5d0"]
TEXTO, TEXTO_SEC, GRADE = "#0b0b0b", "#52514e", "#e4e3df"


def titulou_ate(m, corte):
    """Titulou até `corte` meses após o prazo, dentro do horizonte do grupo."""
    return m["DESFECHO"].isin(utils.FAIXAS_TITULADO) & (m["EXTENSAO_MESES"] <= corte)


def horizonte_comum(nivel, grupo):
    """Horizonte usado ao comparar um grupo com a referência: o menor dos dois."""
    return min(utils.horizonte_meses(nivel, "ref"), utils.horizonte_meses(nivel, grupo))


def indicador_nacional(m, rng):
    linhas = []
    for (nivel, grupo), g in m.groupby(["NIVEL", "GRUPO"], observed=True):
        x, n = int(g["TITULOU_ATE_CORTE"].sum()), len(g)
        p, lo, hi = utils.ic_bootstrap_proporcao(x, n, rng)
        linhas.append(dict(nivel=nivel, grupo=grupo, corte_meses=config.PRORROGACAO_MESES[nivel],
                           matriculas=n, tituladas_ate_corte=x,
                           percentual=round(p, 1), ic_inf=round(lo, 1), ic_sup=round(hi, 1)))
    return utils.suprimir_pequenos(pd.DataFrame(linhas))


def testes_nacionais(m):
    linhas = []
    for nivel in config.GRUPOS:
        x = m[m["NIVEL"] == nivel]
        ref = x[x["GRUPO"] == "ref"]
        for grupo in utils.grupos_expostos(nivel):
            exp = x[x["GRUPO"] == grupo]
            # os dois grupos são reclassificados no mesmo horizonte para serem comparáveis
            h = horizonte_comum(nivel, grupo)
            par = pd.concat([ref, exp])
            faixa, _ = utils.classificar(par, h)
            tabela = pd.crosstab(par["GRUPO"].astype(str), faixa)
            qui2, gl, p_qui2, v = utils.v_cramer(tabela)

            titulado = faixa.isin(utils.FAIXAS_TITULADO)
            tit_ref = par.loc[titulado & (par["GRUPO"] == "ref"), "QT_MES_TITULACAO"]
            tit_exp = par.loc[titulado & (par["GRUPO"] == grupo), "QT_MES_TITULACAO"]
            _, p_mw, delta = utils.mann_whitney_cliff(tit_exp, tit_ref)

            alfa = utils.alfa_corrigido(nivel)
            linhas.append(dict(
                nivel=nivel, comparacao=f"{grupo} vs ref", horizonte_meses=h, alfa_bonferroni=alfa,
                qui2=round(qui2, 1), gl=gl, p_qui2=p_qui2, v_cramer=round(v, 3),
                signif_qui2=p_qui2 < alfa,
                mediana_meses_ref=tit_ref.median(), mediana_meses_exp=tit_exp.median(),
                p_mann_whitney=p_mw, delta_cliff=round(delta, 3), signif_mw=p_mw < alfa,
            ))
    return pd.DataFrame(linhas)


def robustez(m, rng):
    linhas = []
    for corte in sorted(set(config.CORTES_ROBUSTEZ) | set(config.PRORROGACAO_MESES.values())):
        for nivel in config.GRUPOS:
            x = m[m["NIVEL"] == nivel]
            ok = titulou_ate(x, corte)
            ref = ok[x["GRUPO"] == "ref"]
            for grupo in utils.grupos_expostos(nivel):
                exp = ok[x["GRUPO"] == grupo]
                dif, lo, hi = utils.ic_bootstrap_diferenca(exp.sum(), len(exp), ref.sum(), len(ref), rng)
                _, p, _, _ = stats.chi2_contingency([[exp.sum(), len(exp) - exp.sum()], [ref.sum(), len(ref) - ref.sum()]])
                linhas.append(dict(corte_meses=corte, principal=corte == config.PRORROGACAO_MESES[nivel],
                                   nivel=nivel, comparacao=f"{grupo} vs ref",
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


def meses_ate_fracao(m):
    """Meses após o prazo até 25% e 50% de TODO o grupo terem se titulado.

    Ao contrário da média entre titulados, considera todo o grupo no denominador,
    então não ignora quem atrasou além do horizonte. Vazio = não atingido no horizonte.
    """
    linhas = []
    for (nivel, grupo), g in m.groupby(["NIVEL", "GRUPO"], observed=True):
        ext = np.sort(g.loc[g["DESFECHO"].isin(utils.FAIXAS_TITULADO), "EXTENSAO_MESES"].to_numpy())
        linha = dict(nivel=nivel, grupo=grupo, matriculas=len(g), horizonte_meses=utils.horizonte_meses(nivel, grupo))
        for fracao in (0.25, 0.5):
            k = int(np.ceil(fracao * len(g)))
            linha[f"meses_ate_{int(fracao * 100)}pct"] = ext[k - 1] if len(ext) >= k else np.nan
        linha["media_entre_titulados"] = round(ext.mean(), 1)
        linhas.append(linha)
    return pd.DataFrame(linhas)


def rotulo_grupo(nivel, grupo, quebra=" "):
    """Ex.: 'Ingresso 2016-2017 (referência: prazo antes da pandemia)'."""
    return f"{utils.ROTULOS_GRUPOS[nivel][grupo]}{quebra}({utils.DESCRICAO_GRUPOS[grupo]})"


def estilo_eixo(eixo):
    for lado in ["top", "right"]:
        eixo.spines[lado].set_visible(False)
    eixo.tick_params(labelsize=7, colors=TEXTO_SEC)


def anos(meses):
    """24 -> '2 anos', 30 -> '2 anos e meio', 84 -> '7 anos'."""
    inteiro, resto = divmod(meses, 12)
    texto = f"{inteiro} ano" + ("s" if inteiro > 1 else "")
    return texto + (" e meio" if resto == 6 else "" if resto == 0 else f" e {resto} meses")


def figura_faixas(desfechos):
    """Barras empilhadas: como terminou cada grupo, em faixas do regulamento.

    Para a figura, 'mudou de nível' (menos de 1%) e 'sem registro final' viram uma
    só categoria. A tabela qp1_desfechos_nacional.csv mantém as duas separadas.
    """
    pm, pd_ = config.PRAZO_MESES["MESTRADO"], config.PRAZO_MESES["DOUTORADO"]
    rm, rd = config.PRORROGACAO_MESES["MESTRADO"], config.PRORROGACAO_MESES["DOUTORADO"]
    hm, hd = config.HORIZONTE_MESES["MESTRADO"], config.HORIZONTE_MESES["DOUTORADO"]
    categorias = [
        ([utils.FAIXAS[0]], f"até o prazo (mestrado em até {anos(pm)}, doutorado em até {anos(pd_)})", "#1f5fae"),
        ([utils.FAIXAS[1]], f"na prorrogação (mestrado em até {anos(pm + rm)}, doutorado em até {anos(pd_ + rd)})", "#6d9fd8"),
        ([utils.FAIXAS[2]], f"depois da prorrogação (mestrado em até {anos(pm + hm)}, doutorado em até {anos(pd_ + hd)})", "#b9d1ee"),
        ([utils.FAIXAS[3]], "ainda matriculado ao fim do acompanhamento", "#e3a72f"),
        ([utils.FAIXAS[4]], "desligou ou abandonou", "#52514e"),
        (utils.FAIXAS[5:], "outros (mudou de nível ou sumiu da base)", "#d6d5d0"),
    ]
    linhas = [(n, g) for n in utils.ORDEM_NIVEIS for g in utils.ORDEM_GRUPOS if (n, g) in desfechos.index]
    rotulos = [f"{n.capitalize()}, {utils.ROTULOS_GRUPOS[n][g].lower()}\n{utils.DESCRICAO_GRUPOS[g]}"
               + ("*" if (n, g) in config.HORIZONTE_EXCECOES else "") for n, g in linhas]

    fig, eixo = plt.subplots(figsize=(7.2, 0.5 * len(linhas) + 2.3))
    y = np.arange(len(linhas))[::-1].astype(float)
    y[len([l for l in linhas if l[0] == "MESTRADO"]):] -= 0.6  # separa os níveis
    esquerda = np.zeros(len(linhas))
    barras = []
    for faixas, nome, cor in categorias:
        vals = np.array([sum(desfechos.loc[l].get(f, 0.0) for f in faixas) for l in linhas])
        barras.append(eixo.barh(y, vals, left=esquerda, color=cor, height=0.72, edgecolor="white", lw=0.6, label=nome))
        for yi, xi, v in zip(y, esquerda, vals):
            if v >= 5:
                escuro = cor in ("#1f5fae", "#6d9fd8", "#52514e")
                eixo.text(xi + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=6.5,
                          color="white" if escuro else TEXTO)
        esquerda += vals
    eixo.set_yticks(y)
    eixo.set_yticklabels(rotulos, fontsize=7, color=TEXTO)
    eixo.set_xlim(0, 100)
    eixo.set_xlabel("% das matrículas do grupo (cada barra soma 100%)", fontsize=7.5, color=TEXTO_SEC)
    estilo_eixo(eixo)
    eixo.tick_params(axis="y", length=0)
    eixo.spines["left"].set_visible(False)

    # duas legendas: quem titulou (contado desde a matrícula) e quem não titulou
    leg1 = fig.legend(handles=barras[:3], title="Titulou (tempo contado desde a matrícula)", loc="upper left",
                      bbox_to_anchor=(0.01, 0.995), fontsize=6.3, title_fontsize=7, frameon=False, alignment="left")
    fig.legend(handles=barras[3:], title="Não titulou dentro do acompanhamento", loc="upper left",
               bbox_to_anchor=(0.6, 0.995), fontsize=6.3, title_fontsize=7, frameon=False, alignment="left")
    fig.add_artist(leg1)
    nota = (f"Acompanhamento: até {anos(pm + hm)} após a matrícula no mestrado e até {anos(pd_ + hd)} no doutorado, "
            "o máximo que a base (até 2024) permite.")
    if config.HORIZONTE_EXCECOES:
        nota += (f"\n* Acompanhado só até {anos(pd_ + rd)} após a matrícula: quem não titulou até lá aparece como "
                 "ainda matriculado.")
    fig.text(0.01, 0.01, nota, fontsize=6, color=TEXTO_SEC, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    fig.savefig(utils.FIGURAS / "qp1_faixas.png", dpi=300)
    plt.close(fig)


def figura_curva(m):
    """Curva acumulada: % do grupo já titulado, mês a mês, em relação ao prazo."""
    fig, eixos = plt.subplots(1, 2, figsize=(7.2, 4.4), sharey=True)
    for eixo, nivel in zip(eixos, ["MESTRADO", "DOUTORADO"]):
        x = m[m["NIVEL"] == nivel]
        prorrogacao = config.PRORROGACAO_MESES[nivel]
        for grupo in config.GRUPOS[nivel]:
            g = x[x["GRUPO"] == grupo]
            meses = np.arange(-12, utils.horizonte_meses(nivel, grupo) + 1)
            tit = g.loc[g["DESFECHO"].isin(utils.FAIXAS_TITULADO), "EXTENSAO_MESES"].to_numpy()
            acumulada = [100 * (tit <= k).sum() / len(g) for k in meses]
            eixo.plot(meses, acumulada, color=CORES[grupo], ls=ESTILOS[grupo], lw=2,
                      label=rotulo_grupo(nivel, grupo, "\n"))
            if (nivel, grupo) in config.HORIZONTE_EXCECOES:
                eixo.annotate("acompanhado só\naté aqui", (meses[-1], acumulada[-1]), xytext=(meses[-1] + 4, acumulada[-1] - 18),
                              fontsize=6, color=CORES[grupo], arrowprops=dict(arrowstyle="-", color=CORES[grupo], lw=0.8))
        # linhas de referência, com rótulo escrito
        eixo.axvline(0, color=TEXTO_SEC, lw=1)
        eixo.axvline(prorrogacao, color=TEXTO_SEC, lw=1, ls=":")
        eixo.axhline(50, color=TEXTO_SEC, lw=0.6, ls=(0, (1, 3)))
        eixo.text(-0.6, 97, "prazo", fontsize=6.5, color=TEXTO_SEC, ha="right", va="top")
        eixo.text(prorrogacao + 0.6, 97, "fim da\nprorrogação", fontsize=6.5, color=TEXTO_SEC, ha="left", va="top")
        eixo.text(-12, 51.5, "metade do grupo", fontsize=6, color=TEXTO_SEC, ha="left", va="bottom")

        eixo.set_title(f"{nivel.capitalize()} (prazo de {config.PRAZO_MESES[nivel]} meses)", fontsize=9, color=TEXTO)
        eixo.set_xlabel("Meses em relação ao prazo (negativo = antes do prazo)", fontsize=7, color=TEXTO_SEC)
        eixo.grid(axis="y", color=GRADE, lw=0.8)
        # legenda abaixo do gráfico, para não cobrir as curvas
        eixo.legend(fontsize=6.3, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.2), labelspacing=0.7)
        estilo_eixo(eixo)
    eixos[0].set_ylabel("% do grupo já titulado", fontsize=7.5, color=TEXTO_SEC)
    eixos[0].set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(utils.FIGURAS / "qp1_curva_acumulada.png", dpi=300)
    plt.close(fig)


def figura_forest(inst, nacional):
    """Mestrado: quanto o indicador mudou em cada instituição, do ingresso 2016-2017
    para o ingresso 2020-2021 (curso todo na pandemia). Uma linha por instituição.
    A comparação com a reta final fica na tabela qp1_instituicoes.csv.
    """
    grupo = "curso_pandemia"
    corte = config.PRAZO_MESES["MESTRADO"] + config.PRORROGACAO_MESES["MESTRADO"]
    g = inst[inst["grupo"] == grupo].sort_values("diferenca_pp").reset_index(drop=True)
    ys = np.arange(len(g))[::-1]
    unb = g["instituicao"] == "UNB"
    fig, eixo = plt.subplots(figsize=(4.4, 0.155 * len(g) + 1.7))
    eixo.hlines(ys, g["ic_inf"], g["ic_sup"], color="#c9c8c3", lw=1.2, zorder=1)
    eixo.scatter(g["diferenca_pp"], ys, s=16, color=np.where(unb, CORES[grupo], "#52514e"), zorder=2)
    media = nacional.loc[nacional["comparacao"] == f"{grupo} vs ref", "diferenca_pp"].iat[0]
    eixo.axvline(0, color=TEXTO_SEC, lw=1)
    eixo.axvline(media, color=CORES[grupo], lw=1, ls="--")
    eixo.text(media - 1, len(g) - 0.2, f"Brasil: {media:.0f} pontos".replace("-", "−"), fontsize=6,
              color=CORES[grupo], ha="right", va="bottom")
    eixo.text(1, len(g) - 0.2, "sem mudança", fontsize=6, color=TEXTO_SEC, ha="left", va="bottom")
    eixo.set_ylim(-1, len(g) + 1)
    eixo.set_yticks(ys)
    eixo.set_yticklabels(g["instituicao"], fontsize=5.6, color=TEXTO)
    for rotulo in eixo.get_yticklabels():
        if rotulo.get_text() == "UNB":
            rotulo.set_fontweight("bold")
            rotulo.set_color(CORES[grupo])
    eixo.set_xlabel(f"Mudança no % titulado em até {anos(corte)}, em pontos percentuais\n"
                    "← caiu        |        subiu →\n(ponto = mudança, traço cinza = intervalo de confiança de 95%)",
                    fontsize=6.3, color=TEXTO_SEC)
    eixo.set_title("Mestrado: ingresso 2020-2021 (curso todo na pandemia)\ncomparado ao ingresso 2016-2017 (referência)",
                   fontsize=7.5, color=TEXTO)
    estilo_eixo(eixo)
    eixo.tick_params(axis="y", length=0, labelsize=5.6)
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
    metade = meses_ate_fracao(m)

    indicador = utils.ordenar(indicador)
    desfechos = utils.ordenar(desfechos, "NIVEL", "GRUPO")
    regioes = utils.ordenar(regioes, "NIVEL", "GRUPO", apos_nivel=("NM_REGIAO",))
    metade = utils.ordenar(metade)
    desfechos = desfechos.reindex(columns=[f for f in utils.FAIXAS if f in desfechos.columns])
    het = het.sort_values("grupo", key=lambda s: s.str.split().str[0].map(utils.ORDEM_GRUPOS.index))
    inst = inst.sort_values(["instituicao", "grupo"],
                            key=lambda s: s.map(utils.ORDEM_GRUPOS.index) if s.name == "grupo" else s)

    indicador.to_csv(utils.TABELAS / "qp1_indicador_nacional.csv", index=False)
    desfechos.to_csv(utils.TABELAS / "qp1_desfechos_nacional.csv")
    testes.to_csv(utils.TABELAS / "qp1_testes_nacional.csv", index=False)
    rob.to_csv(utils.TABELAS / "qp1_robustez.csv", index=False)
    inst.to_csv(utils.TABELAS / "qp1_instituicoes.csv", index=False)
    het.to_csv(utils.TABELAS / "qp1_heterogeneidade.csv", index=False)
    regioes.to_csv(utils.TABELAS / "qp1_regioes.csv", index=False)
    metade.to_csv(utils.TABELAS / "qp1_meses_ate_metade.csv", index=False)

    figura_faixas(desfechos)
    figura_curva(m)
    principal_mestrado = (rob["nivel"] == "MESTRADO") & rob["principal"]
    figura_forest(inst, rob[principal_mestrado])

    pd.set_option("display.width", 200)
    print(f"Indicador: % tituladas até o prazo + prorrogação {config.PRORROGACAO_MESES} (IC 95%)")
    print(indicador.to_string(index=False))
    print("\nFaixas de desfecho (%)")
    print(desfechos.to_string())
    print("\nTestes nacionais")
    print(testes.to_string(index=False))
    print("\nRobustez")
    print(rob.to_string(index=False))
    print(f"\nInstituições elegíveis no mestrado: {inst['instituicao'].nunique()}")
    print(het.to_string(index=False))
    print("\nRegiões")
    print(regioes.to_string(index=False))
    print("\nMeses após o prazo até 25% e 50% do grupo titularem (vazio = não atingido)")
    print(metade.to_string(index=False))


if __name__ == "__main__":
    main()
