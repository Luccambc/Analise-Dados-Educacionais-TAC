"""Funções comuns (faixas de desfecho, bootstrap, delta de Cliff, V de Cramér, Q de Cochran e I²)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import config  # noqa: E402

TABELAS = RAIZ / "results" / "tabelas"
FIGURAS = RAIZ / "results" / "figuras"
PROCESSADOS = RAIZ / "data" / "processed"

ORDEM_GRUPOS = ["ref", "reta_final", "meio_curso", "curso_pandemia"]
ROTULOS_GRUPOS = {
    "MESTRADO": {"ref": "Ingresso 2016-2017", "reta_final": "Ingresso 2018-2019", "curso_pandemia": "Ingresso 2020-2021"},
    "DOUTORADO": {"ref": "Ingresso 2014-2015", "reta_final": "Ingresso 2016-2017", "meio_curso": "Ingresso 2018-2019"},
}

# Faixas de desfecho, alinhadas ao regulamento (prazo e prorrogação)
FAIXAS = [
    "1 até o prazo",
    "2 na prorrogação",
    "3 depois da prorrogação",
    "4 ainda matriculado ao fim do acompanhamento",
    "5 desligou ou abandonou",
    "6 mudança de nível",
    "7 sem registro final",
]
FAIXAS_TITULADO = FAIXAS[:3]
ULTIMO_ANO_BASE = 2024


def horizonte_meses(nivel, grupo):
    """Meses de acompanhamento após o prazo para um grupo."""
    return config.HORIZONTE_EXCECOES.get((nivel, grupo), config.HORIZONTE_MESES[nivel])


def classificar(m, horizonte):
    """Faixa de desfecho de cada matrícula, dado o horizonte em meses após o prazo.

    `horizonte` pode ser um número (mesmo para todos) ou uma Series alinhada a `m`.
    Retorna (faixa, titulou_ate_prorrogacao).
    """
    sit = m["NM_SITUACAO_DISCENTE"]
    h = pd.Series(np.broadcast_to(np.asarray(horizonte), len(m)), index=m.index)
    fim = pd.Series(pd.NaT, index=m.index, dtype="datetime64[ns]")
    for meses in h.unique():
        sel = h == meses
        fim[sel] = m.loc[sel, "PRAZO"] + pd.DateOffset(months=int(meses))
    no_horizonte = m["DT_SITUACAO"] <= fim
    extensao = m["QT_MES_TITULACAO"] - m["PRAZO_MESES"]
    prorrogacao = m["NIVEL"].map(config.PRORROGACAO_MESES)
    # Titulação no horizonte medida pelo tempo de titulação da CAPES, a mesma medida
    # das faixas. Pelas datas, a diferença é de no máximo 1 mês, o que cortaria de
    # forma desigual o grupo cujo horizonte termina no último mês da base.
    titulou = (sit == "TITULADO") & (extensao <= h)
    sumiu = (sit == "MATRICULADO") & (m["AN_BASE"] < np.minimum(ULTIMO_ANO_BASE, fim.dt.year))

    faixa = pd.Series(np.select(
        [
            titulou & (extensao <= 0),
            titulou & (extensao <= prorrogacao),
            titulou,
            sit.isin(["DESLIGADO", "ABANDONOU"]) & no_horizonte,
            (sit == "MUDANÇA DE NÍVEL SEM DEFESA") & no_horizonte,
            sumiu,
        ],
        [FAIXAS[0], FAIXAS[1], FAIXAS[2], FAIXAS[4], FAIXAS[5], FAIXAS[6]],
        default=FAIXAS[3],  # continuava matriculado (ou só saiu depois do horizonte)
    ), index=m.index)
    return faixa, titulou & (extensao <= prorrogacao)


def carregar_matriculas():
    m = pd.read_parquet(PROCESSADOS / "matriculas.parquet")
    m["GRUPO"] = pd.Categorical(m["GRUPO"], categories=ORDEM_GRUPOS, ordered=True)
    return m


def grupos_expostos(nivel):
    """Grupos comparados à referência em cada nível."""
    return [g for g in config.GRUPOS[nivel] if g != "ref"]


def alfa_corrigido(nivel):
    """Bonferroni: divide o nível de significância pelo número de comparações."""
    return config.ALFA / len(grupos_expostos(nivel))


def gerador():
    return np.random.default_rng(config.SEMENTE)


def ic_bootstrap_diferenca(x_exp, n_exp, x_ref, n_ref, rng, n_boot=None):
    """IC 95% por bootstrap da diferença de proporções (exposto - referência).

    Reamostrar n matrículas com reposição de um grupo com proporção p equivale a
    sortear uma binomial(n, p); por isso o sorteio é feito diretamente assim.
    Retorna (diferença, limite inferior, limite superior) em pontos percentuais.
    """
    n_boot = n_boot or config.N_BOOTSTRAP
    p_exp, p_ref = x_exp / n_exp, x_ref / n_ref
    amostras = rng.binomial(n_exp, p_exp, n_boot) / n_exp - rng.binomial(n_ref, p_ref, n_boot) / n_ref
    lo, hi = np.percentile(amostras, [2.5, 97.5])
    return 100 * (p_exp - p_ref), 100 * lo, 100 * hi


def ic_bootstrap_proporcao(x, n, rng, n_boot=None):
    """IC 95% por bootstrap de uma proporção, em pontos percentuais."""
    n_boot = n_boot or config.N_BOOTSTRAP
    amostras = rng.binomial(n, x / n, n_boot) / n
    lo, hi = np.percentile(amostras, [2.5, 97.5])
    return 100 * x / n, 100 * lo, 100 * hi


def v_cramer(tabela):
    """Qui-quadrado e V de Cramér de uma tabela de contingência."""
    tabela = tabela.loc[:, tabela.sum(axis=0) > 0]  # remove categorias vazias
    qui2, p, gl, _ = stats.chi2_contingency(tabela)
    n = tabela.to_numpy().sum()
    v = np.sqrt(qui2 / (n * (min(tabela.shape) - 1)))
    return qui2, gl, p, v


def mann_whitney_cliff(exposto, referencia):
    """Mann-Whitney e delta de Cliff (positivo = exposto com valores maiores)."""
    exposto, referencia = np.asarray(exposto), np.asarray(referencia)
    u, p = stats.mannwhitneyu(exposto, referencia, alternative="two-sided")
    delta = 2 * u / (len(exposto) * len(referencia)) - 1
    return u, p, delta


def cochran_q_i2(diferencas, variancias):
    """Q de Cochran e I² para um conjunto de estimativas independentes."""
    d, v = np.asarray(diferencas, float), np.asarray(variancias, float)
    w = 1 / v
    media = np.sum(w * d) / np.sum(w)
    q = np.sum(w * (d - media) ** 2)
    gl = len(d) - 1
    p = stats.chi2.sf(q, gl)
    i2 = max(0.0, (q - gl) / q) * 100 if q > 0 else 0.0
    return q, gl, p, i2, media


def variancia_diferenca(x_exp, n_exp, x_ref, n_ref):
    """Variância da diferença de proporções, com correção de 0,5 para evitar zero."""
    p1 = (x_exp + 0.5) / (n_exp + 1)
    p2 = (x_ref + 0.5) / (n_ref + 1)
    return p1 * (1 - p1) / n_exp + p2 * (1 - p2) / n_ref


ORDEM_NIVEIS = ["MESTRADO", "DOUTORADO"]


def ordenar(df, nivel="nivel", grupo="grupo", apos_nivel=(), depois=()):
    """Ordena por nível (mestrado primeiro) e grupo (cronológico).

    Aceita os nomes das colunas ou dos níveis do índice.
    """
    no_indice = nivel in (df.index.names or [])
    if no_indice:
        nomes_indice = list(df.index.names)
        df = df.reset_index()
    chaves = {
        nivel: lambda s: s.map({n: i for i, n in enumerate(ORDEM_NIVEIS)}),
        grupo: lambda s: s.astype(str).map({g: i for i, g in enumerate(ORDEM_GRUPOS)}),
    }
    colunas = [nivel, *apos_nivel, grupo, *depois]
    df = df.sort_values(colunas, key=lambda s: chaves.get(s.name, lambda x: x)(s))
    return df.set_index(nomes_indice) if no_indice else df.reset_index(drop=True)


def suprimir_pequenos(df, coluna_n="matriculas"):
    """Retira linhas com menos de MIN_CELULA matrículas (cuidado ético)."""
    return df[df[coluna_n] >= config.MIN_CELULA]
