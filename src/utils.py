"""Funções comuns (bootstrap, delta de Cliff, V de Cramér, Q de Cochran e I²)."""
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

ORDEM_GRUPOS = ["ref", "reta_final", "curso_pandemia"]
ROTULOS_GRUPOS = {
    "MESTRADO": {"ref": "Ingresso 2016-2017", "reta_final": "Ingresso 2018-2019", "curso_pandemia": "Ingresso 2020-2021"},
    "DOUTORADO": {"ref": "Ingresso 2014-2015", "reta_final": "Ingresso 2016-2017"},
}


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


def suprimir_pequenos(df, coluna_n="matriculas"):
    """Retira linhas com menos de MIN_CELULA matrículas (cuidado ético)."""
    return df[df[coluna_n] >= config.MIN_CELULA]
