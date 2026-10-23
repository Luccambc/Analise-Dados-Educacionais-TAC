"""Decisões metodológicas do estudo, centralizadas em um só lugar."""

AREA = "2"  # Computação (código da área de avaliação da CAPES)

PRAZO_MESES = {"MESTRADO": 24, "DOUTORADO": 48}

GRUPOS = {
    "MESTRADO": {
        "ref": ("2016-03-01", "2017-12-31"),
        "reta_final": ("2018-03-01", "2019-12-31"),
        "curso_pandemia": ("2020-03-01", "2021-12-31"),
    },
    "DOUTORADO": {
        "ref": ("2014-03-01", "2015-12-31"),
        "reta_final": ("2016-03-01", "2017-12-31"),
    },
}

HORIZONTE_MESES = {"MESTRADO": 12, "DOUTORADO": 36}
CORTE_ATIVO_MESES = 12        # ativos 12 meses antes do prazo
LIMITE_CORRECAO_DIAS = 180    # acima disso, reingresso

CORTE_INDICADOR = 6           # titulou até +6 meses após o prazo
CORTES_ROBUSTEZ = [0, 12]

MIN_POR_INSTITUICAO = 20
MIN_POR_PROGRAMA = 10
MIN_CELULA = 10               # não reportar grupos menores

SEMENTE = 42
N_BOOTSTRAP = 2000
ALFA = 0.05
