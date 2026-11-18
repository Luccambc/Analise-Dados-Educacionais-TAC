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
        "meio_curso": ("2018-03-01", "2019-12-31"),  # pandemia no meio do doutorado
    },
}

# Prorrogação máxima permitida pelo regulamento da UnB (Resolução CEPE 0080/2017,
# art. 24, parágrafo único). Define o indicador principal: titulou até o prazo
# mais a prorrogação. A mesma regra existe em outras instituições (UFG, UFPB).
PRORROGACAO_MESES = {"MESTRADO": 6, "DOUTORADO": 12}

# Em 2024 (e em 2023, no caso da UFS) a CAPES mudou a grafia da sigla de algumas
# instituições. Sem padronizar, o mesmo programa apareceria como duas instituições.
# Mantém-se a grafia usada de 2017 a 2023.
SIGLAS_EQUIVALENTES = {
    "CEFET-RJ": "CEFET/RJ",
    "UFPI": "FUFPI",
    "UFS": "FUFSE",
    "PUC MINAS": "PUC/MG",
    "PUCPR": "PUC/PR",
    "PUCRS": "PUC/RS",
    "UFPB-JOÃO PESSOA": "UFPB-JP",
    "UNESP-SJ RIO PRETO": "UNESP-SJRP",
    "USP-RIBEIRÃO PRETO": "USP/RP",
    "USP-SÃO CARLOS": "USP/SC",
}

# Meses de acompanhamento após o prazo. É o máximo que a base (até 2024) permite
# para o grupo mais recente de cada nível, aplicado igualmente aos demais.
HORIZONTE_MESES = {"MESTRADO": 12, "DOUTORADO": 36}
# Exceção: o doutorado com ingresso em 2018-2019 tem prazo em 2022-2023 e só pode
# ser acompanhado por 12 meses. Comparações com ele usam esse horizonte comum.
HORIZONTE_EXCECOES = {("DOUTORADO", "meio_curso"): 12}

CORTE_ATIVO_MESES = 12        # ativos 12 meses antes do prazo
LIMITE_CORRECAO_DIAS = 180    # acima disso, reingresso

# Robustez: no prazo (0) e +6 meses nos dois níveis (prorrogação de bolsa da
# Portaria CAPES 121/2020). O corte principal é PRORROGACAO_MESES.
CORTES_ROBUSTEZ = [0, 6, 12]

MIN_POR_INSTITUICAO = 20
MIN_POR_PROGRAMA = 10
MIN_CELULA = 10               # não reportar grupos menores

SEMENTE = 42
N_BOOTSTRAP = 2000
ALFA = 0.05
