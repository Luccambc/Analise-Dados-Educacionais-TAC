"""Monta matrículas, prazos, grupos e desfechos.

Entrada: data/processed/base.parquet
Saídas:  data/processed/matriculas.parquet
         results/tabelas/tabela1_matriculas.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import config  # noqa: E402

CHAVE = ["ID_PESSOA", "CD_PROGRAMA_IES", "DS_GRAU_ACADEMICO_DISCENTE"]
EVENTOS = ["TITULADO", "DESLIGADO", "ABANDONOU", "MUDANÇA DE NÍVEL SEM DEFESA"]


def identificar_matriculas(base):
    """Etapa 2: uma linha por matrícula, com a situação mais recente.

    Mudança de data de matrícula de até LIMITE_CORRECAO_DIAS = correção (mesma
    matrícula). Acima disso = reingresso (nova matrícula).
    """
    base = base.sort_values(CHAVE + ["AN_BASE", "DT_SITUACAO"]).copy()
    data_anterior = base.groupby(CHAVE)["DT_MATRICULA"].shift()
    diferenca = (base["DT_MATRICULA"] - data_anterior).dt.days.abs()
    nova = data_anterior.isna() | (diferenca > config.LIMITE_CORRECAO_DIAS)
    base["SEQ_MATRICULA"] = nova.groupby([base[c] for c in CHAVE]).cumsum()

    matriculas = base.groupby(CHAVE + ["SEQ_MATRICULA"], as_index=False).tail(1).copy()
    print(f"Matrículas identificadas: {len(matriculas)}")
    print(f"Reingressos excluídos: {(matriculas['SEQ_MATRICULA'] > 1).sum()}")
    return matriculas[matriculas["SEQ_MATRICULA"] == 1].copy()


def atribuir_grupos(m):
    """Etapas 3 e 4: prazo de referência e grupo de ingresso."""
    m["NIVEL"] = m["DS_GRAU_ACADEMICO_DISCENTE"]
    m["PRAZO_MESES"] = m["NIVEL"].map(config.PRAZO_MESES)
    m["GRUPO"] = None
    for nivel, grupos in config.GRUPOS.items():
        e_nivel = m["NIVEL"] == nivel
        for nome, (inicio, fim) in grupos.items():
            na_janela = e_nivel & m["DT_MATRICULA"].between(inicio, fim)
            m.loc[na_janela, "GRUPO"] = nome

    m = m[m["GRUPO"].notna()].copy()
    for nivel, meses in config.PRAZO_MESES.items():
        e_nivel = m["NIVEL"] == nivel
        m.loc[e_nivel, "PRAZO"] = m.loc[e_nivel, "DT_MATRICULA"] + pd.DateOffset(months=meses)
        m.loc[e_nivel, "CORTE_ATIVO"] = m.loc[e_nivel, "PRAZO"] - pd.DateOffset(months=config.CORTE_ATIVO_MESES)
        m.loc[e_nivel, "HORIZONTE"] = m.loc[e_nivel, "PRAZO"] + pd.DateOffset(months=config.HORIZONTE_MESES[nivel])
    return m


def classificar_desfechos(m):
    """Desfecho de cada matrícula dentro do horizonte de acompanhamento."""
    sit = m["NM_SITUACAO_DISCENTE"]
    no_horizonte = m["DT_SITUACAO"] <= m["HORIZONTE"]
    titulou = (sit == "TITULADO") & no_horizonte
    extensao = m["QT_MES_TITULACAO"] - m["PRAZO_MESES"]
    sumiu = (sit == "MATRICULADO") & (m["AN_BASE"] < np.minimum(2024, m["HORIZONTE"].dt.year))

    m["EXTENSAO_MESES"] = extensao.where(sit == "TITULADO")
    m["DESFECHO"] = np.select(
        [
            titulou & (extensao <= 0),
            titulou & (extensao <= 6),
            titulou & (extensao <= 12),
            titulou,
            sit.isin(["DESLIGADO", "ABANDONOU"]) & no_horizonte,
            sit == "MUDANÇA DE NÍVEL SEM DEFESA",
            sumiu,
        ],
        [
            "1 no prazo",
            "2 +1 a +6 meses",
            "3 +7 a +12 meses",
            "4 +13 a +36 meses",
            "6 desligou ou abandonou",
            "7 mudança de nível",
            "8 sem registro final",
        ],
        default="5 sem titular no horizonte",
    )
    m["TITULOU_ATE_CORTE"] = titulou & (extensao <= config.CORTE_INDICADOR)
    return m


def main():
    base = pd.read_parquet(RAIZ / "data" / "processed" / "base.parquet")
    etapas = [("Registros no recorte (área 2, acadêmico)", len(base))]

    m = identificar_matriculas(base)
    etapas.append(("Matrículas (primeira matrícula de cada discente, programa e nível)", len(m)))
    m = atribuir_grupos(m)
    etapas.append(("Matrículas nas janelas de ingresso dos grupos", len(m)))
    na_janela = m.groupby(["NIVEL", "GRUPO"]).size().rename("na_janela")

    # Etapa 5: mantém só quem estava ativo 12 meses antes do prazo
    saiu_antes = m["NM_SITUACAO_DISCENTE"].isin(EVENTOS) & (m["DT_SITUACAO"] < m["CORTE_ATIVO"])
    m = m[~saiu_antes].copy()
    etapas.append(("Matrículas ativas 12 meses antes do prazo (analisadas)", len(m)))
    m = classificar_desfechos(m)
    m.to_parquet(RAIZ / "data" / "processed" / "matriculas.parquet", index=False)

    etapas = pd.DataFrame(etapas, columns=["etapa", "quantidade"])
    etapas.to_csv(RAIZ / "results" / "tabelas" / "etapas_preparacao.csv", index=False)
    print("\nContagem por etapa")
    print(etapas.to_string(index=False))

    tabela1 = pd.concat(
        [
            na_janela,
            m.groupby(["NIVEL", "GRUPO"]).size().rename("analisadas"),
            m.groupby(["NIVEL", "GRUPO"])["CD_PROGRAMA_IES"].nunique().rename("programas"),
            m[m["SG_ENTIDADE_ENSINO"] == "UNB"].groupby(["NIVEL", "GRUPO"]).size().rename("unb"),
        ],
        axis=1,
    ).fillna(0).astype(int)
    ordem = [(n, g) for n in ["MESTRADO", "DOUTORADO"] for g in config.GRUPOS[n]]
    tabela1 = tabela1.reindex(ordem)
    tabela1.to_csv(RAIZ / "results" / "tabelas" / "tabela1_matriculas.csv")
    print("\nTabela I")
    print(tabela1.to_string())


if __name__ == "__main__":
    main()
