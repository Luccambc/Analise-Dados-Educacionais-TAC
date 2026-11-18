# Tempo de titulação na pós-graduação em Computação (2017 a 2024)

## Dados
Baixar no Portal de Dados Abertos da CAPES os arquivos de discentes da
pós-graduação stricto sensu, anos-base 2017 a 2024, e colocar em `data/raw/`.

Versões usadas:
- 2017 a 2020: publicação de 2023-12-01
- 2021 a 2023: publicação de 2025-03-31
- 2024: publicação de 2025-12-01

## Versões testadas
Python 3.13.16, pandas 3.0.5, NumPy 2.5.3, SciPy 1.18.1, matplotlib 3.11.2, pyarrow 26.0.0.

## Como rodar
```
pip install -r requirements.txt
python src/01_carregar.py
python src/02_verificar.py
python src/03_preparar.py
python src/04_qp1.py
python src/05_qp2.py
python src/06_qp3.py
```

Ou tudo de uma vez: `python src/rodar_tudo.py`

## O que cada script gera
| Script | Saídas |
| --- | --- |
| 01_carregar | `data/processed/base.parquet` |
| 02_verificar | `results/verificacao.txt` |
| 03_preparar | `data/processed/matriculas.parquet`, `results/tabelas/etapas_preparacao.csv`, `results/tabelas/tabela1_matriculas.csv` |
| 04_qp1 | tabelas `qp1_*.csv` (inclui `qp1_meses_ate_metade.csv`), figuras `qp1_faixas.png`, `qp1_regioes.png`, `qp1_curva_acumulada.png` e `qp1_forest_instituicoes.png` |
| 05_qp2 | tabelas `qp2_*.csv` |
| 06_qp3 | tabelas `qp3_*.csv` |

Todas as decisões metodológicas (grupos, prazos, prorrogação, horizonte, mínimos, semente) estão em `config.py`.

## Indicador e faixas de desfecho
O indicador principal é a % de matrículas do grupo tituladas até o prazo mais a
prorrogação permitida pelo regulamento da UnB (Resolução CEPE 0080/2017, art. 24):
24 + 6 meses no mestrado e 48 + 12 meses no doutorado. Cada matrícula é classificada em:
até o prazo, na prorrogação, depois da prorrogação, ainda matriculado ao fim do
acompanhamento, desligou ou abandonou, mudança de nível e sem registro final.

