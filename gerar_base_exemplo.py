#!/usr/bin/env python3
"""
Gera planilhas de exemplo em planilhas/ com o novo formato de CEP.
Execute uma vez antes de usar calcular_frete.py.

Colunas geradas:
  transportadora | cep_origem | cep_destino | cep_excluido
  peso_min_kg    | peso_max_kg | maior_lado_min_cm | maior_lado_max_cm
  cubagem_min_m3 | cubagem_max_m3 | valor_frete | prazo_dias

Formatos aceitos em cep_origem / cep_destino / cep_excluido:
  Range único  : "01000000..09999999"
  CEP único    : "84035565"
  Lista mista  : "01000000..09999999, 84035565"
"""

import os
import time

import numpy as np
import pandas as pd

np.random.seed(42)


TRANSPORTADORAS_JAN = {
    'Correios PAC': {
        'fator_base': 1.00, 'por_kg': 1.20, 'por_m3': 800,
        'prazo_base': 6, 'cobertura': 1.0,
    },
    'Correios SEDEX': {
        'fator_base': 1.80, 'por_kg': 2.50, 'por_m3': 1500,
        'prazo_base': 1, 'cobertura': 1.0,
    },
    'Jadlog .Package': {
        'fator_base': 0.85, 'por_kg': 0.90, 'por_m3': 650,
        'prazo_base': 4, 'cobertura': 0.90,
    },
    'Total Express': {
        'fator_base': 1.10, 'por_kg': 1.00, 'por_m3': 700,
        'prazo_base': 3, 'cobertura': 0.80,
    },
    'Braspress': {
        'fator_base': 0.60, 'por_kg': 0.65, 'por_m3': 420,
        'prazo_base': 5, 'cobertura': 0.85,
    },
}

TRANSPORTADORAS_FEV = {
    'Sequóia': {
        'fator_base': 1.30, 'por_kg': 1.10, 'por_m3': 780,
        'prazo_base': 2, 'cobertura': 0.80,
    },
    'Azul Cargo': {
        'fator_base': 0.95, 'por_kg': 0.85, 'por_m3': 600,
        'prazo_base': 3, 'cobertura': 0.85,
    },
    'TNT Mercúrio': {
        'fator_base': 1.50, 'por_kg': 1.80, 'por_m3': 1200,
        'prazo_base': 2, 'cobertura': 0.70,
    },
}

REGIOES = [
    ('SP_Capital',   1_000_000,   9_999_999),
    ('SP_Interior', 13_000_000,  19_999_999),
    ('RJ',          20_000_000,  28_999_999),
    ('MG',          30_000_000,  39_999_999),
    ('BA',          40_000_000,  48_999_999),
    ('PR',          80_000_000,  87_999_999),
    ('SC',          88_000_000,  89_999_999),
    ('RS',          90_000_000,  99_999_999),
]

FAIXAS_PESO    = [(0.0, 1.0), (1.0, 5.0), (5.0, 20.0), (20.0, 100.0)]
FAIXAS_LADO    = [(0, 40), (40, 80), (80, 160)]
FAIXAS_CUBAGEM = [(0.0, 0.02), (0.02, 0.10), (0.10, 0.50), (0.50, 2.00)]

# Exclusões específicas por (transportadora, região_destino).
# Apenas alguns carriers têm restrição, e em ranges pequenos e cirúrgicos.
# A maioria das linhas geradas não terá nenhuma exclusão.
EXCLUIDOS = {
    ('Correios PAC',    'PR'):        '80010000..80019999',   # bairro restrito em Curitiba
    ('Braspress',       'SP_Capital'): '01310050..01310099',  # trecho da Av. Paulista
    ('Total Express',   'RS'):        '90010000..90019999',   # área central de Porto Alegre
    ('Jadlog .Package', 'MG'):        '30110000..30119999',   # centro de BH
}


def calcular_valor(cfg, dist_factor, peso_mid, lado_mid, cub_mid):
    base  = cfg['fator_base'] * 12.0 * dist_factor
    valor = base + cfg['por_kg'] * peso_mid * dist_factor + cfg['por_m3'] * cub_mid * dist_factor
    if cfg['prazo_base'] <= 2:
        pc = cub_mid * 300.0
        if pc > peso_mid:
            valor = base + cfg['por_kg'] * pc * dist_factor
    return valor * np.random.uniform(0.96, 1.04)


def gerar_planilha(transportadoras: dict, sufixo: str) -> pd.DataFrame:
    rows = []
    for transp_nome, cfg in transportadoras.items():
        for i, (reg_orig, cep_oi, cep_of) in enumerate(REGIOES):
            for j, (reg_dest, cep_di, cep_df) in enumerate(REGIOES):
                if np.random.random() > cfg['cobertura']:
                    continue

                dist_factor = 1.0 + abs(i - j) * 0.12

                # Formato novo: range como string
                cep_origem_str   = f"{cep_oi}..{cep_of}"
                cep_destino_str  = f"{cep_di}..{cep_df}"
                cep_excluido_str = EXCLUIDOS.get((transp_nome, reg_dest), '')

                for pm, pM in FAIXAS_PESO:
                    for lm, lM in FAIXAS_LADO:
                        for cm, cM in FAIXAS_CUBAGEM:
                            pm_mid = (pm + pM) / 2
                            cm_mid = (cm + cM) / 2
                            if cm_mid > 0 and pm_mid / cm_mid > 2000:
                                continue
                            if cm_mid > 0.5 and pm_mid < 1.0:
                                continue

                            valor = round(
                                max(5.0, calcular_valor(cfg, dist_factor, pm_mid,
                                                        (lm + lM) / 2, cm_mid)), 2
                            )
                            prazo = max(1, cfg['prazo_base'] + int(abs(i - j) * 0.6))

                            rows.append({
                                'transportadora':    transp_nome,
                                'cep_origem':        cep_origem_str,
                                'cep_destino':       cep_destino_str,
                                'cep_excluido':      cep_excluido_str,
                                'peso_min_kg':       pm,
                                'peso_max_kg':       pM,
                                'maior_lado_min_cm': lm,
                                'maior_lado_max_cm': lM,
                                'cubagem_min_m3':    cm,
                                'cubagem_max_m3':    cM,
                                'valor_frete':       valor,
                                'prazo_dias':        prazo,
                            })
    return pd.DataFrame(rows)


def main():
    os.makedirs('planilhas', exist_ok=True)

    # Remove planilhas antigas
    for f in ['planilhas/base_frete.xlsx',
              'planilhas/frete_jan_2026.xlsx',
              'planilhas/frete_fev_2026.xlsx',
              'planilhas/frete_mar_2026.xlsx']:
        if os.path.exists(f):
            os.remove(f)
            print(f"  removido: {f}")

    # Janeiro — 5 transportadoras originais
    df_jan = gerar_planilha(TRANSPORTADORAS_JAN, 'jan')
    df_jan.to_excel('planilhas/frete_jan_2026.xlsx', index=False)
    print(f"frete_jan_2026.xlsx: {len(df_jan):,} linhas | "
          f"{df_jan['transportadora'].nunique()} transportadoras")

    time.sleep(0.1)  # garante mtime diferente

    # Fevereiro — 3 novas transportadoras
    df_fev = gerar_planilha(TRANSPORTADORAS_FEV, 'fev')
    df_fev.to_excel('planilhas/frete_fev_2026.xlsx', index=False)
    print(f"frete_fev_2026.xlsx: {len(df_fev):,} linhas | "
          f"{df_fev['transportadora'].nunique()} transportadoras")

    print()
    print("Exemplo de linha gerada:")
    print(df_jan[['transportadora', 'cep_origem', 'cep_destino', 'cep_excluido',
                  'peso_min_kg', 'valor_frete']].head(3).to_string(index=False))
    print()
    print("Coloque suas planilhas reais em planilhas/ com as mesmas colunas.")


if __name__ == '__main__':
    main()
