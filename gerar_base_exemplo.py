#!/usr/bin/env python3
"""
Gera base_frete.xlsx com dados sintéticos realistas para teste.
Execute uma vez antes de usar calcular_frete.py.

Estrutura da planilha gerada:
  transportadora | cep_origem_inicio | cep_origem_fim | cep_destino_inicio | cep_destino_fim
  peso_min_kg    | peso_max_kg       | maior_lado_min_cm | maior_lado_max_cm
  cubagem_min_m3 | cubagem_max_m3    | valor_frete       | prazo_dias

Substitua base_frete.xlsx pela sua planilha real mantendo esses nomes de coluna.
"""

import pandas as pd
import numpy as np

np.random.seed(42)

# Transportadoras e seus perfis de precificação
TRANSPORTADORAS = {
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

# Regiões do Brasil por faixa de CEP (primeiros 8 dígitos sem hífen)
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

# Faixas de peso em kg
FAIXAS_PESO = [
    (0.0,   1.0),
    (1.0,   5.0),
    (5.0,  20.0),
    (20.0, 100.0),
]

# Faixas de maior dimensão em cm
FAIXAS_LADO = [
    (0,   40),
    (40,  80),
    (80, 160),
]

# Faixas de cubagem em m³
FAIXAS_CUBAGEM = [
    (0.000, 0.020),
    (0.020, 0.100),
    (0.100, 0.500),
    (0.500, 2.000),
]


def calcular_valor(cfg: dict, dist_factor: float,
                   peso_mid: float, lado_mid: float, cub_mid: float) -> float:
    base   = cfg['fator_base'] * 12.0 * dist_factor
    v_peso = cfg['por_kg'] * peso_mid * dist_factor
    v_cub  = cfg['por_m3'] * cub_mid  * dist_factor
    valor  = base + v_peso + v_cub

    # Peso cubado para serviços expressos (300 kg/m³)
    if cfg['prazo_base'] <= 2:
        peso_cubado = cub_mid * 300.0
        if peso_cubado > peso_mid:
            valor = base + cfg['por_kg'] * peso_cubado * dist_factor

    # Variação realista ±4%
    return valor * np.random.uniform(0.96, 1.04)


def main():
    rows = []

    for transp_nome, cfg in TRANSPORTADORAS.items():
        for i, (reg_orig, cep_orig_ini, cep_orig_fim) in enumerate(REGIOES):
            for j, (reg_dest, cep_dest_ini, cep_dest_fim) in enumerate(REGIOES):

                # Cobertura parcial por transportadora
                if np.random.random() > cfg['cobertura']:
                    continue

                dist_factor = 1.0 + abs(i - j) * 0.12

                for peso_min, peso_max in FAIXAS_PESO:
                    for lado_min, lado_max in FAIXAS_LADO:
                        for cub_min, cub_max in FAIXAS_CUBAGEM:
                            peso_mid = (peso_min + peso_max) / 2
                            lado_mid = (lado_min + lado_max) / 2
                            cub_mid  = (cub_min  + cub_max)  / 2

                            # Filtro de consistência física
                            if cub_mid > 0 and peso_mid / cub_mid > 2000:
                                continue  # densidade impossível
                            if cub_mid > 0.5 and peso_mid < 1.0:
                                continue  # caixa grande mas produto muito leve

                            valor = round(
                                calcular_valor(cfg, dist_factor, peso_mid, lado_mid, cub_mid),
                                2
                            )
                            valor = max(valor, 5.00)  # frete mínimo R$ 5,00
                            prazo = max(1, cfg['prazo_base'] + int(abs(i - j) * 0.6))

                            rows.append({
                                'transportadora':     transp_nome,
                                'cep_origem_inicio':  cep_orig_ini,
                                'cep_origem_fim':     cep_orig_fim,
                                'cep_destino_inicio': cep_dest_ini,
                                'cep_destino_fim':    cep_dest_fim,
                                'peso_min_kg':        peso_min,
                                'peso_max_kg':        peso_max,
                                'maior_lado_min_cm':  lado_min,
                                'maior_lado_max_cm':  lado_max,
                                'cubagem_min_m3':     cub_min,
                                'cubagem_max_m3':     cub_max,
                                'valor_frete':        valor,
                                'prazo_dias':         prazo,
                            })

    df = pd.DataFrame(rows)
    df.to_excel('base_frete.xlsx', index=False)

    print(f"base_frete.xlsx gerada com sucesso.")
    print(f"  Regras      : {len(df):,}")
    print(f"  Transportad.: {df['transportadora'].nunique()}")
    print(f"  Valor mín   : R$ {df['valor_frete'].min():.2f}")
    print(f"  Valor máx   : R$ {df['valor_frete'].max():.2f}")
    print()
    print("Substitua base_frete.xlsx pela sua planilha real")
    print("mantendo os mesmos nomes de coluna.")


if __name__ == '__main__':
    main()
