#!/usr/bin/env python3
"""
Calculadora de Frete com Machine Learning
==========================================
1. Busca exata na tabela de regras (planilha Excel)
2. Se não encontrar correspondência, usa Gradient Boosting para estimar

Uso:
  python calcular_frete.py --interativo
  python calcular_frete.py \\
      --cep-origem 01310100 --cep-destino 30130110 \\
      --peso 3.5 --maior-lado 45 --cubagem 0.025
"""

import argparse
import os
import sys
import time
import warnings
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Dias úteis — feriados nacionais brasileiros
# ---------------------------------------------------------------------------

_MESES = ['jan','fev','mar','abr','mai','jun','jul','ago','set','out','nov','dez']

_FERIADOS_CACHE: dict[int, set[date]] = {}


def _calcular_feriados(ano: int) -> set[date]:
    """Feriados nacionais fixos + móveis (Páscoa, Carnaval, Corpus Christi)."""
    # Algoritmo de Butcher para calcular o Domingo de Páscoa
    a = ano % 19
    b, c = divmod(ano, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes_p = (h + l - 7 * m + 114) // 31
    dia_p = (h + l - 7 * m + 114) % 31 + 1
    pascoa = date(ano, mes_p, dia_p)

    return {
        date(ano,  1,  1),                      # Confraternização Universal
        pascoa - timedelta(days=47),             # Carnaval — segunda
        pascoa - timedelta(days=46),             # Carnaval — terça
        pascoa - timedelta(days=2),              # Sexta-feira Santa
        date(ano,  4, 21),                      # Tiradentes
        date(ano,  5,  1),                      # Dia do Trabalho
        pascoa + timedelta(days=60),             # Corpus Christi
        date(ano,  9,  7),                      # Independência
        date(ano, 10, 12),                      # Nossa Senhora Aparecida
        date(ano, 11,  2),                      # Finados
        date(ano, 11, 15),                      # Proclamação da República
        date(ano, 12, 25),                      # Natal
    }


def _feriados_do_ano(ano: int) -> set[date]:
    if ano not in _FERIADOS_CACHE:
        _FERIADOS_CACHE[ano] = _calcular_feriados(ano)
    return _FERIADOS_CACHE[ano]


def is_dia_util(d: date) -> bool:
    return d.weekday() < 5 and d not in _feriados_do_ano(d.year)


def data_entrega(prazo_uteis: int, inicio: date | None = None) -> date:
    """Calcula a data de entrega contando `prazo_uteis` dias úteis a partir de `inicio`."""
    d = inicio or date.today()
    contados = 0
    while contados < prazo_uteis:
        d += timedelta(days=1)
        if is_dia_util(d):
            contados += 1
    return d


def formatar_data(d: date) -> str:
    return f"{d.day:02d}/{_MESES[d.month - 1]}"

COLUNAS_OBRIGATORIAS = [
    'transportadora',
    'cep_origem_inicio', 'cep_origem_fim',
    'cep_destino_inicio', 'cep_destino_fim',
    'peso_min_kg', 'peso_max_kg',
    'maior_lado_min_cm', 'maior_lado_max_cm',
    'cubagem_min_m3', 'cubagem_max_m3',
    'valor_frete',
]


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def normalizar_cep(valor) -> int:
    """Converte qualquer formato de CEP para inteiro de 8 dígitos."""
    return int(str(valor).replace('-', '').replace('.', '').strip().zfill(8))


def carregar_base(caminho: str) -> pd.DataFrame:
    """Carrega e valida a planilha Excel."""
    try:
        df = pd.read_excel(caminho, engine='openpyxl')
    except FileNotFoundError:
        sys.exit(f"\nErro: planilha não encontrada — {caminho}")
    except Exception as e:
        sys.exit(f"\nErro ao ler planilha: {e}")

    faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltando:
        sys.exit(
            f"\nColunas obrigatórias ausentes: {faltando}"
            f"\nColunas encontradas: {list(df.columns)}"
        )

    for col in ['cep_origem_inicio', 'cep_origem_fim',
                'cep_destino_inicio', 'cep_destino_fim']:
        df[col] = df[col].apply(lambda x: normalizar_cep(int(float(x))))

    return df.dropna(subset=COLUNAS_OBRIGATORIAS).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Busca exata
# ---------------------------------------------------------------------------

def buscar_exato(df: pd.DataFrame, cep_orig: int, cep_dest: int,
                 peso: float, lado: float, cubagem: float) -> pd.DataFrame:
    """Filtra regras que cobrem exatamente os parâmetros informados."""
    mask = (
        (df['cep_origem_inicio']  <= cep_orig)  & (df['cep_origem_fim']   >= cep_orig)  &
        (df['cep_destino_inicio'] <= cep_dest)  & (df['cep_destino_fim']  >= cep_dest)  &
        (df['peso_min_kg']        <= peso)      & (df['peso_max_kg']       >  peso)     &
        (df['maior_lado_min_cm']  <= lado)      & (df['maior_lado_max_cm'] >  lado)     &
        (df['cubagem_min_m3']     <= cubagem)   & (df['cubagem_max_m3']    >  cubagem)
    )
    return df[mask].copy()


def montar_resultados_exatos(exatos: pd.DataFrame) -> list[dict]:
    """Para cada transportadora, retorna a linha com menor valor."""
    tem_prazo = 'prazo_dias' in exatos.columns
    resultados = []
    for transp in exatos['transportadora'].unique():
        subset = exatos[exatos['transportadora'] == transp]
        row = subset.loc[subset['valor_frete'].idxmin()]
        prazo = None
        if tem_prazo and pd.notna(row.get('prazo_dias')):
            prazo = int(row['prazo_dias'])
        resultados.append({
            'transportadora': transp,
            'valor_frete': round(float(row['valor_frete']), 2),
            'prazo_dias': prazo,
            'fonte': 'Tabela (correspondência exata)',
        })
    return resultados


# ---------------------------------------------------------------------------
# Modelo ML
# ---------------------------------------------------------------------------

class ModeloFrete:
    """
    Gradient Boosting treinado nas regras da tabela.
    Usado como fallback quando não há correspondência exata.
    """

    def __init__(self):
        self.modelo = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )
        self.modelo_prazo = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )
        self.le = LabelEncoder()
        self.transportadoras_: list[str] = []
        self.tem_prazo_: bool = False

    def _extrair_features(self, df: pd.DataFrame) -> np.ndarray:
        cep_orig_mid = (df['cep_origem_inicio']  + df['cep_origem_fim'])  / 2
        cep_dest_mid = (df['cep_destino_inicio'] + df['cep_destino_fim']) / 2
        return np.column_stack([
            cep_orig_mid / 1e7,
            cep_dest_mid / 1e7,
            np.abs(cep_orig_mid - cep_dest_mid) / 1e7,        # proxy de distância
            (df['peso_min_kg']       + df['peso_max_kg'])       / 2,
            (df['maior_lado_min_cm'] + df['maior_lado_max_cm']) / 2,
            (df['cubagem_min_m3']    + df['cubagem_max_m3'])    / 2,
            self.le.transform(df['transportadora']),
        ])

    def treinar(self, df: pd.DataFrame) -> tuple[float, float]:
        """Treina os modelos de preço e prazo. Retorna (MAE_preco, MAE_prazo)."""
        if df.empty:
            return 0.0, 0.0

        self.le.fit(df['transportadora'])
        self.transportadoras_ = df['transportadora'].unique().tolist()
        self.tem_prazo_ = 'prazo_dias' in df.columns and df['prazo_dias'].notna().any()

        X = self._extrair_features(df)
        mae_preco = 0.0
        mae_prazo = 0.0

        if len(X) >= 30:
            X_tr, X_te, idx_tr, idx_te = train_test_split(
                X, df.index, test_size=0.15, random_state=42
            )
            y_preco_tr = df.loc[idx_tr, 'valor_frete'].values
            y_preco_te = df.loc[idx_te, 'valor_frete'].values

            self.modelo.fit(X_tr, y_preco_tr)
            mae_preco = float(mean_absolute_error(y_preco_te, self.modelo.predict(X_te)))

            if self.tem_prazo_:
                df_prazo = df.dropna(subset=['prazo_dias'])
                X_p = self._extrair_features(df_prazo)
                y_p = df_prazo['prazo_dias'].values
                X_ptr, X_pte, y_ptr, y_pte = train_test_split(
                    X_p, y_p, test_size=0.15, random_state=42
                )
                self.modelo_prazo.fit(X_ptr, y_ptr)
                mae_prazo = float(mean_absolute_error(y_pte, self.modelo_prazo.predict(X_pte)))
        else:
            self.modelo.fit(X, df['valor_frete'].values)
            if self.tem_prazo_:
                df_prazo = df.dropna(subset=['prazo_dias'])
                self.modelo_prazo.fit(
                    self._extrair_features(df_prazo),
                    df_prazo['prazo_dias'].values,
                )

        return mae_preco, mae_prazo

    def prever_todos(self, cep_orig: int, cep_dest: int,
                     peso: float, lado: float, cubagem: float) -> list[dict]:
        """Estima o valor de frete para cada transportadora conhecida."""
        resultados = []
        for transp in self.transportadoras_:
            row = {
                'transportadora':     transp,
                'cep_origem_inicio':  cep_orig, 'cep_origem_fim':     cep_orig,
                'cep_destino_inicio': cep_dest, 'cep_destino_fim':    cep_dest,
                'peso_min_kg':        peso,     'peso_max_kg':        peso,
                'maior_lado_min_cm':  lado,     'maior_lado_max_cm':  lado,
                'cubagem_min_m3':     cubagem,  'cubagem_max_m3':     cubagem,
            }
            df_row = pd.DataFrame([row])
            X_row = self._extrair_features(df_row)
            preco = max(0.0, float(self.modelo.predict(X_row)[0]))
            prazo = None
            if self.tem_prazo_:
                prazo = max(1, round(float(self.modelo_prazo.predict(X_row)[0])))
            resultados.append({
                'transportadora': transp,
                'valor_frete': round(preco, 2),
                'prazo_dias': prazo,
                'fonte': 'ML (estimativa)',
            })
        return resultados


# ---------------------------------------------------------------------------
# Score combinado preço × prazo
# ---------------------------------------------------------------------------

def calcular_score(resultados: list[dict], w_preco: int) -> list[dict]:
    """
    Ordena as opções por score combinado.
    w_preco (0-100): percentual de peso dado ao preço.
    O restante (100 - w_preco) é dado ao prazo.
    Normalização min-max em [0,1] — 0 = melhor, 1 = pior.
    """
    w_p = w_preco / 100
    w_t = 1.0 - w_p

    precos = [r['valor_frete'] for r in resultados]
    prazos = [r['prazo_dias'] for r in resultados if r.get('prazo_dias') is not None]

    preco_min, preco_max = min(precos), max(precos)
    prazo_min = min(prazos) if prazos else 0
    prazo_max = max(prazos) if prazos else 0

    for r in resultados:
        if preco_max != preco_min:
            preco_norm = (r['valor_frete'] - preco_min) / (preco_max - preco_min)
        else:
            preco_norm = 0.0

        prazo_val = r.get('prazo_dias')
        if prazo_val is not None and prazo_max != prazo_min:
            prazo_norm = (prazo_val - prazo_min) / (prazo_max - prazo_min)
        else:
            prazo_norm = 0.5  # sem dado de prazo → posição neutra

        r['score'] = round(w_p * preco_norm + w_t * prazo_norm, 4)

    return sorted(resultados, key=lambda r: r['score'])


# ---------------------------------------------------------------------------
# Exibição
# ---------------------------------------------------------------------------

DIV = '─' * 68

def exibir_resultados(resultados: list[dict], w_preco: int = 50,
                      elapsed_ms: float | None = None) -> None:
    print()
    print(DIV)
    print('  RESULTADO — OPÇÕES DE FRETE DISPONÍVEIS')
    print(DIV)

    if not resultados:
        print('  Nenhuma opção encontrada para os parâmetros informados.')
        print(DIV)
        return

    w_t = 100 - w_preco
    print(f'  Critério: {w_preco}% preço  /  {w_t}% prazo')
    print(DIV)

    ordenados = calcular_score(resultados, w_preco)
    ranks = [f'{i+1}°' for i in range(len(ordenados))]

    for rank, r in zip(ranks, ordenados):
        if r.get('prazo_dias'):
            entrega = data_entrega(int(r['prazo_dias']))
            prazo_str = f"  {int(r['prazo_dias'])} d.u. → {formatar_data(entrega)}"
        else:
            prazo_str = ''
        fonte_str = f"  [{r['fonte']}]"
        score_str = f"  score: {r['score']:.2f}"
        print(
            f"  {rank:3}  {r['transportadora']:<22}"
            f"  R$ {r['valor_frete']:>9.2f}"
            f"{prazo_str}"
            f"{score_str}"
            f"{fonte_str}"
        )

    print(DIV)
    melhor = ordenados[0]
    if melhor.get('prazo_dias'):
        entrega_m = data_entrega(int(melhor['prazo_dias']))
        prazo_m = f"  —  {int(melhor['prazo_dias'])} d.u. (entrega: {formatar_data(entrega_m)})"
    else:
        prazo_m = ''
    print(f"  MELHOR OPÇÃO: {melhor['transportadora']}  →  R$ {melhor['valor_frete']:.2f}{prazo_m}")
    if elapsed_ms is not None:
        print(f"  Tempo de cotação: {elapsed_ms:.0f} ms")
    print(f"{DIV}\n")


# ---------------------------------------------------------------------------
# Entrada de dados
# ---------------------------------------------------------------------------

def ler_float(prompt: str) -> float:
    return float(input(prompt).strip().replace(',', '.'))


def calcular_dimensoes(altura: float, largura: float, comprimento: float) -> tuple[float, float]:
    """Retorna (maior_lado_cm, cubagem_m3) a partir das três dimensões em cm."""
    maior_lado = max(altura, largura, comprimento)
    cubagem    = (altura * largura * comprimento) / 1_000_000  # cm³ → m³
    return maior_lado, cubagem


def coletar_interativo() -> tuple[str, str, float, float, float, float, float, int]:
    print('\n  Informe os dados do envio:\n')
    cep_orig_str = input('  CEP de origem      (ex: 01310-100): ').strip()
    cep_dest_str = input('  CEP de destino     (ex: 30130-110): ').strip()
    peso         = ler_float('  Peso (kg)                         : ')
    altura       = ler_float('  Altura (cm)                       : ')
    largura      = ler_float('  Largura (cm)                      : ')
    comprimento  = ler_float('  Comprimento (cm)                  : ')
    w_str        = input('  Prioridade preço 0-100 (padrão 50) : ').strip()
    w_preco      = max(0, min(100, int(w_str))) if w_str.isdigit() else 50
    return cep_orig_str, cep_dest_str, peso, altura, largura, comprimento, w_preco


# ---------------------------------------------------------------------------
# Cache do DataFrame em disco
# ---------------------------------------------------------------------------

def _caminho_cache_df(planilha: str) -> str:
    return os.path.splitext(os.path.abspath(planilha))[0] + '_df.pkl'


def carregar_base_com_cache(caminho: str) -> pd.DataFrame:
    """
    Carrega o DataFrame do cache binário se estiver atualizado.
    Na primeira execução (ou quando a planilha mudar), lê o Excel,
    valida e salva o cache para as próximas chamadas.
    """
    cache = _caminho_cache_df(caminho)
    caminho_abs = os.path.abspath(caminho)

    if os.path.exists(cache):
        if os.path.getmtime(cache) >= os.path.getmtime(caminho_abs):
            t0 = time.perf_counter()
            df = joblib.load(cache)
            ms = (time.perf_counter() - t0) * 1000
            print(f'  {len(df):,} regras  |  {df["transportadora"].nunique()} '
                  f'transportadoras  (cache: {ms:.0f} ms)')
            return df

    df = carregar_base(caminho)
    joblib.dump(df, cache)
    print(f'  {len(df):,} regras  |  {df["transportadora"].nunique()} '
          f'transportadoras  (Excel parseado — cache salvo)')
    return df


# ---------------------------------------------------------------------------
# Cache do modelo ML em disco
# ---------------------------------------------------------------------------

def _caminho_cache(planilha: str) -> str:
    """Arquivo .pkl gerado ao lado da planilha."""
    return os.path.splitext(os.path.abspath(planilha))[0] + '_modelo.pkl'


def carregar_ou_treinar_modelo(
    df: pd.DataFrame, planilha: str
) -> tuple['ModeloFrete', float, float]:
    """
    Carrega o modelo serializado se estiver mais recente que a planilha.
    Caso contrário, treina, exibe MAE e salva em disco.
    Retorna (modelo, mae_preco, mae_prazo).
    """
    cache = _caminho_cache(planilha)
    planilha_abs = os.path.abspath(planilha)

    if os.path.exists(cache):
        if os.path.getmtime(cache) >= os.path.getmtime(planilha_abs):
            print('  Carregando modelo do cache ...')
            t0 = _now_ms()
            dados = joblib.load(cache)
            print(f'  Modelo carregado em {_now_ms() - t0:.0f} ms  '
                  f'(MAE preço: R$ {dados["mae_preco"]:.2f}  |  '
                  f'MAE prazo: {dados["mae_prazo"]:.1f} d.u.)')
            return dados['modelo'], dados['mae_preco'], dados['mae_prazo']

    print(f'  Treinando com {len(df):,} exemplos ...')
    t0 = _now_ms()
    modelo = ModeloFrete()
    mae_preco, mae_prazo = modelo.treinar(df)
    elapsed = _now_ms() - t0
    info_p = f'R$ {mae_preco:.2f}' if mae_preco > 0 else 'N/A'
    info_t = f'{mae_prazo:.1f} d.u.' if mae_prazo > 0 else 'N/A'
    print(f'  Modelo treinado em {elapsed:.0f} ms  |  MAE preço: {info_p}  |  MAE prazo: {info_t}')

    joblib.dump({'modelo': modelo, 'mae_preco': mae_preco, 'mae_prazo': mae_prazo}, cache)
    print(f'  Cache salvo: {os.path.basename(cache)}')

    return modelo, mae_preco, mae_prazo


def _now_ms() -> float:
    import time
    return time.perf_counter() * 1000


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Calculadora de Frete com Machine Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Modo interativo (recomendado para uso manual)
  python calcular_frete.py --interativo

  # Modo direto (ideal para integração / scripts)
  python calcular_frete.py \\
      --cep-origem 01310100 --cep-destino 30130110 \\
      --peso 3.5 --altura 20 --largura 30 --comprimento 40

  # Usando planilha personalizada
  python calcular_frete.py --planilha minha_tabela.xlsx --interativo
        """,
    )
    parser.add_argument('--planilha',    default='base_frete.xlsx',
                        help='Planilha Excel com as regras de frete')
    parser.add_argument('--cep-origem',  dest='cep_origem')
    parser.add_argument('--cep-destino', dest='cep_destino')
    parser.add_argument('--peso',        type=float)
    parser.add_argument('--altura',      type=float, help='Altura do objeto em cm')
    parser.add_argument('--largura',     type=float, help='Largura do objeto em cm')
    parser.add_argument('--comprimento', type=float, help='Comprimento do objeto em cm')
    parser.add_argument('--interativo',  action='store_true')
    parser.add_argument(
        '--prioridade-preco',
        dest='prioridade_preco',
        type=int,
        default=50,
        metavar='0-100',
        help='Peso do preço na decisão (0=só prazo, 100=só preço, padrão=50)',
    )

    args = parser.parse_args()

    print()
    print('═' * 68)
    print('  SISTEMA DE CÁLCULO DE FRETE — ML Edition')
    print('═' * 68)

    tem_args = all([args.cep_origem, args.cep_destino,
                    args.peso, args.altura, args.largura, args.comprimento])

    if args.interativo or not tem_args:
        cep_orig_str, cep_dest_str, peso, altura, largura, comprimento, w_preco = coletar_interativo()
    else:
        cep_orig_str = args.cep_origem
        cep_dest_str = args.cep_destino
        peso         = args.peso
        altura       = args.altura
        largura      = args.largura
        comprimento  = args.comprimento
        w_preco      = max(0, min(100, args.prioridade_preco))

    lado, cub = calcular_dimensoes(altura, largura, comprimento)

    cep_orig = normalizar_cep(cep_orig_str)
    cep_dest = normalizar_cep(cep_dest_str)

    print(f'\n  Parâmetros de busca:')
    print(f'    CEP Origem   :  {cep_orig_str}  ({cep_orig})')
    print(f'    CEP Destino  :  {cep_dest_str}  ({cep_dest})')
    print(f'    Peso         :  {peso:.3f} kg')
    print(f'    Dimensões    :  {altura:.0f} × {largura:.0f} × {comprimento:.0f} cm  (A × L × C)')
    print(f'    Maior lado   :  {lado:.1f} cm  (calculado)')
    print(f'    Cubagem      :  {cub:.6f} m³  (calculado)')

    print(f'\n  Carregando base: {args.planilha} ...')
    df = carregar_base_com_cache(args.planilha)

    t_inicio = time.perf_counter()
    print(f'\n  Buscando correspondências exatas ...')
    exatos = buscar_exato(df, cep_orig, cep_dest, peso, lado, cub)

    if not exatos.empty:
        print(f'  {len(exatos)} linha(s) encontrada(s). ✓')
        resultados = montar_resultados_exatos(exatos)
    else:
        if df.empty:
            print('  Base de dados vazia — nenhuma cotação disponível.')
            resultados = []
        else:
            print('  Sem correspondência exata — ativando modelo ML.')
            modelo, _, _ = carregar_ou_treinar_modelo(df, args.planilha)
            resultados = modelo.prever_todos(cep_orig, cep_dest, peso, lado, cub)

    elapsed_ms = (time.perf_counter() - t_inicio) * 1000
    exibir_resultados(resultados, w_preco, elapsed_ms)


if __name__ == '__main__':
    main()
