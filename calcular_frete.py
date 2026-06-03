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
import glob
import json
import os
import sys
import time
import warnings
from datetime import date, timedelta
from itertools import product as iterproduct

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

# Colunas obrigatórias que não dependem do formato de CEP
COLUNAS_BASE = [
    'transportadora',
    'peso_min_kg', 'peso_max_kg',
    'maior_lado_min_cm', 'maior_lado_max_cm',
    'cubagem_min_m3', 'cubagem_max_m3',
    'valor_frete',
]
# Mantido para compatibilidade com código legado que referencia COLUNAS_OBRIGATORIAS
COLUNAS_OBRIGATORIAS = COLUNAS_BASE


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def normalizar_cep(valor) -> int:
    """Converte qualquer formato de CEP para inteiro de 8 dígitos."""
    return int(str(valor).replace('-', '').replace('.', '').strip().zfill(8))


def parsear_ceps(texto) -> list[tuple[int, int]]:
    """
    Parseia string de CEPs com ranges (..) e listas (,).

      "83808000..83880999"            → [(83808000, 83880999)]
      "84035565"                      → [(84035565, 84035565)]
      "83808000..83880999, 84035565"  → [(83808000, 83880999), (84035565, 84035565)]
    """
    if not texto or str(texto).strip() in ('', 'nan', 'None'):
        return []
    resultado = []
    for item in str(texto).split(','):
        item = item.strip()
        if not item:
            continue
        if '..' in item:
            partes = item.split('..', 1)
            ini = normalizar_cep(partes[0].strip())
            fim = normalizar_cep(partes[1].strip())
            resultado.append((min(ini, fim), max(ini, fim)))
        else:
            cep = normalizar_cep(item)
            resultado.append((cep, cep))
    return resultado


def normalizar_planilha(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta o formato de CEP e normaliza para o formato interno.

    Novo formato  (cep_origem, cep_destino, cep_excluido — texto):
      Expande uma linha em N×M linhas pelo produto cartesiano de origens × destinos.
      Armazena os ranges excluídos em '_excluidos' (list de tuplas) por linha.

    Formato antigo (cep_origem_inicio/fim, cep_destino_inicio/fim — inteiros):
      Usa diretamente; adiciona '_excluidos' vazio para compatibilidade.
    """
    tem_novo   = 'cep_destino' in df.columns
    tem_antigo = 'cep_destino_inicio' in df.columns

    if tem_antigo and not tem_novo:
        for col in ['cep_origem_inicio', 'cep_origem_fim',
                    'cep_destino_inicio', 'cep_destino_fim']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: normalizar_cep(int(float(x))))
        df = df.copy()
        df['_excluidos'] = [[] for _ in range(len(df))]
        return df

    if not tem_novo:
        sys.exit(
            "\nColunas de CEP não encontradas.\n"
            "Use 'cep_origem'/'cep_destino' (novo formato) ou "
            "'cep_origem_inicio'/'cep_destino_inicio' (formato antigo)."
        )

    colunas_fixas = [c for c in df.columns
                     if c not in ('cep_origem', 'cep_destino', 'cep_excluido',
                                  'cep_origem_inicio', 'cep_origem_fim',
                                  'cep_destino_inicio', 'cep_destino_fim', '_excluidos')]
    rows = []
    for _, row in df.iterrows():
        origens   = parsear_ceps(row.get('cep_origem',   ''))
        destinos  = parsear_ceps(row.get('cep_destino',  ''))
        excluidos = parsear_ceps(row.get('cep_excluido', ''))

        if not origens or not destinos:
            continue

        base = {c: row[c] for c in colunas_fixas}
        base['_excluidos'] = excluidos

        for (oi, of_), (di, df_) in iterproduct(origens, destinos):
            rows.append({**base,
                         'cep_origem_inicio': oi,  'cep_origem_fim':  of_,
                         'cep_destino_inicio': di, 'cep_destino_fim': df_})

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=colunas_fixas + ['cep_origem_inicio', 'cep_origem_fim',
                                  'cep_destino_inicio', 'cep_destino_fim', '_excluidos']
    )


def carregar_base(caminho: str) -> pd.DataFrame:
    """Carrega, valida e normaliza a planilha Excel (novo ou antigo formato de CEP)."""
    try:
        df = pd.read_excel(caminho, engine='openpyxl')
    except FileNotFoundError:
        sys.exit(f"\nErro: planilha não encontrada — {caminho}")
    except Exception as e:
        sys.exit(f"\nErro ao ler planilha: {e}")

    faltando = [c for c in COLUNAS_BASE if c not in df.columns]
    if faltando:
        sys.exit(
            f"\nColunas obrigatórias ausentes: {faltando}"
            f"\nColunas encontradas: {list(df.columns)}"
        )

    df = normalizar_planilha(df)
    return df.dropna(subset=COLUNAS_BASE).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Busca exata
# ---------------------------------------------------------------------------

def buscar_exato(
    df: pd.DataFrame, cep_orig: int, cep_dest: int,
    peso: float, lado: float, cubagem: float
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Retorna (válidos, excluídos).
    válidos    — DataFrame com as regras que cobrem o CEP e não o excluem.
    excluídos  — lista de {transportadora, motivo} para as regras bloqueadas por cep_excluido.
    """
    mask = (
        (df['cep_origem_inicio']  <= cep_orig)  & (df['cep_origem_fim']   >= cep_orig)  &
        (df['cep_destino_inicio'] <= cep_dest)  & (df['cep_destino_fim']  >= cep_dest)  &
        (df['peso_min_kg']        <= peso)      & (df['peso_max_kg']       >  peso)     &
        (df['maior_lado_min_cm']  <= lado)      & (df['maior_lado_max_cm'] >  lado)     &
        (df['cubagem_min_m3']     <= cubagem)   & (df['cubagem_max_m3']    >  cubagem)
    )
    candidatos = df[mask].copy()

    if candidatos.empty or '_excluidos' not in candidatos.columns:
        return candidatos, []

    validos_idx, excluidos_vistos = [], {}
    for idx, row in candidatos.iterrows():
        motivo_encontrado = None
        for ini, fim in row['_excluidos']:
            if ini <= cep_dest <= fim:
                motivo_encontrado = (f"CEP excluído: {ini}"
                                     if ini == fim
                                     else f"range excluído: {ini}..{fim}")
                break

        if motivo_encontrado:
            transp = row['transportadora']
            if transp not in excluidos_vistos:
                excluidos_vistos[transp] = motivo_encontrado
        else:
            validos_idx.append(idx)

    # Não exibe como excluída uma transportadora que já tem regra válida
    transp_validas = {candidatos.loc[i, 'transportadora'] for i in validos_idx}
    excluidos = [{'transportadora': t, 'motivo': m}
                 for t, m in excluidos_vistos.items()
                 if t not in transp_validas]
    return candidatos.loc[validos_idx], excluidos


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

    def treinar_incremental(self, df_novo: pd.DataFrame) -> bool:
        """
        Atualiza o modelo com dados novos sem reler os dados antigos (warm_start).
        Adiciona árvores proporcionais ao tamanho dos novos dados.
        Retorna False se detectar novas transportadoras — sinal para full retrain.
        """
        if df_novo.empty:
            return True

        transp_novas = set(df_novo['transportadora'].unique()) - set(self.transportadoras_)
        if transp_novas:
            return False  # LabelEncoder não conhece a nova transportadora

        n_add = max(30, min(100, len(df_novo) // 50))

        self.modelo.n_estimators += n_add
        self.modelo.warm_start = True
        X = self._extrair_features(df_novo)
        self.modelo.fit(X, df_novo['valor_frete'].values)

        if self.tem_prazo_:
            df_p = df_novo.dropna(subset=['prazo_dias'])
            if not df_p.empty:
                self.modelo_prazo.n_estimators += max(20, n_add // 2)
                self.modelo_prazo.warm_start = True
                self.modelo_prazo.fit(self._extrair_features(df_p), df_p['prazo_dias'].values)

        return True

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
                      elapsed_ms: float | None = None,
                      excluidos: list[dict] | None = None) -> None:
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
    print(DIV)

    if excluidos:
        print()
        print(f"  Transportadoras não disponíveis para o CEP consultado:")
        for e in excluidos:
            print(f"  ✗  {e['transportadora']:<22}  {e['motivo']}")
    print()


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
# Suporte a pasta com múltiplas planilhas
# ---------------------------------------------------------------------------

PASTA_PADRAO   = 'planilhas'
ARQUIVO_PADRAO = 'base_frete.xlsx'


def descobrir_fonte() -> str:
    """
    Detecta automaticamente a fonte de dados, sem precisar de --planilha ou --pasta.
    Prioridade:
      1. pasta  ./planilhas/  com ao menos um .xlsx
      2. arquivo base_frete.xlsx  no diretório atual
    """
    if os.path.isdir(PASTA_PADRAO):
        arquivos = _xlsx_da_pasta(PASTA_PADRAO)
        if arquivos:
            return PASTA_PADRAO

    if os.path.exists(ARQUIVO_PADRAO):
        return ARQUIVO_PADRAO

    sys.exit(
        f"\nNenhuma fonte de dados encontrada. Opções:\n"
        f"  1. Crie a pasta '{PASTA_PADRAO}/' e adicione arquivos .xlsx\n"
        f"  2. Coloque '{ARQUIVO_PADRAO}' no diretório atual\n"
        f"  3. Use --planilha <arquivo> ou --pasta <pasta>"
    )


def _xlsx_da_pasta(pasta: str) -> list[str]:
    """Lista os .xlsx da pasta excluindo temporários e backups."""
    return [
        f for f in glob.glob(os.path.join(pasta, '*.xlsx'))
        if not os.path.basename(f).startswith('~')
    ]


def _mtime_fonte(fonte: str) -> float:
    """Retorna o mtime mais recente: do arquivo ou do .xlsx mais novo na pasta."""
    if os.path.isdir(fonte):
        arquivos = _xlsx_da_pasta(fonte)
        return max((os.path.getmtime(f) for f in arquivos), default=0.0)
    return os.path.getmtime(os.path.abspath(fonte))


_CHAVE_REGRA = [
    'transportadora',
    'cep_origem_inicio', 'cep_origem_fim',
    'cep_destino_inicio', 'cep_destino_fim',
    'peso_min_kg', 'peso_max_kg',
    'maior_lado_min_cm', 'maior_lado_max_cm',
    'cubagem_min_m3', 'cubagem_max_m3',
]


def _caminho_cache_arquivo(arq: str) -> str:
    """Cache individual oculto para cada .xlsx na mesma pasta."""
    pasta = os.path.dirname(arq)
    nome  = '.' + os.path.splitext(os.path.basename(arq))[0] + '.pkl'
    return os.path.join(pasta, nome)


def carregar_pasta(pasta: str) -> pd.DataFrame:
    """
    Lê e combina todos os .xlsx da pasta.
    Arquivos ordenados do mais antigo ao mais recente (mtime).
    Cada arquivo tem seu próprio cache — só é relido se foi modificado.
    Regras duplicadas são resolvidas mantendo a versão mais recente.
    """
    arquivos = sorted(
        _xlsx_da_pasta(pasta),
        key=os.path.getmtime,   # mais antigo primeiro → mais recente por último
    )
    if not arquivos:
        sys.exit(f"\nNenhum arquivo .xlsx encontrado em: {pasta}")

    dfs = []
    for arq in arquivos:
        cache_arq = _caminho_cache_arquivo(arq)
        if os.path.exists(cache_arq) and os.path.getmtime(cache_arq) >= os.path.getmtime(arq):
            df_arq = joblib.load(cache_arq)
            status = 'cache'
        else:
            df_arq = carregar_base(arq)
            joblib.dump(df_arq, cache_arq)
            status = 'lido'
        dfs.append(df_arq)
        print(f'    {os.path.basename(arq)}: {len(df_arq):,} regras  [{status}]')

    df_bruto = pd.concat(dfs, ignore_index=True)

    # Deduplicação: mesma regra em arquivos diferentes → versão mais recente vence
    chaves_presentes = [c for c in _CHAVE_REGRA if c in df_bruto.columns]
    df = df_bruto.drop_duplicates(subset=chaves_presentes, keep='last').reset_index(drop=True)

    duplicatas = len(df_bruto) - len(df)
    sufixo = f'  ({duplicatas:,} duplicata(s) removida(s))' if duplicatas else ''
    print(f'  Total combinado: {len(df):,} regras  |  '
          f'{df["transportadora"].nunique()} transportadoras{sufixo}')
    return df


# ---------------------------------------------------------------------------
# Cache do DataFrame em disco
# ---------------------------------------------------------------------------

def _caminho_cache_df(fonte: str) -> str:
    if os.path.isdir(fonte):
        return os.path.join(os.path.abspath(fonte), '.cache_df.pkl')
    return os.path.splitext(os.path.abspath(fonte))[0] + '_df.pkl'


def carregar_base_com_cache(fonte: str) -> pd.DataFrame:
    """
    Carrega o DataFrame do cache binário se estiver atualizado.
    Aceita um arquivo .xlsx ou uma pasta com múltiplos .xlsx.
    Reconstrói o cache automaticamente quando qualquer arquivo mudar.
    """
    cache = _caminho_cache_df(fonte)

    if os.path.exists(cache):
        if os.path.getmtime(cache) >= _mtime_fonte(fonte):
            t0 = time.perf_counter()
            df = joblib.load(cache)
            ms = (time.perf_counter() - t0) * 1000
            print(f'  {len(df):,} regras  |  {df["transportadora"].nunique()} '
                  f'transportadoras  (cache: {ms:.0f} ms)')
            return df

    df = carregar_pasta(fonte) if os.path.isdir(fonte) else carregar_base(fonte)
    joblib.dump(df, cache)
    if not os.path.isdir(fonte):
        print(f'  {len(df):,} regras  |  {df["transportadora"].nunique()} '
              f'transportadoras  (Excel parseado — cache salvo)')
    return df


# ---------------------------------------------------------------------------
# Cache do modelo ML em disco
# ---------------------------------------------------------------------------

def _caminho_cache(fonte: str) -> str:
    if os.path.isdir(fonte):
        return os.path.join(os.path.abspath(fonte), '.cache_modelo.pkl')
    return os.path.splitext(os.path.abspath(fonte))[0] + '_modelo.pkl'


def _caminho_manifest(fonte: str) -> str:
    if os.path.isdir(fonte):
        return os.path.join(os.path.abspath(fonte), '.cache_modelo_manifest.json')
    return os.path.splitext(os.path.abspath(fonte))[0] + '_modelo_manifest.json'


def _mapa_atual(fonte: str) -> dict[str, float]:
    """Retorna {nome_arquivo: mtime} para todos os .xlsx da fonte."""
    if os.path.isdir(fonte):
        return {
            os.path.basename(f): round(os.path.getmtime(f), 3)
            for f in sorted(_xlsx_da_pasta(fonte), key=os.path.getmtime)
        }
    abs_fonte = os.path.abspath(fonte)
    return {os.path.basename(abs_fonte): round(os.path.getmtime(abs_fonte), 3)}


def _carregar_df_arquivo(fonte: str, nome: str) -> pd.DataFrame:
    """Carrega o DataFrame de um arquivo específico, do cache individual se disponível."""
    arq = os.path.join(fonte, nome) if os.path.isdir(fonte) else fonte
    pkl = _caminho_cache_arquivo(arq)
    if os.path.exists(pkl) and os.path.getmtime(pkl) >= os.path.getmtime(arq):
        return joblib.load(pkl)
    return carregar_base(arq)


def carregar_ou_treinar_modelo(
    df_completo: pd.DataFrame, fonte: str
) -> tuple['ModeloFrete', float, float]:
    """
    Gerencia o ciclo de vida do modelo ML:
    - Sem modelo salvo → treino completo em todos os arquivos
    - Arquivos existentes modificados → treino completo (dados mudaram)
    - Apenas novos arquivos → treino incremental (warm_start só nos novos)
    - Nada mudou → carrega do cache
    """
    cache         = _caminho_cache(fonte)
    manifest_path = _caminho_manifest(fonte)
    atual         = _mapa_atual(fonte)

    # Carrega manifesto salvo (quais arquivos já foram aprendidos)
    manifest: dict[str, float] = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    def _salvar(modelo, mae_p, mae_t):
        joblib.dump({'modelo': modelo, 'mae_preco': mae_p, 'mae_prazo': mae_t}, cache)
        with open(manifest_path, 'w') as f:
            json.dump(atual, f, indent=2)

    def _full_retrain(motivo: str):
        print(f'  {motivo} → treino completo com {len(df_completo):,} exemplos ...')
        t0 = _now_ms()
        m = ModeloFrete()
        mae_p, mae_t = m.treinar(df_completo)
        elapsed = _now_ms() - t0
        ip = f'R$ {mae_p:.2f}' if mae_p > 0 else 'N/A'
        it = f'{mae_t:.1f} d.u.' if mae_t > 0 else 'N/A'
        print(f'  Treinado em {elapsed:.0f} ms  |  MAE preço: {ip}  |  MAE prazo: {it}')
        _salvar(m, mae_p, mae_t)
        return m, mae_p, mae_t

    # Nenhum modelo salvo ainda
    if not os.path.exists(cache):
        return _full_retrain('Primeiro treinamento')

    # Classifica cada arquivo atual em relação ao manifesto
    novos      = {n: t for n, t in atual.items() if n not in manifest}
    alterados  = {n: t for n, t in atual.items()
                  if n in manifest and manifest[n] != t}
    removidos  = {n for n in manifest if n not in atual}

    # Arquivo alterado ou removido → full retrain (dados mudaram)
    if alterados:
        return _full_retrain(f'Arquivo(s) modificado(s): {list(alterados)}')
    if removidos:
        return _full_retrain(f'Arquivo(s) removido(s): {list(removidos)}')

    # Nada mudou → cache válido
    if not novos:
        print('  Carregando modelo do cache ...')
        t0 = _now_ms()
        dados = joblib.load(cache)
        print(f'  Modelo carregado em {_now_ms() - t0:.0f} ms  '
              f'(MAE preço: R$ {dados["mae_preco"]:.2f}  |  '
              f'MAE prazo: {dados["mae_prazo"]:.1f} d.u.)')
        return dados['modelo'], dados['mae_preco'], dados['mae_prazo']

    # Apenas arquivos novos → treino incremental
    print(f'  {len(novos)} novo(s) arquivo(s): {list(novos)} → treinamento incremental ...')
    dfs_novos = [_carregar_df_arquivo(fonte, nome) for nome in novos]
    df_novo   = pd.concat(dfs_novos, ignore_index=True)

    dados  = joblib.load(cache)
    modelo = dados['modelo']

    t0      = _now_ms()
    sucesso = modelo.treinar_incremental(df_novo)
    elapsed = _now_ms() - t0

    if not sucesso:
        return _full_retrain('Nova transportadora detectada')

    print(f'  Modelo atualizado em {elapsed:.0f} ms  (+{len(df_novo):,} exemplos novos)')
    _salvar(modelo, dados['mae_preco'], dados['mae_prazo'])
    return modelo, dados['mae_preco'], dados['mae_prazo']


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

  # Modo direto — planilha única
  python calcular_frete.py \\
      --cep-origem 01310100 --cep-destino 30130110 \\
      --peso 3.5 --altura 20 --largura 30 --comprimento 40

  # Pasta com múltiplas planilhas (aprendizado evolutivo)
  python calcular_frete.py --pasta ./planilhas/ --interativo
        """,
    )
    parser.add_argument('--planilha',    default=None,
                        help='Planilha .xlsx (padrão: auto-detectado)')
    parser.add_argument('--pasta',       default=None,
                        help='Pasta com múltiplos .xlsx (padrão: auto-detectado)')
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

    # Resolve a fonte antes de qualquer interação com o usuário
    if args.pasta:
        if not os.path.isdir(args.pasta):
            sys.exit(f"\nErro: pasta não encontrada — {args.pasta}")
        fonte = args.pasta
    elif args.planilha:
        fonte = args.planilha
    else:
        fonte = descobrir_fonte()

    if os.path.isdir(fonte):
        print(f'\n  Fonte detectada: pasta {fonte}/')
    else:
        print(f'\n  Fonte detectada: {fonte}')

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

    df = carregar_base_com_cache(fonte)

    t_inicio = time.perf_counter()
    print(f'\n  Buscando correspondências exatas ...')
    exatos, excluidos = buscar_exato(df, cep_orig, cep_dest, peso, lado, cub)

    if not exatos.empty:
        print(f'  {len(exatos)} linha(s) encontrada(s). ✓')
        resultados = montar_resultados_exatos(exatos)
    else:
        if df.empty:
            print('  Base de dados vazia — nenhuma cotação disponível.')
            resultados = []
        else:
            print('  Sem correspondência exata — ativando modelo ML.')
            modelo, _, _ = carregar_ou_treinar_modelo(df, fonte)
            resultados = modelo.prever_todos(cep_orig, cep_dest, peso, lado, cub)

    elapsed_ms = (time.perf_counter() - t_inicio) * 1000
    exibir_resultados(resultados, w_preco, elapsed_ms, excluidos)


if __name__ == '__main__':
    main()
