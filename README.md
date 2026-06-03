# Sistema de Cálculo de Frete com Machine Learning

Calcula e rankeia automaticamente as melhores opções de frete a partir de uma base de conhecimento (planilha Excel), combinando busca exata por faixas de regras e estimativa por Machine Learning como fallback.

---

## Funcionalidades

- **Busca exata** nas regras da planilha por CEP de origem/destino, peso, dimensão e cubagem
- **Fallback ML** (Gradient Boosting) quando não há regra correspondente
- **Prazo em dias úteis** com cálculo automático de feriados nacionais brasileiros
- **Score combinado** preço × prazo configurável por parâmetro (0–100%)
- **Data estimada de entrega** calculada a partir de hoje

---

## Estrutura do projeto

```
teste_frete/
├── calcular_frete.py       # Script principal
├── gerar_base_exemplo.py   # Gerador de planilha de exemplo
├── base_frete.xlsx         # Base de conhecimento (22.096 regras, 8 transportadoras)
├── requirements.txt        # Dependências Python
├── README.md               # Este arquivo
└── DOCUMENTACAO.md         # Documentação técnica detalhada
```

---

## Instalação

```bash
# 1. Instale as dependências
pip install -r requirements.txt

# 2. (Opcional) Gere a planilha de exemplo para testes
python gerar_base_exemplo.py
```

**Dependências:** `pandas`, `openpyxl`, `scikit-learn`, `numpy`

---

## Uso rápido

### Modo interativo

```bash
python calcular_frete.py --interativo
```

O sistema vai perguntar cada parâmetro:

```
  CEP de origem      (ex: 01310-100): 01310-100
  CEP de destino     (ex: 30130-110): 90040-060
  Peso (kg)                         : 4.2
  Altura (cm)                       : 20
  Largura (cm)                      : 30
  Comprimento (cm)                  : 40
  Prioridade preço 0-100 (padrão 50): 70
```

### Modo direto (scripts / integração)

```bash
python calcular_frete.py \
  --cep-origem 01310100 \
  --cep-destino 90040060 \
  --peso 4.2 \
  --altura 20 \
  --largura 30 \
  --comprimento 40 \
  --prioridade-preco 70
```

### Exemplo de saída

```
════════════════════════════════════════════════════════════════════
  SISTEMA DE CÁLCULO DE FRETE — ML Edition
════════════════════════════════════════════════════════════════════

  Parâmetros de busca:
    CEP Origem :  01310100
    CEP Destino:  90040060
    Peso       :  4.200 kg
    Maior lado :  38.0 cm
    Cubagem    :  0.01800 m³

  22,096 regras  |  8 transportadoras
  6 linha(s) encontrada(s). ✓

────────────────────────────────────────────────────────────────────
  RESULTADO — OPÇÕES DE FRETE DISPONÍVEIS
────────────────────────────────────────────────────────────────────
  Critério: 70% preço  /  30% prazo
────────────────────────────────────────────────────────────────────
  1°   Sequóia                 R$     50.34  6 d.u. → 12/jun  score: 0.19
  2°   Braspress               R$     25.05  9 d.u. → 17/jun  score: 0.24
  3°   Total Express           R$     42.52  7 d.u. → 15/jun  score: 0.34
  4°   Jadlog .Package         R$     36.32  8 d.u. → 16/jun  score: 0.36
  5°   Correios SEDEX          R$     81.95  5 d.u. → 11/jun  score: 0.70
  6°   Correios PAC            R$     41.96  10 d.u. → 18/jun  score: 0.73
────────────────────────────────────────────────────────────────────
  MELHOR OPÇÃO: Sequóia  →  R$ 50.34  —  6 d.u. (entrega: 12/jun)
────────────────────────────────────────────────────────────────────
```

---

## Parâmetros do CLI

| Parâmetro | Tipo | Padrão | Descrição |
|---|---|---|---|
| `--planilha` | string | `base_frete.xlsx` | Caminho para a planilha de regras |
| `--cep-origem` | string | — | CEP de origem (com ou sem hífen) |
| `--cep-destino` | string | — | CEP de destino (com ou sem hífen) |
| `--peso` | float | — | Peso do produto em kg |
| `--altura` | float | — | Altura do produto em cm |
| `--largura` | float | — | Largura do produto em cm |
| `--comprimento` | float | — | Comprimento do produto em cm |
| `--prioridade-preco` | int (0–100) | `50` | Peso do preço no ranking (0 = só prazo, 100 = só preço) |
| `--interativo` | flag | — | Ativa entrada interativa pelo terminal |

---

## Estrutura da planilha

A planilha pode ter qualquer nome. Deve conter as seguintes colunas:

| Coluna | Tipo | Exemplo | Obrigatória |
|---|---|---|---|
| `transportadora` | texto | `Jadlog .Package` | Sim |
| `cep_origem_inicio` | inteiro | `1000000` | Sim |
| `cep_origem_fim` | inteiro | `9999999` | Sim |
| `cep_destino_inicio` | inteiro | `30000000` | Sim |
| `cep_destino_fim` | inteiro | `39999999` | Sim |
| `peso_min_kg` | float | `1.0` | Sim |
| `peso_max_kg` | float | `5.0` | Sim |
| `maior_lado_min_cm` | float | `0` | Sim |
| `maior_lado_max_cm` | float | `80` | Sim |
| `cubagem_min_m3` | float | `0.0` | Sim |
| `cubagem_max_m3` | float | `0.1` | Sim |
| `valor_frete` | float | `45.90` | Sim |
| `prazo_dias` | inteiro | `4` | Não (dias úteis) |

> Os CEPs devem ser inteiros de 8 dígitos sem hífen (`01310100`). A coluna `prazo_dias` representa **dias úteis**.

---

## Como funciona o ranking

O score de cada opção é calculado com normalização min-max:

```
score = (prioridade_preco / 100) × preço_norm
      + (1 - prioridade_preco / 100) × prazo_norm
```

Onde `preço_norm` e `prazo_norm` são valores em `[0, 1]` — 0 é o melhor e 1 é o pior entre as opções disponíveis. A opção com **menor score** vence.

| `--prioridade-preco` | Comportamento |
|---|---|
| `100` | Ordena só por preço (ignora prazo) |
| `70` | Preço pesa 70%, prazo pesa 30% |
| `50` | Equilíbrio entre preço e prazo |
| `0` | Ordena só por prazo (ignora preço) |

---

## Cálculo de dias úteis

O sistema considera como dias **não úteis**:

- Sábados e domingos
- 12 feriados nacionais brasileiros (fixos + móveis)

Feriados móveis calculados automaticamente para qualquer ano: Páscoa, Carnaval (2ª e 3ª), Sexta-feira Santa e Corpus Christi.

---

## Usando sua própria planilha

```bash
python calcular_frete.py \
  --planilha /caminho/para/sua_tabela.xlsx \
  --interativo
```

Basta manter os nomes de coluna conforme a tabela acima. Valores de CEP com hífen são normalizados automaticamente.

---

## Performance

Benchmark executado com 50 repetições, base de **22.096 regras** e **8 transportadoras**.

### Tempo por cotação (df já em memória)

| Cenário | Mínimo | Médio | p95 | Modo |
|---|---|---|---|---|
| SP → RS | 2,8 ms | 3,4 ms | 5,6 ms | Exato |
| MG → PR | 2,4 ms | 2,8 ms | 3,6 ms | Exato |
| BA → SC | 2,6 ms | 2,8 ms | 3,3 ms | Exato |
| PI → AM (1ª chamada) | — | ~5.200 ms | — | ML — treino + salva cache |
| PI → AM (2ª chamada+) | — | **10 ms** | — | ML — carrega cache `.pkl` |

### Operações de inicialização (1× por execução)

| Operação | Tempo médio |
|---|---|
| Carga da planilha Excel (22.096 linhas) | ~1.400–2.100 ms |
| Treino do modelo ML (300 árvores, sem cache) | ~5.200 ms |
| Carga do modelo do cache `.pkl` | ~10 ms |

> **Em uso contínuo** (df em memória, cache `.pkl` presente): cotações por busca exata respondem em **~3–5 ms**; fallback ML responde em **~10 ms**.
> O cache é invalidado automaticamente quando a planilha é modificada.

---

## Documentação técnica

Consulte [DOCUMENTACAO.md](DOCUMENTACAO.md) para detalhes sobre a arquitetura, o modelo de ML, inferência passo a passo, cache `.pkl` e como estender o sistema.
