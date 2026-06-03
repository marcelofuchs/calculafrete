# Sistema de Cálculo de Frete com Machine Learning

Calcula e rankeia automaticamente as melhores opções de frete a partir de uma base de conhecimento em planilhas Excel, combinando busca exata por faixas de regras e estimativa por Machine Learning como fallback.

---

## Funcionalidades

- **Busca exata** nas regras da planilha por CEP de origem/destino, peso, dimensão e cubagem
- **Cubagem calculada** automaticamente a partir de altura × largura × comprimento
- **Fallback ML** (Gradient Boosting) quando não há regra correspondente
- **Aprendizado incremental** — novas planilhas mensais atualizam o modelo sem retreinar tudo
- **Prazo em dias úteis** com cálculo automático de feriados nacionais brasileiros
- **Score combinado** preço × prazo configurável (0–100%)
- **Data estimada de entrega** calculada a partir de hoje
- **Auto-detecção de fonte** — lê a pasta `planilhas/` automaticamente, sem argumentos

---

## Estrutura do projeto

```
teste_frete/
├── calcular_frete.py          # Script principal
├── gerar_base_exemplo.py      # Gerador de planilha de exemplo
├── requirements.txt           # Dependências Python
├── README.md                  # Este arquivo
├── DOCUMENTACAO.md            # Documentação técnica detalhada
└── planilhas/                 # Base de conhecimento (auto-detectada)
    ├── frete_jan_2026.xlsx    # Tabela inicial — 12.096 regras, 5 transportadoras
    ├── frete_fev_2026.xlsx    # Fevereiro — 10.000 regras, 3 novas transportadoras
    ├── .cache_df.pkl          # Cache combinado (gerado automaticamente)
    ├── .cache_modelo.pkl      # Modelo ML treinado (gerado automaticamente)
    ├── .cache_modelo_manifest.json  # Registro do que já foi aprendido
    ├── .frete_jan_2026.pkl    # Cache individual por arquivo (gerado automaticamente)
    └── .frete_fev_2026.pkl
```

> Os arquivos `.pkl` e `.json` são gerados automaticamente e não devem ser editados.

---

## Instalação

```bash
# 1. Instale as dependências
pip install -r requirements.txt

# 2. Gere as planilhas de exemplo (cria planilhas/base_frete.xlsx)
python gerar_base_exemplo.py
```

**Dependências:** `pandas`, `openpyxl`, `scikit-learn`, `numpy`, `joblib`

---

## Como rodar

> **Não é necessário informar planilha ou pasta.** O sistema detecta a pasta `planilhas/` automaticamente. Basta rodar o comando.

### Modo interativo (uso manual)

```bash
python calcular_frete.py --interativo
```

O sistema detecta `planilhas/` automaticamente e pergunta os dados:

```
  Fonte detectada: pasta planilhas/
  22,096 regras  |  8 transportadoras  (cache: 2 ms)

  Informe os dados do envio:

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
# Equilíbrio entre preço e prazo (padrão — não precisa informar)
python calcular_frete.py \
  --cep-origem 01310100 --cep-destino 90040060 \
  --peso 4.2 --altura 20 --largura 30 --comprimento 40

# Prioriza preço (70% preço / 30% prazo)
python calcular_frete.py \
  --cep-origem 01310100 --cep-destino 90040060 \
  --peso 4.2 --altura 20 --largura 30 --comprimento 40 \
  --prioridade-preco 70

# Só preço importa
python calcular_frete.py \
  --cep-origem 01310100 --cep-destino 90040060 \
  --peso 4.2 --altura 20 --largura 30 --comprimento 40 \
  --prioridade-preco 100

# Só prazo importa (mais rápido ganha)
python calcular_frete.py \
  --cep-origem 01310100 --cep-destino 90040060 \
  --peso 4.2 --altura 20 --largura 30 --comprimento 40 \
  --prioridade-preco 0
```

### Exemplo de saída

```
════════════════════════════════════════════════════════════════════
  SISTEMA DE CÁLCULO DE FRETE — ML Edition
════════════════════════════════════════════════════════════════════

  Fonte detectada: pasta planilhas/
  22,096 regras  |  8 transportadoras  (cache: 2 ms)

  Parâmetros de busca:
    CEP Origem   :  01310100  (1310100)
    CEP Destino  :  90040060  (90040060)
    Peso         :  4.200 kg
    Dimensões    :  20 × 30 × 40 cm  (A × L × C)
    Maior lado   :  40.0 cm  (calculado)
    Cubagem      :  0.024000 m³  (calculado)

  Buscando correspondências exatas ...
  7 linha(s) encontrada(s). ✓

────────────────────────────────────────────────────────────────────
  RESULTADO — OPÇÕES DE FRETE DISPONÍVEIS
────────────────────────────────────────────────────────────────────
  Critério: 70% preço  /  30% prazo
────────────────────────────────────────────────────────────────────
  1°   Sequóia                 R$     63.60  6 d.u. → 12/jun  score: 0.12
  2°   Braspress               R$     61.54  9 d.u. → 17/jun  score: 0.32
  3°   TNT Mercúrio            R$     95.47  6 d.u. → 12/jun  score: 0.49
  4°   Correios SEDEX          R$    120.10  5 d.u. → 11/jun  score: 0.70
  5°   Total Express           R$    108.82  7 d.u. → 15/jun  score: 0.74
  6°   Jadlog .Package         R$     97.96  8 d.u. → 16/jun  score: 0.75
  7°   Correios PAC            R$    116.17  10 d.u. → 18/jun  score: 0.98
────────────────────────────────────────────────────────────────────
  MELHOR OPÇÃO: Sequóia  →  R$ 63.60  —  6 d.u. (entrega: 12/jun)
  Tempo de cotação: 4 ms
────────────────────────────────────────────────────────────────────
```

---

## Parâmetros do CLI

| Parâmetro | Tipo | Padrão | Descrição |
|---|---|---|---|
| `--cep-origem` | string | — | CEP de origem (com ou sem hífen) |
| `--cep-destino` | string | — | CEP de destino (com ou sem hífen) |
| `--peso` | float | — | Peso do produto em kg |
| `--altura` | float | — | Altura do produto em cm |
| `--largura` | float | — | Largura do produto em cm |
| `--comprimento` | float | — | Comprimento do produto em cm |
| `--prioridade-preco` | int (0–100) | `50` | Peso do preço no ranking (0 = só prazo, 100 = só preço) |
| `--interativo` | flag | — | Ativa entrada interativa pelo terminal |
| `--planilha` | string | — | *(opcional)* Override: força uso de um arquivo `.xlsx` específico |
| `--pasta` | string | — | *(opcional)* Override: força uso de uma pasta específica |

> **Nenhum dos dois é obrigatório.** Sem `--planilha` e sem `--pasta`, o sistema procura automaticamente:
> 1. Pasta `planilhas/` no diretório atual (prioridade)
> 2. Arquivo `base_frete.xlsx` no diretório atual (fallback)
> 3. Nenhum encontrado → exibe mensagem de erro com instruções

---

## Estrutura da planilha

Cada arquivo `.xlsx` pode ter qualquer nome e deve conter as seguintes colunas:

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

> CEPs devem ser inteiros de 8 dígitos sem hífen (`01310100`). A coluna `prazo_dias` representa **dias úteis**.

---

## Adicionando planilhas mês a mês

O sistema foi projetado para crescer incrementalmente. Coloque suas planilhas em `planilhas/` — uma por mês, contendo **apenas as linhas que mudaram ou foram adicionadas**.

### Estrutura recomendada

```
planilhas/
├── frete_jan_2026.xlsx   ← tabela completa inicial
├── frete_fev_2026.xlsx   ← só o que mudou em fevereiro
├── frete_mar_2026.xlsx   ← só o que mudou em março
└── ...
```

### O que o sistema faz quando chega um novo arquivo

```
frete_mar_2026.xlsx adicionado à pasta
         │
         ▼
  Detecta arquivo novo via manifesto
         │
         ├─ jan e fev: carregados do cache .pkl  (~5 ms cada)
         │
         └─ mar: lido do xlsx, cache individual salvo
                 │
                 ▼
         Treino incremental (warm_start)
         Apenas os dados de março são usados
         As 300+ árvores de jan/fev permanecem intactas
         ~40–80 ms  (vs ~5.000 ms do treino completo)
```

### Regra de conflito: qual valor prevalece?

Os arquivos são ordenados do **mais antigo para o mais recente** pela data de modificação. Se a mesma regra (mesma transportadora + mesmas faixas) aparecer em dois arquivos, **o mais recente vence**:

```
frete_jan_2026.xlsx  →  Braspress SP→RS  R$ 25,05
frete_fev_2026.xlsx  →  Braspress SP→RS  R$ 18,00  ← vence (arquivo mais recente)
```

### Quando o retreino completo é acionado

O treino incremental vale apenas para arquivos **novos**. Nas situações abaixo o sistema retreina do zero:

| Situação | Comportamento |
|---|---|
| Primeiro uso (sem cache) | Treino completo em todos os arquivos |
| Arquivo existente **modificado** | Treino completo (dados mudaram) |
| Arquivo existente **removido** | Treino completo (consistência) |
| Nova transportadora detectada | Treino completo (encoder precisa ser refazido) |
| Apenas novo arquivo adicionado | **Treino incremental** (~40 ms) |

### O que colocar no arquivo mensal

Apenas as linhas que sofreram alguma alteração:

- Preços reajustados por transportadora
- Novas faixas de peso, dimensão ou cubagem
- Novos pares de origem/destino no contrato
- Transportadoras novas incorporadas

O que não mudou **não precisa ser repetido** — permanece válido desde o arquivo onde foi definido.

---

## Como funciona o ranking

O score de cada opção é calculado com normalização min-max:

```
score = (prioridade_preco / 100) × preço_norm
      + (1 - prioridade_preco / 100) × prazo_norm
```

`preço_norm` e `prazo_norm` são valores em `[0, 1]` — 0 é o melhor e 1 é o pior entre as opções disponíveis. A opção com **menor score** vence.

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

> **Nota:** o sistema não possui uma base de feriados — os 12 feriados nacionais são calculados algoritmicamente a cada consulta. Feriados estaduais, municipais e pontos facultativos **não são cobertos**. Este ponto pode ser adaptado conforme a necessidade de cada operação, inclusive consultando fontes externas (APIs públicas como a do IBGE ou serviços de calendário), porém requer desenvolvimento adicional.

---

## Performance

Benchmark executado com 50 repetições, base de **22.096 regras** e **8 transportadoras** (jan + fev + mar em cache).

### Tempo de cotação

| Cenário | Mínimo | Médio | p95 | Modo |
|---|---|---|---|---|
| SP → RS | 2,8 ms | 3,1 ms | 3,5 ms | Exato |
| MG → PR | 2,3 ms | 2,5 ms | 2,8 ms | Exato |
| BA → SC | 2,6 ms | 2,8 ms | 3,1 ms | Exato |
| PI → AM | 13,1 ms | 14,4 ms | 16,3 ms | ML (inferência) |

### Inicialização (1× por execução)

| Operação | Tempo médio |
|---|---|
| Carga do df combinado (cache `.pkl`) | ~2,5 ms |
| Carga do modelo ML (cache `.pkl`) | ~12 ms |

### Treino (quando necessário)

| Operação | Tempo médio |
|---|---|
| Treino incremental — arquivo novo (~800 linhas) | ~40 ms |
| Treino completo — 22.096 linhas, 300 árvores | ~5.000 ms |

> **Em uso contínuo** com caches válidos: cotação exata responde em **~3 ms**; fallback ML em **~15 ms**. O treino incremental ocorre apenas uma vez quando um novo arquivo chega à pasta.

---

## Documentação técnica

Consulte [DOCUMENTACAO.md](DOCUMENTACAO.md) para detalhes sobre a arquitetura, modelo de ML, inferência, cache evolutivo com manifesto e como estender o sistema.
