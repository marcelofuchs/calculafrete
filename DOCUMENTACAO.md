# Documentação Técnica — Sistema de Cálculo de Frete com ML

## Sumário

1. [Visão geral da arquitetura](#1-visão-geral-da-arquitetura)
2. [Fluxo de execução](#2-fluxo-de-execução)
3. [Módulo de busca exata](#3-módulo-de-busca-exata)
4. [Modelo de Machine Learning](#4-modelo-de-machine-learning) — algoritmo, features, inferência passo a passo, cache
5. [Algoritmo de score](#5-algoritmo-de-score)
6. [Cálculo de dias úteis e feriados](#6-cálculo-de-dias-úteis-e-feriados)
7. [Base de conhecimento (planilha)](#7-base-de-conhecimento-planilha)
8. [Referência de funções](#8-referência-de-funções)
9. [Como estender o sistema](#9-como-estender-o-sistema)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Visão geral da arquitetura

```
Entrada do usuário
  (CEP orig, CEP dest, peso, maior lado, cubagem, prioridade)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                   calcular_frete.py                 │
│                                                     │
│  ┌──────────────┐     ┌─────────────────────────┐  │
│  │ carregar_base│────▶│      buscar_exato        │  │
│  │  (planilha)  │     │ (filtro por faixas)      │  │
│  └──────────────┘     └────────────┬────────────┘  │
│                                    │                │
│                          encontrou?│                │
│                        ┌───────────┴──────────┐    │
│                        │ Sim                  │Não │
│                        ▼                      ▼    │
│               montar_resultados_    ModeloFrete     │
│               exatos()              .treinar()      │
│                                     .prever_todos() │
│                        │                      │    │
│                        └──────────┬───────────┘    │
│                                   │                 │
│                                   ▼                 │
│                          calcular_score()           │
│                    (normalização min-max + pesos)   │
│                                   │                 │
│                                   ▼                 │
│                          exibir_resultados()        │
│                    (prazo em d.u. + data entrega)   │
└─────────────────────────────────────────────────────┘
        │
        ▼
   Ranking final com score, preço, prazo e data de entrega
```

---

## 2. Fluxo de execução

### Passo 1 — Entrada de dados

O sistema aceita dois modos:

- **Direto:** parâmetros via CLI (`--cep-origem`, `--peso`, etc.)
- **Interativo:** `--interativo` solicita cada valor no terminal

Todos os CEPs são normalizados para inteiro de 8 dígitos (sem hífen, sem ponto):
`01310-100` → `1310100`.

### Passo 2 — Carregamento da base

A planilha Excel é carregada via `pandas.read_excel`. O sistema valida as 12 colunas obrigatórias e interrompe com mensagem clara se alguma estiver faltando.

### Passo 3 — Busca exata

Aplica um filtro booleano vetorizado sobre o DataFrame. Uma linha da planilha satisfaz a consulta quando **todos** os cinco critérios são atendidos simultaneamente:

```
cep_origem_inicio ≤ CEP_orig ≤ cep_origem_fim
cep_destino_inicio ≤ CEP_dest ≤ cep_destino_fim
peso_min_kg ≤ peso < peso_max_kg
maior_lado_min_cm ≤ maior_lado < maior_lado_max_cm
cubagem_min_m3 ≤ cubagem < cubagem_max_m3
```

> O limite superior é **exclusivo** (`<`) para evitar que um produto no limite exato de uma faixa seja capturado por duas regras ao mesmo tempo.

Se houver múltiplas linhas para a mesma transportadora, é retida a que tiver o **menor valor de frete**.

### Passo 4 — Fallback ML

Ativado apenas quando a busca exata retorna vazio. O modelo é treinado na íntegra da base e prediz preço e prazo para cada transportadora conhecida.

### Passo 5 — Score e ordenação

Calcula o score combinado e ordena as opções. A melhor opção (menor score) é destacada.

### Passo 6 — Exibição

Converte o prazo em dias úteis para data de entrega real (contando feriados nacionais a partir de hoje) e exibe o resultado formatado.

---

## 3. Módulo de busca exata

**Função:** `buscar_exato(df, cep_orig, cep_dest, peso, lado, cubagem)`

Implementada com operações vetorizadas do pandas — **não usa loops**. Para uma base de 22.000 linhas, a execução é inferior a 10 ms.

```python
mask = (
    (df['cep_origem_inicio']  <= cep_orig)  & (df['cep_origem_fim']   >= cep_orig)  &
    (df['cep_destino_inicio'] <= cep_dest)  & (df['cep_destino_fim']  >= cep_dest)  &
    (df['peso_min_kg']        <= peso)      & (df['peso_max_kg']       >  peso)     &
    (df['maior_lado_min_cm']  <= lado)      & (df['maior_lado_max_cm'] >  lado)     &
    (df['cubagem_min_m3']     <= cubagem)   & (df['cubagem_max_m3']    >  cubagem)
)
```

**Quando a busca exata falha:**

- O CEP informado não está coberto por nenhuma faixa da planilha
- O peso/dimensão/cubagem do produto cai fora de todos os intervalos definidos
- A combinação de todos os critérios juntos não tem correspondência (mesmo que cada um individualmente tivesse)

---

## 4. Modelo de Machine Learning

### 4.1 Quando o modelo é ativado

O modelo ML é um **fallback**: só entra em cena quando a busca exata não encontra nenhuma linha na planilha que cubra simultaneamente todos os cinco parâmetros (CEP origem, CEP destino, peso, maior lado, cubagem). Situações típicas:

- Rota entre regiões sem cobertura na tabela (ex: PI → AM)
- Produto com peso ou dimensão fora dos intervalos cadastrados
- Combinação válida individualmente, mas inexistente em conjunto

### 4.2 Algoritmo

**Gradient Boosting Regressor** (scikit-learn) — ensemble de árvores de decisão rasas, treinadas em sequência. Cada árvore aprende a corrigir os **resíduos** (erros) deixados pelo conjunto anterior.

Dois modelos totalmente independentes são treinados, um para cada alvo:

| Modelo | Alvo | Estimators | Profundidade | Learning rate |
|---|---|---|---|---|
| `modelo` | `valor_frete` (R$) | 300 | 5 | 0,05 |
| `modelo_prazo` | `prazo_dias` (d.u.) | 200 | 4 | 0,05 |

### 4.3 Features — como os dados de entrada viram números

O modelo não entende CEP, transportadora ou faixa de peso diretamente. A função `_extrair_features()` transforma os dados em um vetor de **7 números** antes de qualquer predição:

```
Entrada do usuário:
  CEP origem  = 64000000   CEP destino = 69000000
  peso = 7,7 kg            maior lado  = 55 cm
  cubagem = 0,066 m³       transportadora = "Braspress"

                    ┌─────────────────────────────────────┐
                    │        _extrair_features()          │
                    └──────────────┬──────────────────────┘
                                   │
         ┌─────────────────────────▼──────────────────────────┐
         │ f1  CEP orig normalizado   64000000 / 1e7 = 6,40   │
         │ f2  CEP dest normalizado   69000000 / 1e7 = 6,90   │
         │ f3  Distância proxy        |6,40 - 6,90| = 0,50    │
         │ f4  Peso                   7,70                     │
         │ f5  Maior lado             55,00                    │
         │ f6  Cubagem                0,066                    │
         │ f7  Transportadora enc.    2  (LabelEncoder)        │
         └────────────────────────────────────────────────────┘
                                   │
                            vetor: [6.40, 6.90, 0.50, 7.70, 55.00, 0.066, 2]
```

**Diferença entre treino e inferência no uso das features:**

Durante o **treino**, os valores vêm das *faixas* da planilha — usa-se o ponto médio de cada intervalo (`peso_mid = (peso_min + peso_max) / 2`). Isso representa o valor "típico" de cada regra cadastrada.

Durante a **inferência**, os valores vêm diretamente do produto real do usuário (altura × largura × comprimento já calculados). O vetor de features é construído com o valor exato, não com um intervalo.

**Por que normalizar os CEPs por 1e7?**

CEPs são inteiros de 8 dígitos (até ~99.999.999). Sem normalização, a feature de CEP domina o gradiente durante o treino por ser ordens de magnitude maior que peso (7 kg) ou cubagem (0,06 m³). Dividir por 10.000.000 coloca todos os CEPs no intervalo [0, 10], compatível com as demais features.

**Por que incluir a distância proxy (f3)?**

A diferença absoluta entre os CEPs de origem e destino é um proxy numérico para a distância geográfica — quanto mais distantes os prefixos, maior a tendência de frete mais caro e mais demorado. Isso ajuda o modelo mesmo quando nunca viu aquele par específico de regiões no treino.

### 4.4 Como o Gradient Boosting faz uma predição (inferência)

O GBR é um **ensemble aditivo**: a predição final é a soma das contribuições de todas as árvores, partindo de uma estimativa inicial.

```
ŷ = ŷ₀  +  lr × T₁(x)  +  lr × T₂(x)  +  ...  +  lr × T₃₀₀(x)
         └─────────────────────────────────────────────────────┘
              300 árvores, cada uma com learning_rate = 0,05
```

Onde:
- `ŷ₀` é a média da variável-alvo no conjunto de treino (estimativa inicial)
- `Tᵢ(x)` é a predição da i-ésima árvore para o vetor `x`
- `lr = 0,05` encolhe a contribuição de cada árvore para evitar overfitting

**O que cada árvore faz internamente:**

Cada árvore de decisão particiona o espaço de features por divisões binárias (`se f4 ≤ 10,5 então esquerda, senão direita`). Com profundidade máxima 5, cada árvore pode ter até 2⁵ = 32 regiões terminais (folhas). A predição de uma árvore para um ponto `x` é o valor médio dos resíduos dos exemplos de treino que caem na mesma folha que `x`.

```
Árvore Tᵢ para o vetor x = [6.40, 6.90, 0.50, 7.70, 55.00, 0.066, 2]:

  f3 (distância) ≤ 0.72?  →  Sim
    f4 (peso) ≤ 12.5?     →  Sim
      f7 (transp) ≤ 2.5?  →  Sim
        f6 (cubagem) ≤ 0.08?  →  Sim
          f1 (CEP orig) ≤ 6.5?  →  Sim
            ► folha: resíduo médio = -3.21   (Braspress nessa região é mais barata)
```

Após percorrer as 300 árvores, somam-se todas as contribuições:

```
ŷ_preço = 58,40 (média inicial)
        + 0,05 × (-3,21)   # árvore 1
        + 0,05 × (+1,84)   # árvore 2
        + ...               # árvores 3–300
        = 35,33             # predição final para Braspress
```

### 4.5 Os dois modelos na prática — `prever_todos()`

Para cada cotação sem correspondência exata, o sistema itera sobre todas as transportadoras conhecidas e executa **duas inferências independentes** por transportadora:

```
Para cada transportadora T em [Braspress, Jadlog, SEDEX, ...]:

  1. Montar vetor de features com o produto real + T codificada
         x = [6.40, 6.90, 0.50, 7.70, 55.00, 0.066, enc(T)]

  2. Inferência de preço:
         preço = max(0.0, modelo.predict(x))       # nunca negativo

  3. Inferência de prazo:
         prazo = max(1, round(modelo_prazo.predict(x)))  # mínimo 1 d.u.

  4. Guardar resultado:
         { transportadora: T, valor_frete: preço, prazo_dias: prazo }
```

O `max(0.0, ...)` e `max(1, round(...))` são salvaguardas: o GBR pode extrapolir para fora do domínio de treino e produzir valores sem sentido físico (frete negativo, prazo zero).

### 4.6 Treinamento e avaliação

O dataset é dividido **85% treino / 15% validação** com `random_state=42`. O MAE (Erro Médio Absoluto) mede o desvio médio da estimativa em relação aos valores reais da tabela:

```
Modelo treinado em 5248 ms  |  MAE preço: R$ 18,59  |  MAE prazo: 0,3 d.u.
```

Um MAE de R$ 18,59 significa que, em média, a estimativa de preço erra em ±R$ 18,59. O MAE de prazo de 0,3 d.u. significa que o modelo acerta o prazo com erro inferior a meio dia útil na maioria dos casos.

### 4.7 Cache em disco — o arquivo `.pkl`

#### O que é o `.pkl`

`.pkl` é a extensão do formato **Pickle** — o mecanismo nativo do Python para converter qualquer objeto em memória em uma sequência de bytes e gravá-la no disco. Não é exclusivo de modelos de ML: você poderia usar pickle para salvar uma lista, um dicionário ou qualquer estrutura Python. No contexto deste projeto, o objeto salvo *é* um modelo de ML treinado, então na prática o arquivo funciona como um **modelo de IA empacotado e pronto para uso**.

#### O que está dentro do arquivo

```
base_frete_modelo.pkl  (~2 MB)
│
├── modelo                      GradientBoostingRegressor — preço
│   ├── estimators_             array 300×1 com todas as árvores construídas
│   ├── learning_rate           0.05
│   ├── init_                   estimativa inicial (média dos preços de treino)
│   └── ...                     demais parâmetros internos do scikit-learn
│
├── modelo_prazo                GradientBoostingRegressor — prazo
│   └── (mesma estrutura, treinado para dias úteis)
│
├── mae_preco                   18.59  (metadado de qualidade salvo junto)
└── mae_prazo                   0.3
```

As 300 árvores são **estruturas de dados puras** — listas de nós com condições (`se peso ≤ 10,5 → esquerda`) e valores nas folhas. Não há neurônios, não há pesos de rede neural. É o resultado de um algoritmo estatístico clássico, serializado byte a byte.

#### Por que salvar no disco em vez de treinar sempre

Treinar é caro porque as 300 árvores são construídas iterativamente, cada uma lendo os 22.096 exemplos para calcular os resíduos. Carregar é barato porque os bytes já representam as árvores prontas — basta reconstruir o objeto Python na memória:

```
Treinar do zero:   ler 22.096 linhas → construir 300 árvores → ~5.200 ms
Carregar do .pkl:  desserializar bytes → objeto pronto        →     10 ms
```

**Analogia:** é como compilar um programa. O código-fonte não é recompilado a cada execução — compila-se uma vez, salva-se o binário e executa-se o binário diretamente. O `.pkl` é o "binário" do modelo treinado.

#### Ciclo de vida do cache

```
Primeira chamada (sem cache ou planilha modificada):

  base_frete.xlsx
       │
       ▼ carregar_base()
  DataFrame (22.096 linhas)
       │
       ▼ ModeloFrete.treinar()       ~5.200 ms
  objeto modelo treinado
       │
       ├──► usado imediatamente para prever_todos()
       │
       └──► joblib.dump()
              base_frete_modelo.pkl  (gravado no disco)


Chamadas seguintes (cache válido):

  base_frete_modelo.pkl
       │
       ▼ joblib.load()               ~10 ms
  objeto modelo treinado
       │
       └──► usado imediatamente para prever_todos()
```

#### Invalidação automática

Antes de carregar o cache, o sistema compara o `mtime` (data de modificação no sistema de arquivos) do `.pkl` com o da planilha. Se a planilha foi salva depois do cache, o modelo é retreinado:

```python
if os.path.getmtime(cache) >= os.path.getmtime(planilha):
    # cache mais recente que a planilha → válido, carrega
else:
    # planilha modificada após o cache → descarta, retreina, salva novo .pkl
```

Isso garante que qualquer alteração na planilha — nova linha, valor corrigido, transportadora adicionada — seja automaticamente incorporada ao modelo na próxima vez que o fallback ML for necessário, sem nenhuma ação manual.

**Limitação do `mtime`:** ferramentas que sobrescrevem o arquivo preservando o timestamp original (alguns scripts com `openpyxl`, `rsync --times`, restaurações de backup) não acionam a invalidação. Nesses casos o cache permanece com dados defasados até ser deletado manualmente ou até o `.pkl` ser mais antigo que a planilha por outro motivo.

**Ganho de desempenho:**

| Cenário | Tempo |
|---|---|
| Treino completo (22.096 linhas, 300 árvores) | ~5.200 ms |
| Carga do cache `.pkl` | ~10 ms |
| Inferência pura (8 transportadoras) | < 1 ms |

### 4.8 Limitações do modelo ML

- Não interpola entre transportadoras — só estima para transportadoras que já existem na base
- Extrapola mal para regiões de CEP muito distantes do domínio de treino (ex: se a base não tem nenhuma rota para uma região, a estimativa para ela é pouco confiável)
- Transportadoras com poucos registros terão estimativas menos precisas que as com cobertura ampla
- O MAE é calculado sobre o conjunto de validação da base sintética; com dados reais, o erro tende a ser diferente

---

## 5. Algoritmo de score

**Função:** `calcular_score(resultados, w_preco)`

### Normalização min-max

Para cada critério, o valor é normalizado em `[0, 1]` onde **0 é o melhor** e **1 é o pior** entre as opções disponíveis naquela consulta:

```
preço_norm  = (preço_i  - preço_min)  / (preço_max  - preço_min)
prazo_norm  = (prazo_i  - prazo_min)  / (prazo_max  - prazo_min)
```

### Fórmula do score

```
score = w_p × preço_norm + w_t × prazo_norm

onde:
  w_p = prioridade_preco / 100
  w_t = 1 - w_p
```

### Casos especiais

| Situação | Comportamento |
|---|---|
| Todos os preços iguais | `preço_norm = 0` para todos → prazo decide |
| Todos os prazos iguais | `prazo_norm = 0` para todos → preço decide |
| Sem dado de prazo (`None`) | `prazo_norm = 0.5` (posição neutra) |

### Exemplo de cálculo

5 opções para a rota SP → RS com `--prioridade-preco 50`:

| Transportadora | Preço | Prazo | preço_norm | prazo_norm | Score (50/50) |
|---|---|---|---|---|---|
| Braspress | R$ 25,05 | 9 d.u. | 0,00 | 0,80 | **0,40** |
| Jadlog | R$ 36,32 | 8 d.u. | 0,20 | 0,60 | **0,40** |
| Total Express | R$ 42,52 | 7 d.u. | 0,31 | 0,40 | **0,35** ← melhor |
| Correios PAC | R$ 41,96 | 10 d.u. | 0,30 | 1,00 | **0,65** |
| SEDEX | R$ 81,95 | 5 d.u. | 1,00 | 0,00 | **0,50** |

---

## 6. Cálculo de dias úteis e feriados

### Feriados nacionais cobertos

O sistema calcula automaticamente para qualquer ano:

| Feriado | Tipo |
|---|---|
| Confraternização Universal (01/jan) | Fixo |
| Carnaval — 2ª e 3ª feira | Móvel (47 e 46 dias antes da Páscoa) |
| Sexta-feira Santa | Móvel (2 dias antes da Páscoa) |
| Tiradentes (21/abr) | Fixo |
| Dia do Trabalho (01/mai) | Fixo |
| Corpus Christi | Móvel (60 dias após a Páscoa) |
| Independência (07/set) | Fixo |
| Nossa Senhora Aparecida (12/out) | Fixo |
| Finados (02/nov) | Fixo |
| Proclamação da República (15/nov) | Fixo |
| Natal (25/dez) | Fixo |

### Algoritmo de Páscoa (Butcher)

A data de Páscoa é calculada pelo algoritmo de Butcher/Jones, válido para qualquer ano do calendário gregoriano, sem consultar tabelas externas:

```python
def _calcular_feriados(ano):
    a = ano % 19
    b, c = divmod(ano, 100)
    # ... (ver calcular_frete.py:39)
    pascoa = date(ano, mes_p, dia_p)
```

### Cache de feriados

Os feriados de cada ano são calculados uma única vez e armazenados no dicionário `_FERIADOS_CACHE`. Chamadas subsequentes para o mesmo ano retornam do cache sem recalcular.

### Função `data_entrega`

```python
def data_entrega(prazo_uteis: int, inicio: date = None) -> date:
```

Avança dia a dia a partir de `inicio` (padrão: hoje), contando apenas dias em que `is_dia_util(d)` retorna `True`. Retorna a data do `prazo_uteis`-ésimo dia útil.

**Não são considerados** feriados municipais ou estaduais — somente feriados nacionais.

---

## 7. Base de conhecimento (planilha)

### Estrutura atual

| Métrica | Valor |
|---|---|
| Total de linhas | 22.096 |
| Transportadoras | 8 |
| Regiões de CEP cobertas | 16 |
| Arquivo | `base_frete.xlsx` (~1,1 MB) |

### Transportadoras

| Transportadora | Perfil | Prazo base |
|---|---|---|
| Correios PAC | Econômico | 6 d.u. |
| Correios SEDEX | Expresso | 1 d.u. |
| Jadlog .Package | Econômico | 4 d.u. |
| Total Express | Padrão | 3 d.u. |
| Braspress | Carga | 5 d.u. |
| Sequóia | Expresso regional | 2 d.u. |
| Azul Cargo | Econômico | 3 d.u. |
| TNT Mercúrio | Premium expresso | 2 d.u. |

### Regiões de CEP

| Região | Faixa de CEP |
|---|---|
| SP Capital | 01000000 – 09999999 |
| SP Interior | 13000000 – 19999999 |
| RJ | 20000000 – 28999999 |
| ES | 29000000 – 29999999 |
| MG | 30000000 – 39999999 |
| BA | 40000000 – 48999999 |
| SE/AL | 49000000 – 49999999 |
| PE/PB | 50000000 – 58999999 |
| CE/RN | 59000000 – 63999999 |
| PI/MA | 64000000 – 65999999 |
| PA/AP | 66000000 – 68999999 |
| GO/DF | 70000000 – 77999999 |
| MT/MS | 78000000 – 79999999 |
| PR | 80000000 – 87999999 |
| SC | 88000000 – 89999999 |
| RS | 90000000 – 99999999 |

### Script gerador

`gerar_base_exemplo.py` gera dados sintéticos com precificação realista:

```
valor = fator_base × 12 × dist_factor
      + por_kg × peso_mid × dist_factor
      + por_m3 × cubagem_mid × dist_factor
```

Onde `dist_factor = 1.0 + |i_orig - i_dest| × 0.12` cresce conforme a distância entre as regiões.

Para serviços expressos, aplica **peso cubado** (300 kg/m³):

```
peso_cubado = cubagem_m3 × 300
peso_efetivo = max(peso_real, peso_cubado)
```

---

## 8. Referência de funções

### `calcular_frete.py`

| Função / Classe | Descrição |
|---|---|
| `normalizar_cep(valor)` | Converte CEP de qualquer formato para inteiro de 8 dígitos |
| `carregar_base(caminho)` | Lê e valida a planilha Excel; interrompe com erro claro se inválida |
| `buscar_exato(df, ...)` | Filtro vetorizado pandas por todas as faixas simultaneamente |
| `montar_resultados_exatos(exatos)` | Agrupa por transportadora e retém o menor valor |
| `ModeloFrete.treinar(df)` | Treina os dois GBR (preço e prazo), retorna `(mae_preco, mae_prazo)` |
| `ModeloFrete.prever_todos(...)` | Estima preço e prazo para cada transportadora conhecida |
| `calcular_score(resultados, w_preco)` | Normalização min-max + ponderação → lista ordenada por score |
| `_calcular_feriados(ano)` | Retorna `set[date]` com 12 feriados nacionais do ano |
| `is_dia_util(d)` | `True` se `d` não é fim de semana nem feriado nacional |
| `data_entrega(prazo_uteis, inicio)` | Data real de entrega contando `prazo_uteis` dias úteis |
| `exibir_resultados(resultados, w_preco)` | Formata e imprime o ranking com score, prazo e data |
| `coletar_interativo()` | Lê os 6 parâmetros do terminal, retorna tupla |
| `main()` | Ponto de entrada: parse de args → carrega → busca → exibe |

---

## 9. Como estender o sistema

### Adicionar novas transportadoras à planilha

Basta incluir novas linhas na planilha com o nome da transportadora e as faixas de cobertura. O sistema as detecta automaticamente ao carregar o arquivo.

### Adicionar feriados estaduais ou municipais

Modifique `_calcular_feriados` para receber um estado como parâmetro e adicione os feriados locais ao `set` retornado:

```python
def _calcular_feriados(ano: int, estado: str = 'BR') -> set[date]:
    feriados = { ... }  # nacionais

    if estado == 'SP':
        feriados.add(date(ano, 7, 9))   # Revolução Constitucionalista
    if estado == 'RJ':
        feriados.add(date(ano, 1, 20))  # São Sebastião

    return feriados
```

### Adicionar um terceiro critério ao score (ex: confiabilidade)

1. Adicione a coluna `confiabilidade` na planilha (ex: 1–5)
2. Inclua-a em `montar_resultados_exatos` e `prever_todos`
3. Ajuste `calcular_score` para receber `w_confiabilidade` como terceiro peso
4. Adicione `--prioridade-confiabilidade` ao argparse

### Trocar o algoritmo de ML

Substitua `GradientBoostingRegressor` por qualquer estimador compatível com a API scikit-learn (`fit` / `predict`). Sugestões para bases muito grandes:

```python
from sklearn.ensemble import RandomForestRegressor   # mais rápido para treinar
from sklearn.ensemble import HistGradientBoostingRegressor  # nativo para dados faltantes
from lightgbm import LGBMRegressor                   # muito mais rápido (requer pip install lightgbm)
```

### Usar múltiplas planilhas

```python
import glob, pandas as pd

arquivos = glob.glob('tabelas/*.xlsx')
df = pd.concat([pd.read_excel(f) for f in arquivos], ignore_index=True)
```

---

## 10. Troubleshooting

### "Colunas obrigatórias ausentes"

A planilha não tem todos os nomes de coluna esperados. Verifique a lista completa em [README.md](README.md#estrutura-da-planilha) e renomeie as colunas conforme necessário.

### "Arquivo não encontrado"

Verifique se `base_frete.xlsx` está no mesmo diretório de execução, ou passe o caminho completo com `--planilha /caminho/absoluto/tabela.xlsx`.

### Todos os resultados vêm com `[ML (estimativa)]`

Nenhuma linha da planilha cobre a combinação de parâmetros informada. Verifique se:

- O CEP informado está dentro de alguma faixa da planilha
- O peso/dimensão/cubagem estão dentro de algum intervalo coberto
- A combinação CEP × peso × cubagem existe (pode ser que cada critério individualmente exista mas não juntos)

### O modelo ML demora muito para treinar

Com bases acima de 50.000 linhas, o treinamento pode demorar alguns segundos. Para acelerar, reduza `n_estimators`:

```python
# Em ModeloFrete.__init__, linha ~186
self.modelo = GradientBoostingRegressor(n_estimators=100, ...)  # padrão: 300
```

Ou substitua por `HistGradientBoostingRegressor` que é significativamente mais rápido.

### Score empata entre duas opções

O sistema retém a ordem de inserção do pandas em caso de empate no score. Para desempatar explicitamente pelo preço, altere a chave de ordenação em `calcular_score`:

```python
return sorted(resultados, key=lambda r: (r['score'], r['valor_frete']))
```
