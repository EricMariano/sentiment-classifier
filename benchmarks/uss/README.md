# Satisfação (CSAT) no JDDC: original em chinês e traduzido para português

Pergunta: dá para estimar a nota de satisfação de uma conversa de atendimento sem treinar
nada, só perguntando ao Jev? E isso se mantém em português?

## Dados

[USS](https://github.com/sunnweiwei/user-satisfaction-simulation) (Sun et al., SIGIR 2021),
subconjunto **JDDC**: atendimento humano real do JD.com, em chinês, com nota de 1 a 5 dada
por anotadores para a conversa inteira (usamos a média dos anotadores, por isso há notas
como 2,33). Não existe conjunto público em PT-BR com conversa e CSAT real; este é o proxy
mais próximo que encontramos.

Amostras estratificadas pela nota arredondada (a distribuição real do JDDC é quase toda 4, e
só existem 2 conversas com nota 1). **Insatisfeita** = nota real abaixo de 2,5.

```bash
python benchmarks/uss/prepare.py JDDC --por-nota 8 --saida benchmarks/data/jddc_piloto.jsonl
python benchmarks/uss/translate.py benchmarks/data/jddc_piloto.jsonl --para pt   # e --para en
python -m sentiment_classifier lote benchmarks/data/jddc_piloto_pt.jsonl --saida benchmarks/results/jddc_piloto_pt
python -m sentiment_classifier csat avaliar benchmarks/results/jddc_piloto_pt.jsonl
```

## 1. Piloto: 34 conversas, três línguas

Traduções com `gpt-5-nano`, cada conversa inteira de uma vez, com instrução para manter o tom
e não suavizar ([`translate.py`](translate.py)). O inglês entra como controle: se o inglês
fica bem e o português cai, o problema é o português, não a tradução.

**A mesma conversa em línguas diferentes** (estimativa de satisfação do Jev):

| par | Spearman | diferença média | maior diferença |
|---|---:|---:|---:|
| chinês × português | 0,96 | 0,11 | 0,36 |
| chinês × inglês | 0,95 | 0,17 | 0,52 |
| português × inglês | 0,94 | 0,14 | 0,71 |

**Contra a nota real** (estimativa do Jev; 10 conversas insatisfeitas):

| língua | MAE | Spearman | insatisfeitas achadas | alarmes falsos |
|---|---:|---:|---:|---:|
| chinês (original) | 0,69 | 0,80 | 6 de 10 | 2 |
| português | 0,73 | 0,75 | 6 de 10 | 2 |
| inglês | 0,74 | 0,77 | 6 de 10 | 2 |

A tradução preserva o sinal, e o português não fica atrás do inglês. Sobraram caracteres
chineses em 3% das mensagens em português (1,6% em inglês): quase tudo marcadores do dataset
(`[时间x]`, `[电话x]`), nomes de produto e endereços.

## 2. Em escala: 297 conversas em português

Amostra de 302 conversas (75 por nota, seed 1). A tradução foi feita em duas partes: 207
conversas com `gpt-5-nano` (até acabarem os créditos) e 90 com Mistral Small 3.2 pela
OpenRouter (o campo `tradutor` de cada linha diz qual). Cinco conversas longas (28 a 65
mensagens) ficaram de fora: o Mistral juntava ou dividia mensagens e a checagem rejeitou.
Notas: 2 / 74 / 74 / 73 / 74 conversas com nota 1 / 2 / 3 / 4 / 5 (arredondada). Cerca de
7 mil requisições ao Jev, nenhuma falha.

| método | MAE | Spearman | insatisfeitas achadas (de 76) | alarmes falsos |
|---|---:|---:|---:|---:|
| sempre a média | 0,94 | – | 0 | 0 |
| sentimento final do cliente (Jev por turno) | 0,73 | 0,63 | 45 | 21 |
| **estimativa do Jev** (1 `Score` na conversa) | 0,74 | **0,71** | **54** | 27 |
| modelo calibrado (ridge, CV 5 dobras) | **0,58** | **0,76** | 27 | 5 |

- Spearman da estimativa do Jev: IC95% [0,64; 0,76] (bootstrap).
- Ordena bem, mas comprime a escala: quase nunca passa de 4,5.
- Com quase 300 conversas, o modelo calibrado passa a ganhar também na ordenação (0,76) e
  erra menos o valor, mas fica conservador: acha só 27 das 76 insatisfeitas, com 5 alarmes
  falsos. Os sinais que mais pesam são a estimativa do Jev e o sentimento médio do cliente.

**Por tradutor.** Nas 207 do `gpt-5-nano` o Spearman foi 0,74; nas 90 do Mistral, 0,63. Para
saber se era o tradutor, as mesmas 90 foram traduzidas também com `gpt-5-nano` pela
OpenRouter (89 passaram nas checagens) e o Jev estimou a satisfação nas duas versões
([`translator_check.py`](translator_check.py)):

| | Spearman com a nota real | MAE | insatisfeitas achadas |
|---|---:|---:|---:|
| tradução Mistral Small 3.2 | 0,63 | 0,81 | 16 de 24 |
| tradução `gpt-5-nano` | 0,66 | 0,78 | 16 de 24 |

- A mesma conversa, traduzida pelos dois, recebe quase a mesma nota (Spearman 0,94 entre as
  versões; diferença média 0,21 ponto).
- ΔSpearman +0,03, IC95% [−0,04; +0,09]: **o tradutor não explica a diferença.** Aquelas 90
  conversas são, simplesmente, mais difíceis.
- Pela OpenRouter, o `gpt-5-nano` só traduziu com `--metodo json_schema`; com o padrão
  (`function_calling`), devolveu o original em chinês em 49 de 90 conversas, e a checagem de
  texto não traduzido rejeitou todas. Custo do teste: ~US$ 0,80, quase tudo em tokens de
  raciocínio do `gpt-5-nano` (o Mistral traduziu as 90 por US$ 0,16).

Na primeira versão desta seção, com 207 conversas (só `gpt-5-nano`): estimativa do Jev
Spearman 0,74 e 38 de 52 insatisfeitas; modelo calibrado MAE 0,60 e Spearman 0,74.

## Leitura

- **Sem treino, a estimativa do Jev ordena bem as conversas em português** (Spearman 0,71) e
  acha 71% das insatisfeitas, com cerca de 1 alarme falso para cada 2 acertos.
- **Calibrar com notas reais compensa** no valor e, com ~300 conversas, também na ordem; mas o
  modelo calibrado fica conservador. Use a estimativa pura para triagem (achar insatisfeitos)
  e o calibrado quando precisar do número.
- Limites: conversas traduzidas do chinês, não PT-BR nativo; nota de anotador, não do cliente;
  amostra estratificada (na vida real há bem menos insatisfeitos, e a precisão cai).

## Correção registrada

A primeira versão desta página contava como insatisfeitas só as conversas com nota ≤ 2,0. A
nota é média de anotadores, e as conversas de 2,33 ficavam de fora, embora a amostragem as
contasse como nota 2. A definição passou a ser nota < 2,5, e o piloto foi recalculado: a
estimativa do Jev acha 6 de 10 insatisfeitas, não as 8 de 10 relatadas antes.

O piloto em chinês rodou antes de as perguntas por turno ganharem a frase sobre sinais de
`voz`; a pergunta de satisfação não mudou.

## Próximo passo

Separar o efeito do tradutor (mesmas conversas, dois tradutores) e repetir com conversas reais
em PT-BR com a nota da pesquisa, quando houver.
