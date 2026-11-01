# A voz ajuda o Jev a detectar sarcasmo? (MUStARD)

**Resposta, com estes sinais: não.** A voz não mudou a detecção, e os sinais de voz sozinhos
quase não separam sarcasmo de fala sincera.

## Desenho

[MUStARD](https://github.com/soujanyaporia/MUStARD) (Castro et al., ACL 2019, MIT): 690 falas
de séries em inglês (Friends, The Big Bang Theory, Golden Girls, Sarcasmaholics), 345
sarcásticas, com as falas anteriores como contexto e o vídeo da fala-alvo.

- Áudio da fala-alvo extraído do vídeo, mono 16 kHz ([`prepare.py`](prepare.py)).
- Sinais do app (`sentiment_classifier.audio.acoustics`): volume, tom, **variação do tom**
  (acrescentada antes de rodar, por ser associada a sarcasmo na literatura) e velocidade,
  relativos à mediana daquele personagem no dataset, mais a leitura em palavras.
- Uma pergunta `Noul` ("a fala é sarcástica ou irônica?"), igual nas duas condições. Cada fala
  foi ao Jev duas vezes, **texto** e **texto + voz**, em ordem aleatória ([`run.py`](run.py)).
- 1.380 requisições, sem falhas.

```bash
python benchmarks/mustard/prepare.py   # baixe antes mmsd_raw_data.zip e sarcasm_data.json
python benchmarks/mustard/run.py
```

## Resultado (24/10/2026, uma execução)

| condição | AUC | macro-F1 | acerto | p média (sarcástica / sincera) |
|---|---:|---:|---:|---:|
| texto | 0,814 | 0,697 | 0,704 | 0,55 / 0,26 |
| texto + voz | 0,809 | 0,698 | 0,706 | 0,54 / 0,25 |

- **ΔAUC (voz − texto): −0,005**, IC95% [−0,015; +0,005] (bootstrap pareado).
- **Δmacro-F1: +0,001**, IC95% [−0,016; +0,018].
- A voz mudou a resposta (limiar 0,5) em 37 de 690 falas, para os dois lados.
- **Só os números de voz** (regressão logística, validação cruzada em 5 dobras): **AUC 0,525**,
  praticamente o acaso.

Por série (AUC texto → texto + voz): BBT 0,809 → 0,821 (n=280); Friends 0,855 → 0,832
(n=356). Golden Girls tem 39 sarcásticas e 1 sincera, então a AUC dela não diz nada; nota-se
apenas que o sarcasmo seco da Dorothy, sem pista no texto, passa abaixo de 0,5 na maioria.

## Leitura

- **O Jev detecta sarcasmo bem só pelo texto e pelo contexto** (AUC 0,81, sem treino).
- **Nossos sinais de voz não carregam a informação.** Com AUC 0,525 sozinhos, não havia o
  que o Jev aproveitar. O problema está nos sinais, não em como foram entregues.
- Por que os sinais falham aqui (hipóteses, não testadas): medianas da fala inteira apagam a
  ênfase numa palavra só ("*Oh*, great"); o clipe tem risada de plateia e silêncio nas bordas;
  é atuação de comédia, não atendimento.
- O resultado das ligações sintéticas (a voz baixando o sentimento) **não** se estende ao
  sarcasmo: lá era volume e interrupção explícitos, injetados de propósito.

## O que tentar, se valer a pena

1. Prosódia por palavra, usando os tempos do Whisper: ênfase, alongamento, pausa antes da
   palavra-chave, em vez de medianas da fala inteira.
2. Um modelo pré-treinado de emoção na fala (por exemplo, emotion2vec ou wav2vec2) como mais
   um sinal no JSON, mantendo o Jev como quem decide.
3. Testar em fala de atendimento real, onde o sinal relevante é irritação (volume,
   interrupção), não sarcasmo de sitcom.
