# Sentiment Classifier

Analisa conversas de atendimento em português, a partir de chats ou gravações. Mostra a
evolução do sentimento, pontos de virada, alertas de cancelamento ou escalada e uma
estimativa de satisfação.

O fluxo usa LangGraph para coordenar a análise dos turnos com o Jev, da TypeSafe. Um LLM
gera o resumo dos momentos relevantes, e casos sinalizados podem passar por revisão humana.

## Como rodar

### 1. Baixe o projeto

É necessário ter Git e Python 3.10 ou superior instalados. No terminal:

```bash
git clone https://github.com/EricMariano/sentiment-classifier.git
cd sentiment-classifier
```

Se você já baixou o projeto, basta abrir o terminal na pasta dele.

### 2. Crie o ambiente e instale as dependências

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[app,audio,openrouter]"
```

### 3. Configure as APIs (opcional)

Para usar o Jev e os resumos com LLM, crie o arquivo de configuração:

```bash
cp .env.example .env
```

Abra o arquivo `.env` em um editor e preencha:

- `TYPESAFE_API_KEY`: chave para a análise com o Jev.
- `OPENROUTER_API_KEY`: chave para o resumo com LLM pela OpenRouter. O modelo padrão está
  definido em `CONVERSA_CHAT_MODEL` no mesmo arquivo.

Você pode pular esta etapa para experimentar o modo offline, que usa uma heurística local e
resumos por template. Nesse modo, a análise não usa o Jev nem um LLM.

### 4. Inicie a aplicação

```bash
.venv/bin/python -m streamlit run app.py
```

Abra no navegador o endereço exibido no terminal. Para encerrar a aplicação, pressione
`Ctrl+C` no terminal.

### 5. Analise uma conversa

Na aba **Conversa**, cole um diálogo com uma fala por linha, identificada por `Cliente:` ou
`Atendente:`, e clique em **Analisar**. Por exemplo:

```text
Cliente: Meu pedido ainda não chegou.
Atendente: Vou verificar o que aconteceu.
```

Você também pode usar a aba **Ligação** para uma gravação ou **Lote** para analisar várias
conversas. Há exemplos em [`samples/`](samples/).

Para áudio estéreo, a transcrição é local. Para áudio mono com a OpenAI, instale também
o extra abaixo e configure `OPENAI_API_KEY` no `.env`; nesse caso, o áudio original é enviado
à OpenAI. A anonimização do texto usa padrões e pode não identificar todos os dados pessoais.

```bash
.venv/bin/python -m pip install -e ".[openai]"
```

### Linha de comando

```bash
.venv/bin/python -m sentiment_classifier analisar samples/04_sarcasmo.txt
.venv/bin/python -m sentiment_classifier ligacao samples/audio/ligacao_irritada.flac
.venv/bin/python -m sentiment_classifier lote samples/ --saida relatorio
```

Para consultar as opções disponíveis:

```bash
.venv/bin/python -m sentiment_classifier --help
```
