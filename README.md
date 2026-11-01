\# Classificador Inteligente de Chamados de Manutenção Industrial



API-first com \*\*motor híbrido\*\*: modelo clássico (scikit-learn) + LLM (Gemini). Recebe a descrição de um chamado em português e devolve categoria (Mecânica, Elétrica, Automação), prioridade (Baixa, Média, Alta), confiança, explicação e ação sugerida. Compara, combina e audita os dois motores e aprende com correções humanas.



\## Arquitetura



```

Streamlit (UI) --HTTP--> FastAPI --> \[ ML Engine | Gemini Engine | Hybrid Router ] --> SQLite

```



O Streamlit não contém lógica de ML nem chama o Gemini: só consome a API.



```

app/            API, engines (ml, gemini, hybrid), regras de segurança, persistência

ml/             geração do dataset, treino e benchmark

ui/app.py       cliente Streamlit

tests/          pytest (Gemini sempre mockado)

data/ models/ reports/

```



Fluxo do modo `hybrid`:



1\. O ML classifica (rápido e sem custo).

2\. Se a confiança fica abaixo de `ML\_CONFIDENCE\_THRESHOLD` (padrão 0,65), o chamado é escalado ao Gemini.

3\. Se ML e Gemini discordam, a resposta marca `revisao\_humana` e devolve as duas opiniões.

4\. As regras de segurança (fumaça, incêndio, choque, vazamento de gás, emergência etc.) elevam a prioridade para Alta e ativam `regra\_aplicada`.

5\. Se o Gemini falha ou a chave não existe, a API segue com o ML e informa `fallback` e `motivo\_fallback`.



\## Como obter a chave do Gemini



1\. Acesse https://aistudio.google.com/apikey e crie uma chave.

2\. Copie `.env.example` para `.env` e preencha `GEMINI\_API\_KEY` e `GEMINI\_MODEL`.

3\. A chave só é lida do ambiente e nunca é registrada em logs. Sem chave, o sistema funciona só com ML.

4\. \*\*Nunca versione o `.env`.\*\* Ele está no `.gitignore`.



\## Configuração (`.env`)



Os valores do `.env` têm prioridade sobre os padrões definidos em `app/config.py`. Depois de alterar o `.env`, reinicie a API: o `--reload` observa apenas arquivos `.py`.



| Variável | Padrão no código | Descrição |

|---|---|---|

| `GEMINI\_API\_KEY` | vazio | Chave da API. Sem ela, só o ML funciona. |

| `GEMINI\_MODEL` | vazio | Nome do modelo (ex.: `gemini-3.8-flash`). Modelos antigos podem ser aposentados para contas novas. |

| `GEMINI\_TIMEOUT\_SECONDS` | 45 | Tempo máximo de cada tentativa. Em testes, 20 s foi insuficiente. |

| `GEMINI\_MAX\_ATTEMPTS` | 3 | Tentativas por chamado. Cada tentativa conta na cota da API. |

| `ML\_CONFIDENCE\_THRESHOLD` | 0,65 | Abaixo disso, o modo híbrido escala ao Gemini. |



\## Passo a passo



Linux/macOS:



```bash

python -m venv .venv \&\& source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env

python -m ml.gerar\_dataset

python -m ml.train

pytest

uvicorn app.main:app --reload

streamlit run ui/app.py

python -m ml.benchmark --amostra 30

```



Windows (cmd):



```bat

python -m venv .venv

.venv\\Scripts\\activate

pip install -r requirements.txt

copy .env.example .env

python -m ml.gerar\_dataset

python -m ml.train

pytest

uvicorn app.main:app --reload

```



Em um segundo cmd (também com o venv ativado), na mesma pasta:



```bat

streamlit run ui/app.py

```



A API e a interface rodam ao mesmo tempo, cada uma em seu terminal. A API não tem página na raiz: abrir `http://localhost:8000` mostra `{"detail":"Not Found"}`, o que é esperado. Use `/docs` ou `/health`.



Benchmark sem Gemini: `python -m ml.benchmark --sem-gemini`. O conjunto de teste (`data/test\_frozen.csv`) é congelado: `gerar\_dataset` recusa sobrescrevê-lo sem `--forcar`.



Docker:



```bash

docker compose run --rm api python -m ml.gerar\_dataset

docker compose run --rm api python -m ml.train

docker compose up --build

```



API em http://localhost:8000/docs e UI em http://localhost:8501.



\## Exemplos de curl



```bash

curl localhost:8000/health



curl -X POST localhost:8000/classificar -H 'Content-Type: application/json' \\

&#x20; -d '{"texto": "Motor da esteira desarmando com cheiro de queimado", "modo": "hybrid"}'



curl -X POST localhost:8000/classificar/lote -H 'Content-Type: application/json' \\

&#x20; -d '{"textos": \["CLP sem comunicação", "Lubrificação programada do redutor"], "modo": "ml"}'



curl 'localhost:8000/chamados?categoria=Elétrica\&status=pendente\&pagina=1\&tamanho=20'



curl -X PATCH localhost:8000/chamados/1/correcao -H 'Content-Type: application/json' \\

&#x20; -d '{"categoria": "Automação", "prioridade": "Média"}'



curl -X POST localhost:8000/chamados/1/confirmar



curl localhost:8000/metricas



curl -X POST localhost:8000/retreinar



curl localhost:8000/benchmark

```



\## Resultados



\### Benchmark medido (conjunto de teste congelado: 82 chamados sintéticos)



| Motor | Acc. categoria | F1-macro categoria | Acc. prioridade | F1-macro prioridade | Latência média | p95 | Custo/1000 |

|---|---|---|---|---|---|---|---|

| ML (regressão logística) | 0,939 | 0,940 | 1,000 | 1,000 | 7,0 ms | 7,2 ms | US$ 0 |

| Gemini | não medido | | | | | | |

| Híbrido | não medido | | | | | | |



A prioridade chega a 100% porque, no dataset sintético, ela é determinada por palavras-chave dos templates. Esse número não deve ser lido como desempenho esperado em chamados reais.



\*\*Por que o Gemini não foi medido:\*\* o plano gratuito da API permite 20 requisições por dia por modelo (`gemini-3.8-flash`), e a cota se esgotou durante os testes manuais. O benchmark completo exige conta com cobrança ativa ou amostra reduzida (ver "Rodando o benchmark do Gemini").



\### Observações manuais do Gemini (não são benchmark)



Testes isolados com `gemini-3.8-flash`, sem rigor estatístico:



\- Classificação com sucesso em 10 a 49 s por chamado. Em algumas chamadas houve timeout (20 s e 60 s) ou erro 503 de alta demanda do provedor.

\- Custo estimado pelo próprio app: cerca de US$ 0,0005 a 0,0007 por chamada (aprox. 745 tokens de entrada e 130 a 175 de saída).

\- Exemplo de roteamento híbrido: o chamado "esteira parando sozinha no turno da noite" teve confiança de 57% no ML (Mecânica), foi escalado ao Gemini, que classificou como Automação (CLP/sensores) com confiança de 80%, e a resposta marcou revisão humana por divergência. Um único exemplo não prova que o Gemini acerta mais.

\- O cache em memória funcionou: repetir o mesmo texto retornou em menos de 1 ms, com custo zero.

\- O fallback para o ML funcionou nos casos de timeout, 503 e modelo recusado.

\- Foi configurado `thinking\_budget=0` na chamada ao Gemini para reduzir o tempo de raciocínio do modelo. Em um teste direto à API, 1 chamada sem o limite levou 74 s com 363 tokens de raciocínio, e outra com `thinking\_budget=0` levou 7 s. \*\*O efeito dentro do app não foi confirmado\*\*: as chamadas pelo app continuaram em 46 a 49 s. A cota esgotou antes de medir o campo `thoughts\_token\_count`.



\### Rodando o benchmark do Gemini



No plano gratuito, use amostra pequena e uma tentativa por chamado, para os retries não consumirem a cota:



```

GEMINI\_MAX\_ATTEMPTS=1      # no .env, e reinicie a API

python -m ml.benchmark --amostra 8

```



O benchmark pode chamar o Gemini mais de uma vez por amostra (modos `llm` e `hybrid`). Com cobrança ativa, use `--amostra 30` e substitua a tabela acima por `reports/benchmark.md`.



\## Solução de problemas



| Sintoma | Causa provável | O que fazer |

|---|---|---|

| `fallback` com `requisicao\_recusada` | Modelo inexistente ou aposentado, ou chave inválida | Confira `GEMINI\_MODEL` e a chave. O erro 404 do Google indica o modelo sugerido. |

| `fallback` com `timeout` | Chamada passou do limite de tempo | Aumente `GEMINI\_TIMEOUT\_SECONDS` no `.env` e reinicie a API. |

| `fallback` com `erro\_servidor` (503) | Alta demanda no Google | Tente de novo mais tarde. |

| `fallback` com `rate\_limit` (429) | Cota esgotada (plano gratuito: 20 req/dia/modelo) | Espere a renovação, reduza `GEMINI\_MAX\_ATTEMPTS` ou ative cobrança. |

| Mudei o `.env` e nada mudou | A API lê o `.env` só ao iniciar | Pare o uvicorn (Ctrl+C) e suba de novo. |

| `'pytest' não é reconhecido` ou `No module named ...` | Venv não ativado no terminal | Rode `.venv\\Scripts\\activate` (Windows). O prompt deve mostrar `(.venv)`. |

| `API offline` na interface | A API não está rodando | Suba o uvicorn em outro terminal. |



\## Decisões de projeto



\- \*\*Por que híbrido:\*\* o ML responde em milissegundos e sem custo, mas é frágil em chamados ambíguos. O Gemini generaliza melhor, mas custa, tem latência e envia dados a terceiros. O ML cuida do volume e o LLM entra só quando a confiança é baixa.

\- \*\*Regras de segurança fora dos modelos:\*\* palavras críticas são uma salvaguarda determinística e auditável (`regra\_aplicada`). A regra é conservadora e pode gerar falsos positivos, como "teste do botão de emergência" em uma preventiva.

\- \*\*Saída estruturada do Gemini:\*\* schema Pydantic como `response\_schema`, validado de novo na resposta. JSON inválido ou valor fora das classes entra no ciclo de retry.

\- \*\*Retry e fallback:\*\* timeout, rate limit, erro de servidor e resposta inválida são tentados de novo com backoff exponencial. Esgotadas as tentativas, a API responde com o ML e informa o motivo.

\- \*\*Prompt injection:\*\* o chamado vai entre `<chamado>` e `</chamado>`, as marcas dentro do texto são removidas e o prompt de sistema declara o conteúdo como dado.

\- \*\*Sem vazamento de dados:\*\* TF-IDF dentro do pipeline na validação cruzada, teste congelado usado só para relatar, e correções com texto igual ao do teste congelado são descartadas no retreino.

\- \*\*Retreino:\*\* só correções humanas e confirmações entram como rótulo. O novo modelo sempre substitui o anterior, e as métricas antes/depois são reportadas sobre o mesmo teste congelado. Não há promoção condicionada ao teste, porque isso o transformaria em conjunto de ajuste.

\- \*\*Custo:\*\* preços por milhão de tokens são parametrizados no `.env`. O custo é estimado a partir do `usage\_metadata` da resposta. Respostas em cache custam zero.



\## Limitações



\- O dataset é sintético e templatizado: variações do mesmo template caem em treino e teste, o que infla as métricas. Desempenho em chamados reais não está garantido.

\- O desempenho do Gemini não foi medido em benchmark (ver "Resultados"). A latência observada, de dezenas de segundos, é alta para uso interativo e depende da carga do provedor.

\- Resultados do Gemini variam por modelo e versão. Reexecute o benchmark ao trocar `GEMINI\_MODEL`.

\- O plano gratuito da API Gemini limita a 20 requisições por dia por modelo, e cada retry conta.

\- Enviar descrições de manutenção a uma API externa tem implicações de custo, privacidade e confidencialidade. Remova nomes, tags de ativos e dados sensíveis antes, ou avalie uma opção com contrato de dados adequado.

\- O cache do Gemini é em memória e se perde ao reiniciar a API.

\- Não há autenticação nem rate limit na API.

\- O ML explica por coeficientes (ou importâncias) de termos, uma aproximação para modelos não lineares.



\## Próximos passos



\- Confirmar se `thinking\_budget=0` reduz a latência no app (medir `thoughts\_token\_count`) e, se necessário, atualizar o SDK `google-genai`.

\- Rodar o benchmark do Gemini e do híbrido com cota suficiente e preencher a tabela de resultados.

\- Validar com chamados reais anonimizados e um conjunto de teste rotulado por técnicos.

\- Calibrar o limiar do híbrido pela curva custo × acurácia.

\- Autenticação, rate limit e fila para lotes grandes.

\- Migrar SQLite para PostgreSQL e agendar o retreino com monitoramento de drift.

