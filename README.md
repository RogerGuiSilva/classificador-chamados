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



O plano gratuito é suficiente para este projeto, mas os limites variam por modelo (ver "Cotas e escolha do modelo").



\## Configuração (`.env`)



Os valores do `.env` têm prioridade sobre os padrões definidos em `app/config.py`. Depois de alterar o `.env`, reinicie a API: o `--reload` observa apenas arquivos `.py`.



| Variável | Padrão no código | Descrição |

|---|---|---|

| `GEMINI\_API\_KEY` | vazio | Chave da API. Sem ela, só o ML funciona. |

| `GEMINI\_MODEL` | vazio | Nome do modelo. O usado nos resultados é `gemini-3.5-flash-lite`. |

| `GEMINI\_TIMEOUT\_SECONDS` | 45 | Tempo máximo de cada tentativa. |

| `GEMINI\_MAX\_ATTEMPTS` | 3 | Tentativas por chamado. Cada tentativa conta na cota. |

| `GEMINI\_BACKOFF\_BASE\_SECONDS` | 1 | Espera base entre tentativas (dobra a cada tentativa). Em lotes no plano gratuito, 5 ajuda a respeitar o limite por minuto. |

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

python -m streamlit run ui/app.py

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



\### Benchmark: 30 chamados do teste congelado



Amostra de 30 dos 82 chamados sintéticos do conjunto de teste congelado, modelo `gemini-3.5-flash-lite`, 0 chamadas do Gemini substituídas pelo ML (relatório completo em `reports/benchmark.md`).



| Motor | Acc. categoria | F1 categoria | Acc. prioridade | F1 prioridade | Latência média | p95 | Custo/1000 (estimado) |

|---|---|---|---|---|---|---|---|

| ML (regressão logística) | 0,900 | 0,895 | 1,000 | 1,000 | 7,7 ms | 7,7 ms | US$ 0 |

| Gemini | 0,967 | 0,965 | 0,933 | 0,939 | 4,5 s | 13,1 s | ver nota |

| Híbrido | 0,900 | 0,895 | 1,000 | 1,000 | 7,7 ms | 7,7 ms | US$ 0 |



Com o conjunto completo (82 chamados), o ML obteve acurácia de categoria 0,939, F1-macro 0,940 e prioridade 1,000.



\*\*Como ler esses números:\*\*



\- \*\*A amostra é pequena.\*\* A diferença de categoria entre Gemini (29/30) e ML (27/30) é de 2 chamados e não permite concluir que um motor é melhor que o outro.

\- \*\*O híbrido foi idêntico ao ML porque nenhum chamado foi escalado:\*\* o ML teve confiança igual ou acima de 0,65 em todos os 30. O benchmark, portanto, \*\*não exercita\*\* o roteador do modo híbrido. Em testes manuais com chamados ambíguos, o escalonamento ocorreu (ver abaixo).

\- \*\*A prioridade do ML chega a 100%\*\* porque, no dataset sintético, ela é determinada por palavras-chave dos templates. Esse número não deve ser lido como desempenho esperado em chamados reais.

\- \*\*Latência do Gemini:\*\* média de 4,5 s e p95 de 13,1 s no benchmark; em testes manuais isolados, cerca de 1,3 a 1,5 s. A variação não foi investigada.

\- \*\*Custo:\*\* o valor é estimado pelo app a partir dos tokens e dos preços configurados (US$ 0,30 por milhão de tokens de entrada e US$ 2,50 de saída, valores de referência de outro modelo, não os do `gemini-3.5-flash-lite`). No plano gratuito não há cobrança.

\- O relatório também lista as discordâncias entre ML e Gemini (3 em 30). O critério de contagem do campo "quem acertou" ainda não foi verificado e não é usado aqui como resultado.



\### Exemplo manual de roteamento híbrido



Com o modelo `gemini-3.8-flash`, o chamado "esteira parando sozinha no turno da noite" teve confiança de 57% no ML (Mecânica), foi escalado ao Gemini, que classificou como Automação (CLP/sensores) com confiança de 80%, e a resposta marcou revisão humana por divergência. Um único exemplo não prova que o Gemini acerta mais.



\### Cotas e escolha do modelo



O modelo inicial, `gemini-3.8-flash`, mostrou-se inadequado no plano gratuito:



\- Latência de 10 a 58 s por chamado, com tokens de raciocínio ativos (205 a cerca de 400 por chamada) e vários erros 503 e 504 do provedor.

\- `thinking\_budget=0` não desligou o raciocínio nesse modelo, o nível `MINIMAL` não é suportado, e o teste com `LOW` terminou em 504 sem resposta.

\- Cota diária de 20 requisições, esgotada durante os testes (erro 429).



Os modelos `gemini-3.5-flash-lite` e `gemini-3.1-flash-lite` responderam em cerca de 1,3 a 1,5 s sem tokens de raciocínio. Foi adotado o `gemini-3.5-flash-lite`, cujo limite gratuito, conforme o painel do AI Studio, é de 500 requisições por dia e 15 por minuto. O `gemini-2.5-flash` e o `gemini-2.5-flash-lite` retornam 404 para contas novas.



Para lotes no plano gratuito, use `GEMINI\_MAX\_ATTEMPTS=4` e `GEMINI\_BACKOFF\_BASE\_SECONDS=5`, para que recusas por limite por minuto sejam repetidas em vez de cair no ML. Os limites mudam com o tempo: confira em https://ai.dev/rate-limit.



\## Solução de problemas



| Sintoma | Causa provável | O que fazer |

|---|---|---|

| `fallback` com `requisicao\_recusada` | Modelo inexistente ou aposentado, ou chave inválida | Confira `GEMINI\_MODEL` (o nome completo, por exemplo `gemini-3.5-flash-lite`) e a chave. O erro 404 do Google indica o modelo sugerido. |

| `fallback` com `timeout` | Chamada passou do limite de tempo | Aumente `GEMINI\_TIMEOUT\_SECONDS` no `.env`, reinicie a API ou troque para um modelo mais rápido. |

| `fallback` com `erro\_servidor` (5xx) | Alta demanda ou erro no Google | Tente de novo mais tarde. |

| `fallback` com `rate\_limit` (429) | Cota esgotada, por minuto ou por dia, conforme o modelo | Espere a renovação, aumente o backoff ou use outro modelo. |

| Mudei o `.env` e nada mudou | A API lê o `.env` só ao iniciar | Pare o uvicorn (Ctrl+C) e suba de novo. |

| `'pytest' não é reconhecido`, `No module named ...` | Venv não ativado no terminal | Rode `.venv\\Scripts\\activate` (Windows). O prompt deve mostrar `(.venv)`. |

| `No module named 'plotly'` no Streamlit | Streamlit iniciado fora do venv | Ative o venv e use `python -m streamlit run ui/app.py`. |

| `API offline` na interface | A API não está rodando | Suba o uvicorn em outro terminal. |



Para descobrir quais modelos a sua chave enxerga, liste com o SDK (`client.models.list()`) e confira o campo `supported\_actions`.



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

\- O benchmark do Gemini usou 30 chamados, amostra pequena demais para comparar os motores com segurança, e não exercitou o roteador híbrido (0% escalado).

\- Os preços usados na estimativa de custo não são os do modelo adotado.

\- Resultados do Gemini variam por modelo e versão. Reexecute o benchmark ao trocar `GEMINI\_MODEL`. Modelos antigos são aposentados para contas novas, e os limites gratuitos mudam.

\- Enviar descrições de manutenção a uma API externa tem implicações de custo, privacidade e confidencialidade. Remova nomes, tags de ativos e dados sensíveis antes, ou avalie uma opção com contrato de dados adequado.

\- O cache do Gemini é em memória e se perde ao reiniciar a API.

\- Não há autenticação nem rate limit na API.

\- O ML explica por coeficientes (ou importâncias) de termos, uma aproximação para modelos não lineares.



\## Próximos passos



\- Exercitar o roteador híbrido: avaliar o limiar de confiança em chamados ambíguos ou reais, onde o ML escale ao Gemini, e medir custo × acurácia.

\- Repetir o benchmark com o conjunto de teste completo (82 chamados) e, se possível, com mais de uma amostra.

\- Corrigir os preços de referência para os do modelo adotado e verificar o critério do campo "quem acertou" no relatório.

\- Validar com chamados reais anonimizados e um conjunto de teste rotulado por técnicos.

\- Autenticação, rate limit e fila para lotes grandes.

\- Migrar SQLite para PostgreSQL e agendar o retreino com monitoramento de drift.

