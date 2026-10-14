# NOTES

Fluxo:

1. Transforma cada pedaço do texto em um vetor (Embedding)
2. Texto com significado parecido viram vetores próximos
3. Aí você pergunta algo e o banco de devolve os top-K trechos mais próximos.

[text](https://neo4j.com) mas conhecido como bando de grafos também pode ser utilizado como vector database
ou banco de vetores.

1. Você guarda os chunks como nós
2. Armazena embeddings como propriedades
3. Cria índices vetorial
4. Consulta por similaridade trazendo score e os top resultados.

IMPORTANT!

Vamos gerar embeddings locais utilizando Transform.js lib para JavaScript sem torar a maquina.
Não precisa de:

- chave
- docker
- cloud
- roda com node
- podemos reproduzir tudo do zero!

Porque embedding são tão importantes?
Porque K-top existe?
Como uma banco vetorial pensa?
Porque embedding vira a fundação do RAG?

Para acessar o banco de dados neo4j basta acessar o endereço: [http://localhost:7474](http://localhost:7474/browser/)

- Dado um documento em PDF quebrou ele em chunks e salvou ele como embbeding no banco de dados Neo4j
- Com esses dados podemos buscar chunk similares a pergunta.

[OpenRouter](https://openrouter.ai/)

Hoje é a melhor opção para criar uma bridge entre sua
aplicação uma LLM para que o seu negócio não fique
acoplado a uma terceira.

OpenRouter segue o padrão de configuração de API da OpenAI

[LangChain](https://www.langchain.com/)
