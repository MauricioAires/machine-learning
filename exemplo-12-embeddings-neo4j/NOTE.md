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
