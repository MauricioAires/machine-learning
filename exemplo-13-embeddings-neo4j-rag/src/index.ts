import { Neo4jVectorStore } from "@langchain/community/vectorstores/neo4j_vector";
import { CONFIG } from "./config.ts";
import { DocumentProcessor } from "./documentProcessor.ts";
import { type PretrainedOptions } from "@huggingface/transformers";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { ChatOpenAI } from "@langchain/openai";
import { AI } from "./ai.ts";
import { mkdir, writeFile } from "node:fs";

let _neo4jVectorStore = null;

async function clearAll(vectorStore: Neo4jVectorStore, nodeLabel: string) {
  console.log("🗑️ Removendo todos os documentos existentes...");

  try {
    await vectorStore.query(`MATCH (n:\`${nodeLabel}\`) DETACH DELETE n`);
  } catch (err) {
    console.log(err);
  }

  console.log("✅ Documentos removidos com sucesso!\n");
}

try {
  console.log("🚀 Inicializando sistema de Embeddings com Neo4j...\n");

  const documentProcessor = new DocumentProcessor(CONFIG.pdf.path, CONFIG.textSplitter);

  const documents = await documentProcessor.loadAndSplit();

  const embeddings = new HuggingFaceTransformersEmbeddings({
    model: CONFIG.embedding.modelName,
    pretrainedOptions: CONFIG.embedding.pretrainedOptions as PretrainedOptions,
  });

  /**
   * Quando eu quero resposta humana NLP Natural Language Processor
   */
  const nlp = new ChatOpenAI({
    temperature: CONFIG.openRouter.temperature,
    maxRetries: CONFIG.openRouter.maxRetries,
    modelName: CONFIG.openRouter.nlpModel,
    openAIApiKey: CONFIG.openRouter.apiKey,

    configuration: {
      baseURL: CONFIG.openRouter.url,
      defaultHeaders: CONFIG.openRouter.defaultHeaders,
    },
  });

  // const response = await embeddings.embedQuery("JavaScript");
  // const response = await embeddings.embedDocuments(["a"]);
  // console.log(response);
  // console.log(documents);

  _neo4jVectorStore = await Neo4jVectorStore.fromExistingGraph(embeddings, CONFIG.neo4j);

  await clearAll(_neo4jVectorStore, CONFIG.neo4j.nodeLabel);

  for (const [index, doc] of documents.entries()) {
    console.log(`✅ Adicionando documento ${index + 1}/${documents.length}`);

    await _neo4jVectorStore.addDocuments([doc]);
  }

  console.log("\n✅ Base de dados populada com sucesso!\n");

  // ==================== STEP 2: RUN SIMILARITY SEARCH ====================
  console.log("🔍 ETAPA 2: Executando buscas por similaridade...\n");

  const questions = [
    "O que são tensores e como são representados em JavaScript?",
    "Como converter objetos JavaScript em tensores?",
    "O que é normalização de dados e por que é necessária?",
    "Como funciona uma rede neural no TensorFlow.js?",
    "O que significa treinar uma rede neural?",
    "o que é hot enconding e quando usar?",
  ];

  const ai = new AI({
    debugLog: console.log,
    vectorStore: _neo4jVectorStore,
    nlpModel: nlp,
    promptConfig: CONFIG.promptConfig,
    templateText: CONFIG.templateText,
    topK: CONFIG.similarity.topK,
  });

  for (const index in questions) {
    const question = questions[index];

    console.log(`\n${"=".repeat(80)}`);
    console.log(`📌 PERGUNTA: ${question}`);
    console.log("=".repeat(80));

    const result = await ai.answerQuestions(question!);

    if (result.error) {
      console.log(`\n❌ Erro: ${result.error}\n`);
      continue;
    }

    /**
     * Adicionar o resultado em um arquivo para que eu possa analisar a formação, porque
     * gerar as respostas custam token, então dessa forma vou salvar em um arquivo para
     * analisar a resposta com mais calma.
     */

    console.log(`\n${result.answer}\n`);

    mkdir(
      CONFIG.output.answerFolder,
      {
        recursive: true,
      },
      () => {
        console.log("Pasta criada!");
      },
    );

    const fileName = `${CONFIG.output.answerFolder}/${CONFIG.output.filename}-${index}-${Date.now()}.md`;

    writeFile(fileName, result.answer!, () => {
      console.log("Arquivo criado!");
    });

    // Cleanup
    console.log(`\n${"=".repeat(80)}`);
    console.log("✅ Processamento concluído com sucesso!\n");
  }
} catch (err) {
  console.info(`[ERRO]: ${err}`);
} finally {
  /**
   * Lembrar de encerrar a conexão.
   */
  await _neo4jVectorStore?.close();
}
