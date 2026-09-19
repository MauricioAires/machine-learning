import { Neo4jVectorStore } from "@langchain/community/vectorstores/neo4j_vector";
import { CONFIG } from "./config.ts";
import { DocumentProcessor } from "./documentProcessor.ts";
import { type PretrainedOptions } from "@huggingface/transformers";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { displayResults } from "./util.ts";

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
  console.log("Inicializando sistema de Embeddings com Neo4j...\n");

  const documentProcessor = new DocumentProcessor(CONFIG.pdf.path, CONFIG.textSplitter);

  const documents = await documentProcessor.loadAndSplit();

  const embeddings = new HuggingFaceTransformersEmbeddings({
    model: CONFIG.embedding.modelName,
    pretrainedOptions: CONFIG.embedding.pretrainedOptions as PretrainedOptions,
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

  console.log("\n✅ Base de dados populada com sucesso!");

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
  for (const question of questions) {
    console.log(`\n${"=".repeat(80)}`);
    console.log(`📌 PERGUNTA: ${question}`);
    console.log("=".repeat(80));

    const results = await _neo4jVectorStore.similaritySearch(question, CONFIG.similarity.topK);

    /**
     * Resumindo esse é o RAG o banco de dados está retornando uma lista com chunks
     * de dados similares a pergunta, esses dados vão ser utilizados como contexto
     * para responder.
     */
    displayResults(results);
    // console.log(results);
  }
} catch (err) {
  console.info(`[ERRO]: ${err}`);
} finally {
  /**
   * Lembrar de encerrar a conexão.
   */
  await _neo4jVectorStore?.close();
}
