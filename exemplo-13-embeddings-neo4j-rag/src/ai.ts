import { type Neo4jVectorStore } from "@langchain/community/vectorstores/neo4j_vector";
import { SimilarityMetric } from "@langchain/community/vectorstores/rockset";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { ChatOpenAI } from "@langchain/openai";
import { debuglog } from "node:util";

type DebugLog = (...args: unknown[]) => void;

export interface Params {
  debugLog: DebugLog;
  vectorStore: Neo4jVectorStore;
  nlpModel: ChatOpenAI;
  promptConfig: any;
  templateText: string;
  topK: number;
}

/**
 * Chain tem esse nome porque é uma corrente! => pipeline =>
 * rebe a resposta de uma pergunta e manda para a outra
 * em cadeia.
 */
export interface ChainState {
  question: string;
  context?: string;
  topScore?: number;
  error?: string;
  answer?: string;
}

export class AI {
  private params: Params;

  constructor(params: Params) {
    this.params = params;
  }

  async retrieveVectorSearchResults(input: ChainState): Promise<ChainState> {
    this.params.debugLog("🔎 Buscando resultados de similaridade no Neo4j...");

    const vectorResult = await this.params.vectorStore.similaritySearchWithScore(input.question, this.params.topK);

    if (!vectorResult.length) {
      this.params.debugLog("⚠️  Nenhum resultado encontrado no vector store.");
      return {
        ...input,
        error: "Desculpe, não encontrei informações relevantes sobre essa pergunta na base de conhecimento.",
      };
    }

    /**
     * Para obter um melhor resultado o banco deve conter mais dados ou até mesmo chunk menores.
     */
    const topScore = vectorResult[0]![1];

    this.params.debugLog(`\n✅ Encontrados ${vectorResult.length} resultados relevantes (melhor score: ${topScore.toFixed(3)})`);

    const contexts = vectorResult
      .filter(([, score]) => score > 0.5)
      .map(([doc]) => doc.pageContent)
      .join("\n\n---\n\n");

    return {
      ...input,
      context: contexts,
      topScore,
    };
  }

  async generateNLPResponse(input: ChainState): Promise<ChainState> {
    if (input.error) return input;

    this.params.debugLog("\n🤖 Gerando reposta com IA...");

    const responsePrompt = ChatPromptTemplate.fromTemplate(this.params.templateText);

    const responseChain = responsePrompt.pipe(this.params.nlpModel).pipe(new StringOutputParser());

    /**
     * Ficou um pouco redundante porque criamos um JSON prompt mas estamos traduzindo ele para texto
     * mas estamos fazendo isso apenas porque o LangChain esta fazendo várias operações que facilita.
     *
     * Mas JSON Prompt ainda faz muito sentido quando estamos utilizando API diretamente sem
     * qualquer API (Como a LangChain)
     */
    const rawResponse = await responseChain.invoke({
      role: this.params.promptConfig.role,
      task: this.params.promptConfig.task,
      tone: this.params.promptConfig.tone,
      language: this.params.promptConfig.language,
      format: this.params.promptConfig.format,
      instructions: this.params.promptConfig.instructions.map((instruction: string, idx: number) => `${idx + 1}. ${instruction}`).join("\n"),
      question: input.question,
      context: input.context,
    });

    return {
      ...input,
      answer: rawResponse,
    };
  }

  async answerQuestions(question: string) {
    const chain = RunnableSequence.from([this.retrieveVectorSearchResults.bind(this), this.generateNLPResponse.bind(this)]);

    const result = await chain.invoke({
      question,
    });

    this.params.debugLog(`\n🎙️ Pergunta:`);
    this.params.debugLog(question, "\n");
    this.params.debugLog(`💭 Resposta:`);
    this.params.debugLog(result.answer || result.error, "\n");

    return result;
  }
}

/**
 * 1. Primeiro passo obter os dados de similaridade do Neo4J que é
 * buscar informação da nossa base de dados para ser utilizada como
 * contexto para answer da AI
 */

/**
 * O JSON Prompt é utilizado para que a AI seja o mais
 * determinista possível
 *
 */
