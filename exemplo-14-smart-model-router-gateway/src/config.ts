console.assert(process.env.OPENROUTER_API_KEY, "OPENROUTER_API_KEY is not set in env variables");

export type ModelConfig = {
  apiKey: string;
  httpReferer: string;
  xTitle: string;
  port: number;
  models: string[];
  temperature: number;
  maxTokens: number;
  systemPrompt: string;

  provider: {
    sort: {
      by: string;
      partition: string;
    };
  };
};

export const config: ModelConfig = {
  apiKey: process.env.OPENROUTER_API_KEY as string,
  httpReferer: "http://pos-ai.com", // Aqui geralmente é domínio
  xTitle: "SmartModelRouterGateway",
  port: 3000,
  models: [
    // Não encontrei o top 4 então vai esse mesmo
    // "google/gemma-4-26b-a4b-it:free",
    // top 4 para a listagem ordenada por preço // NOT FREE
    // "arcee-ai/trinity-large-preview:free",
    // top 3 para listagem de throughput
    // "nvidia/nemotron-3-nano-30b-a3b:free", // NOT FREE,
    // Segundo de teste para que a LLM escolha qual modelo utilizar
    // "google/gemma-4-31b-it:free", // RATE LIMIT
    // "google/gemma-4-26b-a4b-it:free", // RATE limit
    // MODELOS QUE REALMENTE ESTÃO FUNCIONANDO
    // "recraft/recraft-v4.1-flash-v2:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    // "nvidia/nemotron-3.5-super-vl:free",
    // "inclusionai/ling-3.2-flash:free", // RATE LIMIT
    // "baidu/ernie-5.5-flash-preview:free",
    // "liquid/lfm-3-1.7b:free",
    "recursive/flint-v1.0:free",
  ],
  temperature: 0.2,
  maxTokens: 100,
  systemPrompt: "Your are a helpful assistant.",
  provider: {
    sort: {
      // by: "price",
      // by: "latency",
      by: "throughput",
      partition: "none",
    },
  },
};
