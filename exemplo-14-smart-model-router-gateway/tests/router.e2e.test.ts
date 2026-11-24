/**
 * Nada será instalado
 * será apenas node nativo
 */
import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "../src/server.ts";
import { config } from "../src/config.ts";
import { type LLMResponse, OpenRouterService } from "../src/openRouterService.ts";

console.assert(process.env.OPENROUTER_API_KEY, "OPENROUTER_API_KEY is not set in env variables");

test("routes to cheapest model by default", async () => {
  const customConfig = {
    ...config,
    provider: {
      ...config.provider,
      sort: {
        ...config.provider.sort,
        by: "price",
      },
    },
  };

  const routerService = new OpenRouterService(customConfig);

  const app = createServer(routerService);

  /**
   * esse inject é utilizado exatamente para realizar testes.
   *
   */
  const response = await app.inject({
    method: "POST",
    url: "/chat",
    body: {
      question: "What is rate limiting?",
    },
  });

  assert.equal(response.statusCode, 200);
  const body = response.json() as LLMResponse;

  assert.equal(body.model, "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free");
});

test("routes to highest throughput model by default", async () => {
  const customConfig = {
    ...config,
    provider: {
      ...config.provider,
      sort: {
        ...config.provider.sort,
        by: "throughput",
      },
    },
  };

  const routerService = new OpenRouterService(customConfig);

  const app = createServer(routerService);

  /**
   * esse inject é utilizado exatamente para realizar testes.
   *
   */
  const response = await app.inject({
    method: "POST",
    url: "/chat",
    body: {
      question: "What is rate limiting?",
    },
  });

  assert.equal(response.statusCode, 200);
  const body = response.json() as LLMResponse;

  /**
   * Esse teste só é valido por agora o price ou o throughput pode mudar
   * e você deve ajustar isso de acordo com o valor atual.
   */
  assert.equal(body.model, "nvidia/nemotron-3-ultra-550b-a55b:free");
});
