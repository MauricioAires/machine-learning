/**
 * Nada será instalado
 * será apenas node nativo
 */
import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "../src/server.ts";

test("routes to cheapest model by default", async () => {
  const app = createServer();

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
});
