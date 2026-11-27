import { END, MessagesZodMeta, START, StateGraph } from "@langchain/langgraph";
import { withLangGraph } from "@langchain/langgraph/zod";
import { BaseMessage } from "langchain";
import { z } from "zod/v3";
import { identifyIntent } from "./nodes/identifyIntentNode.ts";
import { chatResponseNode } from "./nodes/chatResponseNode.ts";
import { upperCaseNode } from "./nodes/upperCaseNode.ts";
import { lowerCaseNode } from "./nodes/lowerCaseNode.ts";

const graphState = z.object({
  messages: withLangGraph(z.custom<BaseMessage[]>(), MessagesZodMeta),
  output: z.string(),
  command: z.enum(["uppercase", "lowercase", "unknown"]),
});

export type GraphState = z.infer<typeof graphState>;

export function buildGraph() {
  const workflow = new StateGraph({
    state: graphState,
  })
    .addNode("identifyIntent", identifyIntent)
    .addNode("chatResponse", chatResponseNode)

    .addNode("upperCase", upperCaseNode)
    .addNode("lowerCase", lowerCaseNode)
    // .addNode("identifyIntent", (state: GraphState) => {
    //   return {
    //     ...state,
    //     output: "test",
    //   };
    // })
    .addEdge(START, "identifyIntent")
    .addConditionalEdges(
      "identifyIntent",
      (state: GraphState) => {
        switch (state.command) {
          case "uppercase": {
            return "uppercase";
          }
          case "lowercase": {
            return "lowercase";
          }
          default: {
            return "fallback";
          }
        }
      },
      {
        uppercase: "upperCase",
        lowercase: "lowerCase",
      },
    )
    .addEdge("upperCase", "chatResponse")
    .addEdge("lowerCase", "chatResponse")
    .addEdge("chatResponse", END);

  return workflow.compile();
}
