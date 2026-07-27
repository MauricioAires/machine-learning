source .env

API_URL="https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_SITE_URL="http://localhost:3000"
OPENROUTER_SITE_NAME="My Example"


# Natural Language Processing (NLP)
NLP_MODEL="google/gemma-4-26b-a4b-it:free"

curl --silent -X POST "$API_URL" \
 -H "Content-Type: application/json" \
 -H "Authorization: Bearer $OPENROUTER_API_KEY" \
 -H "HTTP-Referrer: $OPENROUTER_SITE_URL" \
 -H "X-Title: $OPENROUTER_SITE_NAME" \
 -d '
    {
      "model": '"'$NLP_MODEL'"',
      "messages": [
        {
          "role": "user",
          "content": "Me conta uma curiosidade uma curiosidade sobre LLMs"
        }
      ],
      "temperature": 0.3,
      "max_tokens": 1_000,

    }
  ' | jq


#   {
#   "id": "gen-1785175967-0yUbjrUVCs9RPx0pedNA",
#   "object": "chat.completion",
#   "created": 1785175967,
#   "model": "google/gemma-4-26b-a4b-it:free",
#   "provider": "Darkbloom",
#   "system_fingerprint": null,
#   "service_tier": null,
#   "choices": [
#     {
#       "index": 0,
#       "logprobs": null,
#       "finish_reason": "stop",
#       "native_finish_reason": "stop",
#       "message": {
#         "role": "assistant",
#         "content": "Aqui vai uma curiosidade fascinante que ajuda a entender por que os LLMs (como o ChatGPT) às vezes parecem \"alucinar\" ou cometer erros bobos:\n\n### O fenômeno da \"Atenção\" e a falta de um \"Mundo Real\"\n\nDiferente de um ser humano, um LLM **não tem um modelo mental do mundo**. Ele não sabe o que é uma \"maçã\" no sentido de saber que ela é doce, redonda, que cai de uma árvore e que você pode morder.\n\nPara o modelo, a palavra \"maçã\" é apenas um **vetor matemático** (uma lista de números) que tem uma alta probabilidade de aparecer perto das palavras \"comer\", \"fruta\" ou \"vermelha\".\n\n**A curiosidade técnica:**\nO \"cérebro\" do LLM funciona através de um mecanismo chamado **Self-Attention (Auto-Atenção)**. Em vez de entender conceitos, o modelo calcula a relação estatística entre cada palavra de uma frase e todas as outras palavras da mesma frase.\n\n**Por que isso é importante?**\nIsso explica por que os LLMs são tão bons em imitar estilos de escrita e resolver problemas complexos de lógica, mas falham miseravelmente em tarefas simples de contagem ou aritmética básica se não forem estimulados corretamente. \n\nPor exemplo: se você perguntar *\"Quantas letras tem a palavra 'banana'?\"*, um LLM pode errar porque ele não \"vê\" as letras; ele vê o token (o bloco de texto) \"banana\". Para ele, \"banana\" é um símbolo único, e ele precisa \"prever\" qual número de letras costuma vir depois desse símbolo, em vez de contar visualmente.\n\n**Em resumo:** Um LLM é como um mestre da estatística que leu todos os livros do mundo, mas nunca saiu de uma sala escura. Ele sabe exatamente qual palavra deve vir a seguir, mas não tem a menor ideia do que as palavras significam na realidade física.",
#         "refusal": null,
#         "reasoning": null
#       }
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 24,
#     "completion_tokens": 431,
#     "total_tokens": 455,
#     "cost": 0,
#     "is_byok": false,
#     "prompt_tokens_details": {
#       "cached_tokens": 0,
#       "cache_write_tokens": 0,
#       "audio_tokens": 0,
#       "video_tokens": 0
#     },
#     "cost_details": {
#       "upstream_inference_cost": 0.000071835,
#       "upstream_inference_prompt_cost": 7.2E-7,
#       "upstream_inference_completions_cost": 0.000071115
#     },
#     "completion_tokens_details": {
#       "reasoning_tokens": 0,
#       "image_tokens": 0,
#       "audio_tokens": 0
#     }
#   }
# }