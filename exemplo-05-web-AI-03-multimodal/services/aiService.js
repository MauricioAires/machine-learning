export class AIService {
  constructor() {
    this.session = null;
    this.abortController = null;
  }

  async checkRequirements() {
    const errors = [];

    // @ts-ignore
    const isChrome = !!window.chrome;
    if (!isChrome) {
      errors.push("⚠️ Este recurso só funciona no Google Chrome ou Chrome Canary (versão recente).");
    }

    if (!("LanguageModel" in self)) {
      errors.push("⚠️ As APIs nativas de IA não estão ativas.");
      errors.push("Ative a seguinte flag em chrome://flags/:");
      errors.push("- Prompt API for Gemini Nano (chrome://flags/#prompt-api-for-gemini-nano)");
      errors.push("Depois reinicie o Chrome e tente novamente.");
      return errors;
    }

    // Check Translator availability
    if ("Translator" in self) {
      const translatorAvailability = await Translator.availability({
        sourceLanguage: "en",
        targetLanguage: "pt",
      });
      console.log("Translator Availability:", translatorAvailability);

      if (translatorAvailability === "no") {
        errors.push("⚠️ Tradução de inglês para português não está disponível.");
      }

      if (translatorAvailability === "downloadable") {
        const btnDownloadTranslator = window.document.getElementById("downloadTranslator");

        btnDownloadTranslator.style = "display: block";
      }
    } else {
      errors.push("⚠️ A API de Tradução não está ativa.");
      errors.push("Ative a seguinte flag em chrome://flags/:");
      errors.push("- Translation API (chrome://flags/#translation-api)");
    }

    // Check Language Detection API
    if (!("LanguageDetector" in self)) {
      errors.push("⚠️ A API de Detecção de Idioma não está ativa.");
      errors.push("Ative a seguinte flag em chrome://flags/:");
      errors.push("- Language Detection API (chrome://flags/#language-detector-api)");
    }

    if (errors.length > 0) {
      return errors;
    }

    const availability = await LanguageModel.availability({ languages: ["en"] });
    console.log("Language Model Availability:", availability);

    if (availability === "available") {
      return null;
    }

    if (availability === "unavailable") {
      errors.push(`⚠️ O seu dispositivo não suporta modelos de linguagem nativos de IA.`);
    }

    if (availability === "downloading") {
      errors.push(`⚠️ O modelo de linguagem de IA está sendo baixado. Por favor, aguarde alguns minutos e tente novamente.`);
    }

    if (availability === "downloadable") {
      errors.push(`⚠️ O modelo de linguagem de IA precisa ser baixado, baixando agora... (acompanhe o progresso no terminal do chrome)`);
      try {
        const session = await LanguageModel.create({
          expectedInputLanguages: ["en"],
          monitor(m) {
            m.addEventListener("downloadprogress", (e) => {
              const percent = ((e.loaded / e.total) * 100).toFixed(0);
              console.log(`Downloaded ${percent}%`);
            });
          },
        });
        await session.prompt("Hello");
        session.destroy();

        // Re-check availability after download
        const newAvailability = await LanguageModel.availability({ languages: ["en"] });
        if (newAvailability === "available") {
          return null; // Download successful
        }
      } catch (error) {
        console.error("Error downloading model:", error);
        errors.push(`⚠️ Erro ao baixar o modelo: ${error.message}`);
      }
    }

    return errors.length > 0 ? errors : null;
  }

  async getParams() {
    let params = {
      defaultTemperature: 1,
      defaultTopK: 3,
      maxTemperature: 2,
      maxTopK: 128,
    };

    if (LanguageModel.params) {
      params = await LanguageModel.params();
      console.log("Language Model Params:", params);
    }

    return params;
  }

  async *createSession(question, temperature, topK, file = null) {
    this.abortController?.abort();
    this.abortController = new AbortController();

    // Destroy previous session and create new one with updated parameters
    if (this.session) {
      this.session.destroy();
    }

    /**
     * @TODO MauricioAires
     *
     * Multimodal significa que aceita mais de um tipo de formato
     * nesse caso texto audio e imagem
     *
     * Não tem a opção de documento se tivesse dava para fazer algo
     * como ler documento e preencher form.
     */
    this.session = await LanguageModel.create({
      expectedInputs: [{ type: "text", languages: ["en"] }, { type: "audio" }, { type: "image" }],
      expectedOutputs: [{ type: "text", languages: ["en"] }],
      temperature: temperature,
      topK: topK,
      initialPrompts: [
        {
          role: "system",
          content: [
            {
              type: "text",
              value: `You are an AI assistant that responds clearly and objectively.
                        Always respond in plain text format instead of markdown.`,
            },
          ],
        },
      ],
    });

    // Build content array with text and optional file
    const contentArray = [{ type: "text", value: question }];

    if (file) {
      const fileType = file.type.split("/")[0];

      if (fileType === "image") {
        try {
          console.log("Convertendo imagem para ImageBitmap...");

          const imageBitmap = await window.createImageBitmap(file);

          contentArray.push({
            type: "image",
            value: imageBitmap,
          });
          console.log(`Imagem anexada com sucesso ao prompt:`, file.name);
        } catch (imgError) {
          console.error("Erro ao processar a imagem:", imgError);
        }
      } else if (fileType === "audio") {
        // Áudio pode ter regras diferentes, mas para imagem o ImageBitmap é a chave
        const blob = new Blob([await file.arrayBuffer()], { type: file.type });
        contentArray.push({ type: "audio", value: blob });
      }
    }

    const responseStream = await this.session.promptStreaming(
      [
        {
          role: "user",
          content: contentArray,
        },
      ],
      {
        signal: this.abortController.signal,
      }
    );

    for await (const chunk of responseStream) {
      if (this.abortController.signal.aborted) {
        break;
      }
      yield chunk;
    }
  }

  abort() {
    this.abortController?.abort();
  }

  isAborted() {
    return this.abortController?.signal.aborted;
  }
}
