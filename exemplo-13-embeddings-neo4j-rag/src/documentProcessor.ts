/**
 * Langchain é a maior biblioteca de referência
 * para o processamento de AI com JavaScript.
 */

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { type TextSplitterConfig } from "./config.ts";

export class DocumentProcessor {
  private pdfPath: string;
  private textSplitterConfig: TextSplitterConfig;

  constructor(pdfPath: string, textSplitterConfig: TextSplitterConfig) {
    this.pdfPath = pdfPath;
    this.textSplitterConfig = textSplitterConfig;
  }

  async loadAndSplit() {
    const loader = new PDFLoader(this.pdfPath);
    const rawDocuments = await loader.load();
    console.log(`📄 Loaded ${rawDocuments.length} pages from PDF`);
    const splitter = new RecursiveCharacterTextSplitter(this.textSplitterConfig);

    const documents = await splitter.splitDocuments(rawDocuments);

    console.log(`✂️ Split into ${documents.length} chunks`);

    return documents.map((doc) => ({
      ...doc,
      /**
       * Apenas para controle interno para saber de onde veio
       * a informação.
       */
      metadata: {
        source: doc.metadata.source,
      },
    }));
  }
}

/**
 * OpenRouter é o orquestrador de AI.
 */
