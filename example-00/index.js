import tf from "@tensorflow/tfjs-node";

async function trainModel(inputXs, outputYs) {
  const model = tf.sequential();

  // Primeira camada da rede
  // entrada de 7 posições (idade normalizada + 3 cores + 3 localizações)

  model.add(
    tf.layers.dense({
      inputShape: [7],
      // 80 neurônios => aqui coloquei tudo isso, pq tem pouca base de treinamento
      // quanto mais neurônios, mais complexidade a rede pode aprender
      // e consequentemente mais processamento ela vai usar
      units: 100, // quantidade de neurônios

      // A ReLU age como um filtro:
      // /e como se ela deixasse somente os dados interessantes seguirem viagem
      // na rede, se a informação chegou neurônio é positiva, passa para frente!
      // Se for zero ou negativa, pode jogar fora, não vi servir para nada.
      activation: "relu",
    })
  );

  // Saida: 3 neurônios
  // Um para cada categoria: premium, medium, basic
  model.add(
    tf.layers.dense({
      units: 3,
      // a função softmax é usada para classificação,
      // ela vai transformar os dados em probabilidades
      activation: "softmax",
    })
  );

  // Compilando o modelo
  model.compile({
    // Adam(Adaptive Moment Estimation)
    // é um treinador pessoal moderno para redes neurais:
    // ajusta os pesos de forma eficiente e inteligente
    // aprender com histórico de erros e acertos.
    optimizer: "adam",
    // loss:  ele compara o que o modelo "acha" (os scores de cada categoria)
    // com a resposta certa
    // a categoria premium será sempre [1,0,0]
    loss: "categoricalCrossentropy",
    /**
     * Quanto mais distante da previsão do modelo da resposta correta
     * maior o erro (loss)
     * Exemplo clássico: classificação de imagens, recomendação, categorização
     * de usuário.
     * Qualquer coisa em que a resposta certa é "apenas uma entre várias
     * possíveis"
     */
    metrics: ["accuracy"],
  });

  // Treinando o modelo

  await model.fit(inputXs, outputYs, {
    /**
     * Desabilitar os logs
     */
    verbose: 0,
    /**
     * Vai passar 100x por toda a base enviada
     */
    epochs: 100,
    /**
     * Embaralha, para evitar que o modelo aprenda a ordem
     * dos dados, e sim os padrões
     */
    shuffle: true,
    // callbacks: {
    //   onEpochEnd: (epoch, log) => {
    //     console.log(`Epoch: ${epoch}: loss = ${log.loss}`);
    //   },
    // },
  });

  return model;
}

async function predict(model, pessoa) {
  /**
   * Transformar o array js para tensor (tsjs)
   */

  const tfInput = tf.tensor2d(pessoa);
  /**
   * Faz a predição ( o output será um vetor de 3 probabilidades)
   */

  const pred = model.predict(tfInput);
  const predArray = await pred.array();

  return predArray[0].map((prob, index) => ({ prob, index }));
}

const tensorPessoasNormalizado = [
  [0.33, 1, 0, 0, 1, 0, 0], // Erick
  [0, 0, 1, 0, 0, 1, 0], // Ana
  [1, 0, 0, 1, 0, 0, 1], // Carlos
];

const labelsNomes = ["Premium", "Medium", "Basic"];
const tensorLabels = [
  [1, 0, 0], // Erick
  [0, 1, 0], // Ana
  [0, 0, 1], // Carlos
];

const inputXs = tf.tensor2d(tensorPessoasNormalizado);
const outputYs = tf.tensor2d(tensorLabels);

// inputXs.print();
// outputYs.print();

/**
 * Quanto mais dado melhor, assim o algoritmo consegue
 * entender melhor os padrões complexos dos dados.
 */
const model = await trainModel(inputXs, outputYs);

const pessoa = {
  nome: "zé",
  idade: 28,
  cor: "verde",
  localizacao: "Curitiba",
};

/**
 * Normalizando a idade da nova pessoa usando o mesmo padrão do treino
 * Exemplo: idade min = 25, idade_max = 40, então (28 - 25) / (40 - 25) = 0.2
 */

const pessoaTensorNormalizado = [
  [
    0.2, // idade normalizada
    1, // cor azul
    0, // cor vermelho
    0, // cor verde
    0, // localização São Paulo
    1, // localização Rio
    0, // localização Curitiba
  ],
];

const predictions = await predict(model, pessoaTensorNormalizado);

const results = predictions
  .sort((a, b) => b.prob - a.prob)
  .map((prob) => `- ${labelsNomes[[prob.index]]} (${(prob.prob * 100).toFixed(2)}%)`)
  .join("\n");

console.log(results);
