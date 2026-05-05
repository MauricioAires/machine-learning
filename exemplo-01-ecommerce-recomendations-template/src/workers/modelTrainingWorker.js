import "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
import { workerEvents } from "../events/constants.js";

console.log("Model training worker initialized");
let _globalCtx = {};
let _model = null;

const WEIGHTS = {
  category: 0.4,
  color: 0.3,
  price: 0.2,
  age: 0.1,
};

/**
 * Normalize continuos values (price, age) to 0-1 range
 * Why? Keeps all features balanced so no one dominates training
 * Formula: (val - min) / (max - min)
 * Example: price=129.99, minPrice=39.99, maxPrices=199.99 -> 0.56
 */
const normalize = (val, min, max) => (val - min) / (max - min || 1);

function makeContext(products, users) {
  const ages = users.map((user) => user.age);
  const prices = products.map((p) => p.price);

  const minAge = Math.min(...ages);
  const maxAge = Math.max(...ages);

  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);

  // New Set para garantir que não tenha itens duplicados.
  const colors = [...new Set(products.map((p) => p.color))];
  const categories = [...new Set(products.map((p) => p.category))];

  const colorsIndex = Object.fromEntries(colors.map((color, index) => [color, index]));
  const categoriesIndex = Object.fromEntries(categories.map((category, index) => [category, index]));

  /**
   * Computar a média de idade dos compradores por produto
   * (ajuda a personalizar)
   */
  const midAge = (minAge + maxAge) / 2;
  const ageSums = {};
  const ageCounts = {};

  users.forEach((user) => {
    user.purchases.forEach((p) => {
      ageSums[p.name] = (ageSums[p.name] || 0) + user.age;
      ageCounts[p.name] = (ageCounts[p.name] || 0) + 1;
    });
  });

  const productAvgAgeNorm = Object.fromEntries(
    products.map((product) => {
      /**
       * Se não tiver um registro de idade para um produto, usar a idade média geral (midAge) como fallback.
       */
      const avg = ageCounts[product.name] ? ageSums[product.name] / ageCounts[product.name] : midAge;

      return [product.name, normalize(avg, minAge, maxAge)];
    })
  );

  return {
    products,
    users,
    colorsIndex,
    categoriesIndex,
    productAvgAgeNorm,
    minAge,
    maxAge,
    minPrice,
    maxPrice,
    numCategories: categories.length,
    numColors: colors.length,
    /**
     * Porque 2? Porque vamos usar idade e preço como features numéricas normalizadas.
     * + cada uma das categorias
     * + cada uma das cores
     */
    dimensions: 2 + categories.length + colors.length,
  };
}

const oneHotWeighted = (index, length, weight) => tf.oneHot(index, length).cast("float32").mul(weight);

function encodeProduct(product, context) {
  /**
   * Normalizando dados para ficar de 0 a 1 e aplicando pesos ma
   * recomendação
   */
  const price = tf.tensor1d([normalize(product.price, context.minPrice, context.maxPrice) * WEIGHTS.price]);
  const age = tf.tensor1d([(context.productAvgAgeNorm[product.name] ?? 0.5) * WEIGHTS.age]);
  const category = oneHotWeighted(context.categoriesIndex[product.category], context.numCategories, WEIGHTS.category);
  const color = oneHotWeighted(context.colorsIndex[product.color], context.numColors, WEIGHTS.color);

  return tf.concat([price, age, category, color]);
}

/**
 * Retornar o perfil de compra do usuário.
 */

function encodeUser(user, context) {
  /**
   * Tensor para todos os usuários que possui compras.
   */
  if (user.purchases.length) {
    // Cada linha que tiver vai ser uma compra do usuário
    return tf
      .stack(user.purchases.map((purchase) => encodeProduct(purchase, context)))
      .mean(0) // Média dos vetores para retornar apenas um valor
      .reshape([
        1, // um linha no shape
        context.dimensions, // colunas
      ]);
  }

  /**
   * Para usuários sem compras, criamos um vetor neutro (zeros) para preço, categoria e cor,
   * onde a única informação que será utilizada para recomendar será a idade mas em um sistema
   * real poderíamos utilizar outras features como localização, gênero, etc. para ajudar a personalizar as recomendações mesmo
   */
  return tf
    .concat1d([
      tf.zeros([1]), // preço é ignorado,
      tf.tensor1d([normalize(user.age, context.minAge, context.maxAge) * WEIGHTS.age]),
      tf.zeros([context.numCategories]), // categoria ignorada,
      tf.zeros([context.numColors]), // color ignorada,
    ])
    .reshape([1, context.dimensions]);
}

function createTrainingData(context) {
  const inputs = [];
  const labels = [];

  context.users
    .filter((u) => u.purchases.length)
    .forEach((user) => {
      const userVector = encodeUser(user, context).dataSync();
      context.products.forEach((product) => {
        const productVector = encodeProduct(product, context).dataSync();
        // 1 se comprou, 0 caso contrário
        const label = user.purchases.some((purchase) => purchase.name === product.name) ? 1 : 0;

        // Combinar usuário + product
        inputs.push([...userVector, ...productVector]);
        labels.push(label);
      });
    });

  return {
    xs: tf.tensor2d(inputs), // Input
    ys: tf.tensor2d(labels, [
      labels.length, // Tamanho das dimensões
      1,
    ]), // Como ele deve
    inputDimensions: context.dimensions * 2, // Porque estamos combinando usuário + produto
  };
}

// ====================================================================
// 📌 Exemplo de como um usuário é ANTES da codificação
// ====================================================================
/*
const exampleUser = {
    id: 201,
    name: 'Rafael Souza',
    age: 27,
    purchases: [
        { id: 8, name: 'Boné Estiloso', category: 'acessórios', price: 39.99, color: 'preto' },
        { id: 9, name: 'Mochila Executiva', category: 'acessórios', price: 159.99, color: 'cinza' }
    ]
};
*/

// ====================================================================
// 📌 Após a codificação, o modelo NÃO vê nomes ou palavras.
// Ele vê um VETOR NUMÉRICO (todos normalizados entre 0–1).
// Exemplo: [preço_normalizado, idade_normalizada, cat_one_hot..., cor_one_hot...]
//
// Suponha categorias = ['acessórios', 'eletrônicos', 'vestuário']
// Suponha cores      = ['preto', 'cinza', 'azul']
//
// Para Rafael (idade 27, categoria: acessórios, cores: preto/cinza),
// o vetor poderia ficar assim:
//
// [
//   0.45,            // peso do preço normalizado
//   0.60,            // idade normalizada
//   1, 0, 0,         // one-hot de categoria (acessórios = ativo)
//   1, 0, 0          // one-hot de cores (preto e cinza ativos, azul inativo)
// ]
//
// São esses números que vão para a rede neural.
// ====================================================================

// ====================================================================
// 🧠 Configuração e treinamento da rede neural
// ====================================================================
async function configureNeuralNetAndTrain(trainData) {
  const model = tf.sequential();

  // Camada de entrada
  // - inputShape: Número de features por exemplo de treino (trainData.inputDim)
  //   Exemplo: Se o vetor produto + usuário = 20 números, então inputDim = 20
  // - units: 128 neurônios (muitos "olhos" para detectar padrões)
  // - activation: 'relu' (mantém apenas sinais positivos, ajuda a aprender padrões não-lineares)
  model.add(
    tf.layers.dense({
      inputShape: [trainData.inputDimensions],
      // Neurônios quanto menos dados tiver para treinar mas neurônios é necessário para eles prever alguma coisa.
      units: 128, // Entender a base de dados, entender os padrões e retorna os valores
      activation: "relu",
    })
  );

  // Camada oculta 1
  // - 64 neurônios (menos que a primeira camada: começa a comprimir informação)
  // - activation: 'relu' (ainda extraindo combinações relevantes de features)
  model.add(
    tf.layers.dense({
      units: 64, // Trabalha com os dados retornar pela camada anterior e tenta entender padrões mais complexos
    })
  );

  // Camada oculta 2
  // - 32 neurônios (mais estreita de novo, destilando as informações mais importantes)
  //   Exemplo: De muitos sinais, mantém apenas os padrões mais fortes
  // - activation: 'relu'
  model.add(
    tf.layers.dense({
      units: 32, // Treina com menos dados para entender padrões mais complexos
    })
  );

  // Camada de saída
  // - 1 neurônio porque vamos retornar apenas uma pontuação de recomendação
  // - activation: 'sigmoid' comprime o resultado para o intervalo 0–1
  //   Exemplo: 0.9 = recomendação forte, 0.1 = recomendação fraca
  model.add(
    tf.layers.dense({
      units: 1,
      activation: "sigmoid", // Ativação sigmoid para saída entre 0 e 1 (probabilidade de compra)
    })
  );

  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  await model.fit(trainData.xs, trainData.ys, {
    epochs: 100, // Quantidade de vezes que o modelo vai ver o mesmo dado (quanto mais, melhor ele aprende, mas cuidado com overfitting)
    batchSize: 32,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        postMessage({
          type: workerEvents.trainingLog,
          epoch: epoch,
          loss: logs.loss,
          accuracy: logs.acc,
        });
      },
    },
  });

  return model;
}

async function trainModel({ users }) {
  console.log("Training model with users:", users);

  postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });

  const products = await (await fetch("/data/products.json")).json();

  const context = makeContext(products, users);

  /**
   * DICA: para grandes projetos os vetores devem
   * ser armazenados em bancos de dados para ser apenas consumidos
   * e não processar do lado do cliente onde pode causar lentidão
   * com o processamento.
   */
  context.productVectors = products.map((product) => {
    return {
      name: product.name,
      meta: {
        ...product,
      },
      vector: encodeProduct(product, context).dataSync(),
    };
  });

  /***
   * OBS:. Os dados estão sendo armazenados de forma global apenas para serem
   * acessados posteriormente mas em um caso real esses dados veriam do banco de
   * dados.
   */
  _globalCtx = context;

  const trainData = createTrainingData(context);
  _model = await configureNeuralNetAndTrain(trainData);

  postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
  postMessage({ type: workerEvents.trainingComplete });
}

function recommend(user, ctx) {
  if (!_model) return;
  const context = _globalCtx;
  // 1️⃣ Converta o usuário fornecido no vetor de features codificadas
  //    (preço ignorado, idade normalizada, categorias ignoradas)
  //    Isso transforma as informações do usuário no mesmo formato numérico
  //    que foi usado para treinar o modelo.
  const userVector = encodeUser(user, _globalCtx).dataSync();
  // Em aplicações reais:
  //  Armazene todos os vetores de produtos em um banco de dados vetorial (como Postgres, Neo4j ou Pinecone)
  //  Consulta: Encontre os 200 produtos mais próximos do vetor do usuário
  //  Execute _model.predict() apenas nesses produtos

  // 2️⃣ Crie pares de entrada: para cada produto, concatene o vetor do usuário
  //    com o vetor codificado do produto.
  //    Por quê? O modelo prevê o "score de compatibilidade" para cada par (usuário, produto).
  const inputs = context.productVectors.map(({ vector }) => {
    return [...userVector, ...vector];
  });
  // 3️⃣ Converta todos esses pares (usuário, produto) em um único Tensor.
  //    Formato: [numProdutos, inputDim]
  const inputTensor = tf.tensor2d(inputs);
  // 4️⃣ Rode a rede neural treinada em todos os pares (usuário, produto) de uma vez.
  //    O resultado é uma pontuação para cada produto entre 0 e 1.
  //    Quanto maior, maior a probabilidade do usuário querer aquele produto.
  const predictions = _model.predict(inputTensor);
  // 5️⃣ Extraia as pontuações para um array JS normal.
  const scores = predictions.dataSync();

  const recommendations = context.productVectors.map((item, index) => {
    return {
      ...item.meta,
      name: item.name,
      score: scores[index], // previsão para este produto
    };
  });

  const sortedItems = recommendations.sort((a, b) => b.score - a.score);

  // 8️⃣ Envie a lista ordenada de produtos recomendados
  //    para a thread principal (a UI pode exibi-los agora).
  postMessage({
    type: workerEvents.recommend,
    user,
    recommendations: sortedItems,
  });
}

const handlers = {
  [workerEvents.trainModel]: trainModel,
  [workerEvents.recommend]: (d) => recommend(d.user, _globalCtx),
};

self.onmessage = (e) => {
  const { action, ...data } = e.data;
  if (handlers[action]) handlers[action](data);
};

/**
 * Recomendação
 *
 * Utilizar o banco de dados CromaDB [https://www.trychroma.com/]
 * para armazenar os vetores, um banco de dados especializado para isso, e utilizar o modelo apenas para gerar os vetores de usuário e produto.
 *
 * 1. Fazer uma request com 100 registros mais próximo do vetor e rodar
 * o comando _model.predict() apenas nesses registros para otimizar o processo.
 *
 * 2. Para não fazer dados fictícios, podemos utilizar dados reais de e-commerce como o dataset da Amazon disponível no Kaggle [https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews] ou o dataset de produtos do * Walmart [https://www.kaggle.com/datasets/c/walmart-recruiting-store-sales-forecasting]. Esses datasets possuem informações sobre produtos, usuários e suas interações, permitindo treinar um modelo de recomendação mais * realista.
 * */
