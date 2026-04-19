import "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
import { workerEvents } from "../events/constants.js";

console.log("Model training worker initialized");
let _globalCtx = {};
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
}

function createTrainingData(context) {
  const inputs = [];
  const labels = [];

  context.users.forEach((user) => {
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

  debugger;

  postMessage({
    type: workerEvents.trainingLog,
    epoch: 1,
    loss: 1,
    accuracy: 1,
  });

  setTimeout(() => {
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
  }, 1000);
}
function recommend(user, ctx) {
  console.log("will recommend for user:", user);
  // postMessage({
  //     type: workerEvents.recommend,
  //     user,
  //     recommendations: []
  // });
}

const handlers = {
  [workerEvents.trainModel]: trainModel,
  [workerEvents.recommend]: (d) => recommend(d.user, _globalCtx),
};

self.onmessage = (e) => {
  const { action, ...data } = e.data;
  if (handlers[action]) handlers[action](data);
};
