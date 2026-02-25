import "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
import { workerEvents } from "../events/constants.js";
let _globalCtx = {};
let _model = null;

const WEIGHTS = {
  category: 0.5,
  price: 0.3,
  age: 0.2,
};

// 🔢 Normalize continuous values (price, age) to 0–1 range
// Why? Keeps all features balanced so no one dominates training
// Formula: (val - min) / (max - min)
// Example: price=79.90, minPrice=29.90, maxPrice=149.90 → 0.42
const normalize = (value, min, max) => (value - min) / (max - min || 1);

function makeContext(products, users) {
  const ages = users.map((u) => u.age);
  const prices = products.map((p) => p.price);

  const minAge = Math.min(...ages);
  const maxAge = Math.max(...ages);

  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);

  const categories = [...new Set(products.map((p) => p.category))];

  const categoriesIndex = Object.fromEntries(
    categories.map((category, index) => {
      return [category, index];
    }),
  );

  // Computar a média de idade dos assinantes por plano
  // (ajuda a personalizar)
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
      const avg = ageCounts[product.name]
        ? ageSums[product.name] / ageCounts[product.name]
        : midAge;

      return [product.name, normalize(avg, minAge, maxAge)];
    }),
  );

  return {
    products,
    users,
    categoriesIndex,
    productAvgAgeNorm,
    minAge,
    maxAge,
    minPrice,
    maxPrice,
    numCategories: categories.length,
    // price + age + categories
    dimentions: 2 + categories.length,
  };
}

const oneHotWeighted = (index, length, weight) =>
  tf.oneHot(index, length).cast("float32").mul(weight);

function encodeProduct(product, context) {
  // normalizando dados para ficar de 0 a 1 e
  // aplicar o peso na recomendação
  const price = tf.tensor1d([
    normalize(product.price, context.minPrice, context.maxPrice) *
      WEIGHTS.price,
  ]);

  const age = tf.tensor1d([
    (context.productAvgAgeNorm[product.name] ?? 0.5) * WEIGHTS.age,
  ]);

  const category = oneHotWeighted(
    context.categoriesIndex[product.category],
    context.numCategories,
    WEIGHTS.category,
  );

  return tf.concat1d([price, age, category]);
}

function encodeUser(user, context) {
  if (user.purchases.length) {
    return tf
      .stack(user.purchases.map((product) => encodeProduct(product, context)))
      .mean(0)
      .reshape([1, context.dimentions]);
  }

  return tf
    .concat1d([
      tf.zeros([1]), // preço é ignorado,
      tf.tensor1d([
        normalize(user.age, context.minAge, context.maxAge) * WEIGHTS.age,
      ]),
      tf.zeros([context.numCategories]), // categoria ignorada,
    ])
    .reshape([1, context.dimentions]);
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

        const label = user.purchases.some((purchase) =>
          purchase.name === product.name ? 1 : 0,
        );
        // combinar user + product
        inputs.push([...userVector, ...productVector]);
        labels.push(label);
      });
    });

  return {
    xs: tf.tensor2d(inputs),
    ys: tf.tensor2d(labels, [labels.length, 1]),
    inputDimention: context.dimentions * 2,
    // tamanho = userVector + productVector
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
        { id: 1, name: 'Fibra 100 Mega', category: 'fibra', price: 79.90 },
        { id: 4, name: 'Chip 5G Ilimitado', category: 'movel', price: 29.90 }
    ]
};
*/

// ====================================================================
// 📌 Após a codificação, o modelo NÃO vê nomes ou palavras.
// Ele vê um VETOR NUMÉRICO (todos normalizados entre 0–1).
// Exemplo: [preço_normalizado, idade_normalizada, cat_one_hot...]
//
// Suponha categorias = ['fibra', 'movel']
//
// Para Rafael (idade 27, categoria: fibra + movel),
// o vetor poderia ficar assim:
//
// [
//   0.35,            // peso do preço normalizado
//   0.60,            // idade normalizada
//   1, 0             // one-hot de categoria (fibra = ativo)
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
  model.add(
    tf.layers.dense({
      inputShape: [trainData.inputDimention],
      units: 128,
      activation: "relu",
    }),
  );
  // Camada oculta 1
  model.add(
    tf.layers.dense({
      units: 64,
      activation: "relu",
    }),
  );

  // Camada oculta 2
  model.add(
    tf.layers.dense({
      units: 32,
      activation: "relu",
    }),
  );
  // Camada de saída
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  await model.fit(trainData.xs, trainData.ys, {
    epochs: 100,
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
  postMessage({ type: workerEvents.progressUpdate, progress: { progress: 1 } });
  const products = await (await fetch("/data/products.json")).json();

  const context = makeContext(products, users);
  context.productVectors = products.map((product) => {
    return {
      name: product.name,
      meta: { ...product },
      vector: encodeProduct(product, context).dataSync(),
    };
  });

  _globalCtx = context;

  const trainData = createTrainingData(context);
  _model = await configureNeuralNetAndTrain(trainData);

  postMessage({
    type: workerEvents.progressUpdate,
    progress: { progress: 100 },
  });
  postMessage({ type: workerEvents.trainingComplete });
}
function recommend({ user }) {
  if (!_model) return;
  const context = _globalCtx;
  // 1️⃣ Converta o usuário fornecido no vetor de features codificadas

  const userVector = encodeUser(user, context).dataSync();

  // 2️⃣ Crie pares de entrada: para cada plano, concatene o vetor do usuário
  //    com o vetor codificado do plano.

  const inputs = context.productVectors.map(({ vector }) => {
    return [...userVector, ...vector];
  });

  // 3️⃣ Converta todos esses pares (usuário, plano) em um único Tensor.
  const inputTensor = tf.tensor2d(inputs);

  // 4️⃣ Rode a rede neural treinada em todos os pares (usuário, plano) de uma vez.
  const predictions = _model.predict(inputTensor);

  // 5️⃣ Extraia as pontuações para um array JS normal.
  const scores = predictions.dataSync();
  const recommendations = context.productVectors.map((item, index) => {
    return {
      ...item.meta,
      name: item.name,
      score: scores[index],
    };
  });

  const sortedItems = recommendations.sort((a, b) => b.score - a.score);

  // 6️⃣ Envie a lista ordenada de planos recomendados
  //    para a thread principal (a UI pode exibi-los agora).
  postMessage({
    type: workerEvents.recommend,
    user,
    recommendations: sortedItems,
  });
}
const handlers = {
  [workerEvents.trainModel]: trainModel,
  [workerEvents.recommend]: recommend,
};

self.onmessage = (e) => {
  const { action, ...data } = e.data;
  if (handlers[action]) handlers[action](data);
};
