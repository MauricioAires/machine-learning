import "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
import { workerEvents } from "../events/constants.js";

console.log("Credit risk worker initialized");

let _model = null;
let _context = null;

const FEATURE_RANGES = {
  age: { min: 18, max: 70 },
  monthlyIncome: { min: 1500, max: 25000 },
  creditLimit: { min: 800, max: 50000 },
  usedAmount: { min: 0, max: 50000 },
  utilizationRate: { min: 0, max: 100 },
  delinquencyCount: { min: 0, max: 8 },
  openInstallments: { min: 0, max: 12 },
  relationshipYears: { min: 0, max: 25 },
  recentInquiries: { min: 0, max: 10 },
  activeCards: { min: 1, max: 8 },
};

const clamp = (value, min = 0, max = 1) => Math.min(max, Math.max(min, value));
const normalize = (value, min, max) => (value - min) / (max - min || 1);
const randomBetween = (min, max) => min + Math.random() * (max - min);
const randomInt = (min, max) => Math.floor(randomBetween(min, max + 1));
const round = (value, digits = 2) => Number(value.toFixed(digits));

function getRiskLevel(score) {
  if (score >= 75) {
    return {
      label: "Risco alto",
      className: "text-bg-danger",
      description: "Perfil com forte probabilidade de inadimplência.",
    };
  }

  if (score >= 45) {
    return {
      label: "Risco moderado",
      className: "text-bg-warning",
      description: "Perfil com sinais de atenção e acompanhamento.",
    };
  }

  return {
    label: "Risco baixo",
    className: "text-bg-success",
    description: "Perfil com baixa propensão à inadimplência.",
  };
}

function labelSample(riskScore) {
  if (riskScore >= 0.58) {
    return { label: 1, labelText: "inadimplente" };
  }

  return { label: 0, labelText: "adimplente" };
}

function generateSyntheticSamples(sampleCount = 36) {
  const samples = [];

  for (let index = 1; index <= sampleCount; index += 1) {
    const age = randomInt(FEATURE_RANGES.age.min, FEATURE_RANGES.age.max);
    const monthlyIncome = round(randomBetween(FEATURE_RANGES.monthlyIncome.min, FEATURE_RANGES.monthlyIncome.max), 2);
    const creditLimit = round(randomBetween(FEATURE_RANGES.creditLimit.min, FEATURE_RANGES.creditLimit.max), 2);
    const utilizationRate = round(randomBetween(FEATURE_RANGES.utilizationRate.min, FEATURE_RANGES.utilizationRate.max), 1);
    const usedAmount = round((creditLimit * utilizationRate) / 100, 2);
    const delinquencyCount = randomInt(FEATURE_RANGES.delinquencyCount.min, FEATURE_RANGES.delinquencyCount.max);
    const openInstallments = randomInt(FEATURE_RANGES.openInstallments.min, FEATURE_RANGES.openInstallments.max);
    const relationshipYears = round(randomBetween(FEATURE_RANGES.relationshipYears.min, FEATURE_RANGES.relationshipYears.max), 1);
    const recentInquiries = randomInt(FEATURE_RANGES.recentInquiries.min, FEATURE_RANGES.recentInquiries.max);
    const activeCards = randomInt(FEATURE_RANGES.activeCards.min, FEATURE_RANGES.activeCards.max);

    const riskScore = clamp(
      0.22 * (1 - normalize(monthlyIncome, FEATURE_RANGES.monthlyIncome.min, FEATURE_RANGES.monthlyIncome.max)) +
        0.22 * normalize(utilizationRate, FEATURE_RANGES.utilizationRate.min, FEATURE_RANGES.utilizationRate.max) +
        0.18 * normalize(delinquencyCount, FEATURE_RANGES.delinquencyCount.min, FEATURE_RANGES.delinquencyCount.max) +
        0.12 * normalize(openInstallments, FEATURE_RANGES.openInstallments.min, FEATURE_RANGES.openInstallments.max) +
        0.1 * normalize(recentInquiries, FEATURE_RANGES.recentInquiries.min, FEATURE_RANGES.recentInquiries.max) +
        0.08 * (1 - normalize(relationshipYears, FEATURE_RANGES.relationshipYears.min, FEATURE_RANGES.relationshipYears.max)) +
        0.04 * normalize(activeCards, FEATURE_RANGES.activeCards.min, FEATURE_RANGES.activeCards.max) +
        0.04 * (1 - normalize(age, FEATURE_RANGES.age.min, FEATURE_RANGES.age.max)),
      0,
      1,
    );

    const riskPercent = round(riskScore * 100, 1);
    const { label, labelText } = labelSample(riskScore);

    samples.push({
      id: index,
      age,
      monthlyIncome,
      creditLimit,
      usedAmount,
      utilizationRate,
      delinquencyCount,
      openInstallments,
      relationshipYears,
      recentInquiries,
      activeCards,
      riskScore: riskPercent,
      label,
      labelText,
    });
  }

  return samples;
}

function buildContext(samples) {
  return {
    ranges: Object.fromEntries(
      Object.keys(FEATURE_RANGES).map((key) => {
        const values = samples.map((sample) => sample[key]);
        return [
          key,
          {
            min: Math.min(FEATURE_RANGES[key].min, ...values),
            max: Math.max(FEATURE_RANGES[key].max, ...values),
          },
        ];
      }),
    ),
    featureKeys: ["age", "monthlyIncome", "creditLimit", "usedAmount", "utilizationRate", "delinquencyCount", "openInstallments", "relationshipYears", "recentInquiries", "activeCards"],
  };
}

function encodeProfile(profile, context) {
  return tf.tensor1d(
    context.featureKeys.map((key) => {
      const range = context.ranges[key];
      return normalize(profile[key], range.min, range.max);
    }),
  );
}

function createTrainingData(samples, context) {
  const inputs = samples.map((sample) => encodeProfile(sample, context).dataSync());
  const labels = samples.map((sample) => sample.label);

  return {
    xs: tf.tensor2d(inputs),
    ys: tf.tensor2d(labels, [labels.length, 1]),
    inputDimensions: context.featureKeys.length,
  };
}

function buildSummary(samples) {
  const total = samples.length;
  const bad = samples.filter((sample) => sample.label === 1).length;

  return {
    total,
    bad,
    good: total - bad,
    badRate: round((bad / (total || 1)) * 100, 1),
  };
}

async function configureNeuralNetAndTrain(trainData) {
  const model = tf.sequential();

  model.add(tf.layers.dense({ inputShape: [trainData.inputDimensions], units: 32, activation: "relu" }));
  model.add(tf.layers.dense({ units: 16, activation: "relu" }));
  model.add(tf.layers.dense({ units: 8, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  await model.fit(trainData.xs, trainData.ys, {
    epochs: 80,
    batchSize: 8,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        postMessage({
          type: workerEvents.trainingLog,
          epoch,
          loss: logs.loss,
          accuracy: logs.acc,
        });
      },
    },
  });

  return model;
}

async function trainModel() {
  postMessage({ type: workerEvents.progressUpdate, progress: { progress: 10 } });

  const samples = generateSyntheticSamples();
  const context = buildContext(samples);
  const summary = buildSummary(samples);

  _context = context;

  postMessage({
    type: workerEvents.trainingDataReady,
    samples,
    summary,
  });

  postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });

  const trainData = createTrainingData(samples, context);
  _model = await configureNeuralNetAndTrain(trainData);

  postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
  postMessage({ type: workerEvents.trainingComplete, summary });
}

function calculateRisk({ profile }) {
  if (!_model || !_context) return;

  const input = encodeProfile(profile, _context).reshape([1, _context.featureKeys.length]);

  const score = tf.tidy(() => {
    const prediction = _model.predict(input);
    return prediction.dataSync()[0];
  });

  const riskPercent = round(score * 100, 1);

  postMessage({
    type: workerEvents.riskCalculated,
    score: riskPercent,
    level: getRiskLevel(riskPercent),
    profile,
  });
}

const handlers = {
  [workerEvents.trainModel]: trainModel,
  [workerEvents.calculateRisk]: calculateRisk,
};

self.onmessage = (event) => {
  const { action, ...data } = event.data;

  if (handlers[action]) {
    handlers[action](data);
  }
};
