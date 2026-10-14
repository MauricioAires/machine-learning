import { workerEvents } from "../events/constants.js";

export class WorkerController {
  #worker;
  #events;
  #alreadyTrained = false;
  constructor({ worker, events }) {
    this.#worker = worker;
    this.#events = events;
    this.#alreadyTrained = false;
    this.init();
  }

  async init() {
    this.setupCallbacks();
  }

  static init(deps) {
    return new WorkerController(deps);
  }

  setupCallbacks() {
    this.#events.onTrainModel((data) => {
      this.#alreadyTrained = false;
      this.triggerTrain(data);
    });

    this.#events.onCalculateRisk((data) => {
      if (!this.#alreadyTrained) return;

      this.triggerCalculateRisk(data);
    });

    this.#events.onTrainingComplete(() => {
      this.#alreadyTrained = true;
    });

    const eventsToIgnoreLogs = [workerEvents.progressUpdate, workerEvents.trainingLog, workerEvents.tfVisLogs, workerEvents.trainingComplete, workerEvents.trainingDataReady, workerEvents.riskCalculated];
    this.#worker.onmessage = (event) => {
      if (!eventsToIgnoreLogs.includes(event.data.type)) console.log(event.data);

      if (event.data.type === workerEvents.progressUpdate) {
        this.#events.dispatchProgressUpdate(event.data.progress);
      }

      if (event.data.type === workerEvents.trainingComplete) {
        this.#events.dispatchTrainingComplete(event.data);
      }

      if (event.data.type === workerEvents.trainingDataReady) {
        this.#events.dispatchTrainingDataReady(event.data);
      }

      if (event.data.type === workerEvents.trainingLog) {
        this.#events.dispatchTFVisLogs(event.data);
      }

      if (event.data.type === workerEvents.riskCalculated) {
        this.#events.dispatchRiskCalculated(event.data);
      }
    };
  }

  triggerTrain() {
    this.#worker.postMessage({ action: workerEvents.trainModel });
  }

  triggerCalculateRisk(profile) {
    this.#worker.postMessage({ action: workerEvents.calculateRisk, profile });
  }
}
