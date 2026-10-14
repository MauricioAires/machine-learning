import { events } from "./constants.js";

export default class Events {
  static onTrainingComplete(callback) {
    document.addEventListener(events.trainingComplete, (event) => {
      return callback(event.detail);
    });
  }
  static dispatchTrainingComplete(data) {
    const event = new CustomEvent(events.trainingComplete, {
      detail: data,
    });
    document.dispatchEvent(event);
  }

  static onTrainingDataReady(callback) {
    document.addEventListener(events.trainingDataReady, (event) => {
      return callback(event.detail);
    });
  }
  static dispatchTrainingDataReady(data) {
    const event = new CustomEvent(events.trainingDataReady, {
      detail: data,
    });
    document.dispatchEvent(event);
  }

  static onRiskCalculated(callback) {
    document.addEventListener(events.riskCalculated, (event) => {
      return callback(event.detail);
    });
  }
  static dispatchRiskCalculated(data) {
    const event = new CustomEvent(events.riskCalculated, {
      detail: data,
    });
    document.dispatchEvent(event);
  }

  static onCalculateRisk(callback) {
    document.addEventListener(events.riskRequested, (event) => {
      return callback(event.detail);
    });
  }

  static dispatchCalculateRisk(data) {
    const event = new CustomEvent(events.riskRequested, {
      detail: data,
    });
    document.dispatchEvent(event);
  }

  static onTrainModel(callback) {
    document.addEventListener(events.modelTrain, (event) => {
      return callback(event.detail);
    });
  }
  static dispatchTrainModel(data) {
    const event = new CustomEvent(events.modelTrain, {
      detail: data,
    });
    document.dispatchEvent(event);
  }

  static onTFVisLogs(callback) {
    document.addEventListener(events.tfvisLogs, (event) => {
      return callback(event.detail);
    });
  }

  static dispatchTFVisLogs(data) {
    const event = new CustomEvent(events.tfvisLogs, {
      detail: data,
    });
    document.dispatchEvent(event);
  }

  static onTFVisorData(callback) {
    document.addEventListener(events.tfvisData, (event) => {
      return callback(event.detail);
    });
  }
  static dispatchTFVisorData(data) {
    const event = new CustomEvent(events.tfvisData, {
      detail: data,
    });
    document.dispatchEvent(event);
  }

  static onProgressUpdate(callback) {
    document.addEventListener(events.modelProgressUpdate, (event) => {
      return callback(event.detail);
    });
  }

  static dispatchProgressUpdate(progressData) {
    const event = new CustomEvent(events.modelProgressUpdate, {
      detail: progressData,
    });
    document.dispatchEvent(event);
  }
}
