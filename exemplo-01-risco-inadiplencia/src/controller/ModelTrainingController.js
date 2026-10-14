export class ModelController {
  #modelView;
  #events;
  #alreadyTrained = false;
  constructor({ modelView, events }) {
    this.#modelView = modelView;
    this.#events = events;

    this.init();
  }

  static init(deps) {
    return new ModelController(deps);
  }

  async init() {
    this.setupCallbacks();
  }

  setupCallbacks() {
    this.#modelView.registerTrainModelCallback(this.handleTrainModel.bind(this));
    this.#modelView.registerCalculateRiskCallback(this.handleCalculateRisk.bind(this));

    this.#events.onTrainingComplete(() => {
      this.#alreadyTrained = true;
      this.#modelView.setModelReady(true);
      this.#modelView.updateTrainingProgress({ progress: 100 });
      this.handleCalculateRisk();
    });

    this.#events.onTrainingDataReady(({ samples, summary }) => {
      this.#modelView.renderTrainingData(samples, summary);
    });

    this.#events.onRiskCalculated((result) => {
      this.#modelView.renderRiskResult(result);
    });

    this.#events.onProgressUpdate((progress) => {
      this.handleTrainingProgressUpdate(progress);
    });
  }

  async handleTrainModel() {
    this.#alreadyTrained = false;
    this.#modelView.setModelReady(false);
    this.#events.dispatchTrainModel();
  }

  handleTrainingProgressUpdate(progress) {
    this.#modelView.updateTrainingProgress(progress);
  }

  handleCalculateRisk() {
    if (!this.#alreadyTrained) {
      return;
    }

    this.#events.dispatchCalculateRisk(this.#modelView.getRiskProfile());
  }
}
