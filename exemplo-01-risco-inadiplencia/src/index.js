import { ModelController } from "./controller/ModelTrainingController.js";
import { TFVisorController } from "./controller/TFVisorController.js";
import { TFVisorView } from "./view/TFVisorView.js";
import { ModelView } from "./view/ModelTrainingView.js";
import Events from "./events/events.js";
import { WorkerController } from "./controller/WorkerController.js";

// Create views
const modelView = new ModelView();
const tfVisorView = new TFVisorView();
const mlWorker = new Worker("/src/workers/modelTrainingWorker.js", { type: "module" });

// Set up worker message handler
const w = WorkerController.init({
  worker: mlWorker,
  events: Events,
});

ModelController.init({
  modelView,
  events: Events,
});

TFVisorController.init({
  tfVisorView,
  events: Events,
});

Events.dispatchTrainModel();

userController.renderUsers({
  id: 99,
  name: "Josézin da Silva",
  age: 30,
  purchases: [],
});
