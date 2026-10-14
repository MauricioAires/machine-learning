import { View } from "./View.js";

export class ModelView extends View {
  #trainModelBtn = document.querySelector("#trainModelBtn");
  #calculateRiskBtn = document.querySelector("#calculateRiskBtn");
  #riskForm = document.querySelector("#riskForm");
  #creditLimitInput = document.querySelector("#creditLimitInput");
  #utilizationInput = document.querySelector("#utilizationInput");
  #usedAmountInput = document.querySelector("#usedAmountInput");
  #trainingDataCount = document.querySelector("#trainingDataCount");
  #trainingDataTableBody = document.querySelector("#trainingDataTableBody");
  #riskScoreValue = document.querySelector("#riskScoreValue");
  #riskLevelBadge = document.querySelector("#riskLevelBadge");
  #riskDescription = document.querySelector("#riskDescription");
  #modelStatusText = document.querySelector("#modelStatusText");
  #trainingProgressBar = document.querySelector("#trainingProgressBar");
  #trainingProgressText = document.querySelector("#trainingProgressText");
  #onTrainModel;
  #onCalculateRisk;

  constructor() {
    super();
    this.attachEventListeners();
    this.syncUsedAmount();
    this.setModelReady(false);
  }

  registerTrainModelCallback(callback) {
    this.#onTrainModel = callback;
  }
  registerCalculateRiskCallback(callback) {
    this.#onCalculateRisk = callback;
  }

  attachEventListeners() {
    this.#trainModelBtn.addEventListener("click", () => {
      this.#onTrainModel();
    });
    this.#calculateRiskBtn.addEventListener("click", () => {
      this.triggerRiskCalculation();
    });

    [this.#creditLimitInput, this.#utilizationInput].forEach((input) => {
      input.addEventListener("input", () => {
        this.syncUsedAmount();
        this.triggerRiskCalculation();
      });
    });

    this.#riskForm.addEventListener("submit", (event) => {
      event.preventDefault();
      this.triggerRiskCalculation();
    });
  }

  triggerRiskCalculation() {
    if (this.#onCalculateRisk) {
      this.#onCalculateRisk(this.getRiskProfile());
    }
  }

  updateTrainingProgress(progress) {
    const progressValue = progress?.progress ?? 0;

    this.#trainModelBtn.disabled = true;
    this.#calculateRiskBtn.disabled = true;
    this.#modelStatusText.textContent = "Treinando modelo de risco...";
    this.#trainingProgressBar.style.width = `${progressValue}%`;
    this.#trainingProgressBar.setAttribute("aria-valuenow", String(progressValue));
    this.#trainingProgressText.textContent = `${Math.round(progressValue)}%`;

    if (progressValue < 100) {
      this.#trainModelBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Treinando...';
      return;
    }

    this.#trainModelBtn.disabled = false;
    this.#calculateRiskBtn.disabled = false;
    this.#trainModelBtn.innerHTML = "Treinar modelo";
    this.#modelStatusText.textContent = "Modelo treinado e pronto para simular risco.";
  }

  setModelReady(isReady) {
    this.#calculateRiskBtn.disabled = !isReady;
    if (!isReady) {
      this.#modelStatusText.textContent = "Treine o modelo para habilitar a simulação.";
    }
  }

  renderTrainingData(samples, summary) {
    if (!samples || samples.length === 0) {
      this.#trainingDataTableBody.innerHTML = '<tr><td colspan="12" class="text-center text-muted py-4">Nenhum dado de treinamento disponível.</td></tr>';
      this.#trainingDataCount.textContent = "0 registros";
      return;
    }

    const html = samples
      .map((sample) => {
        return `
                <tr>
                    <td>${sample.id}</td>
                    <td>${sample.age}</td>
                    <td>$${this.formatCurrency(sample.monthlyIncome)}</td>
                    <td>$${this.formatCurrency(sample.creditLimit)}</td>
                    <td>$${this.formatCurrency(sample.usedAmount)}</td>
                    <td>${sample.utilizationRate.toFixed(1)}%</td>
                    <td>${sample.delinquencyCount}</td>
                    <td>${sample.openInstallments}</td>
                    <td>${sample.relationshipYears.toFixed(1)}</td>
                    <td>${sample.recentInquiries}</td>
                    <td>${sample.activeCards}</td>
                    <td><span class="badge ${sample.label ? "text-bg-danger" : "text-bg-success"}">${sample.labelText}</span></td>
                    <td>${sample.riskScore.toFixed(1)}%</td>
                </tr>
            `;
      })
      .join("");

    this.#trainingDataTableBody.innerHTML = html;
    this.#trainingDataCount.textContent = `${summary?.total ?? samples.length} registros`;
  }

  renderRiskResult(result) {
    const score = Number(result?.score ?? 0);
    const level = result?.level ?? this.getRiskLevel(score);

    this.#riskScoreValue.textContent = `${Math.round(score)}%`;
    this.#riskLevelBadge.textContent = level.label;
    this.#riskLevelBadge.className = `badge ${level.className}`;
    this.#riskDescription.textContent = level.description;
  }

  getRiskProfile() {
    return {
      age: Number(document.querySelector("#ageInput").value),
      monthlyIncome: Number(document.querySelector("#monthlyIncomeInput").value),
      creditLimit: Number(this.#creditLimitInput.value),
      usedAmount: Number(this.#usedAmountInput.value),
      utilizationRate: Number(this.#utilizationInput.value),
      delinquencyCount: Number(document.querySelector("#delinquencyCountInput").value),
      openInstallments: Number(document.querySelector("#openInstallmentsInput").value),
      relationshipYears: Number(document.querySelector("#relationshipYearsInput").value),
      recentInquiries: Number(document.querySelector("#recentInquiriesInput").value),
      activeCards: Number(document.querySelector("#activeCardsInput").value),
    };
  }

  syncUsedAmount() {
    const creditLimit = Number(this.#creditLimitInput.value || 0);
    const utilizationRate = Number(this.#utilizationInput.value || 0);
    const usedAmount = (creditLimit * utilizationRate) / 100;

    this.#usedAmountInput.value = usedAmount ? usedAmount.toFixed(2) : "0.00";
  }

  setDefaultProfile(profile) {
    document.querySelector("#ageInput").value = profile.age;
    document.querySelector("#monthlyIncomeInput").value = profile.monthlyIncome;
    this.#creditLimitInput.value = profile.creditLimit;
    this.#utilizationInput.value = profile.utilizationRate;
    document.querySelector("#delinquencyCountInput").value = profile.delinquencyCount;
    document.querySelector("#openInstallmentsInput").value = profile.openInstallments;
    document.querySelector("#relationshipYearsInput").value = profile.relationshipYears;
    document.querySelector("#recentInquiriesInput").value = profile.recentInquiries;
    document.querySelector("#activeCardsInput").value = profile.activeCards;
    this.syncUsedAmount();
  }

  getRiskLevel(score) {
    if (score >= 75) {
      return {
        label: "Risco alto",
        className: "text-bg-danger",
        description: "O perfil apresenta sinais fortes de inadimplência.",
      };
    }

    if (score >= 45) {
      return {
        label: "Risco moderado",
        className: "text-bg-warning",
        description: "O perfil merece atenção e acompanhamento.",
      };
    }

    return {
      label: "Risco baixo",
      className: "text-bg-success",
      description: "O perfil mostra baixa propensão a inadimplência.",
    };
  }

  formatCurrency(value) {
    return Number(value).toLocaleString("pt-BR", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  }
}
