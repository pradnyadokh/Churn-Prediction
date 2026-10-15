const API_BASE_URL = window.API_BASE_URL || "http://127.0.0.1:8000";

const form = document.getElementById("churn-form");
const resultSection = document.getElementById("result-section");
const errorMessage = document.getElementById("error-message");
const loadingMessage = document.getElementById("loading-message");
const predictionLabel = document.getElementById("prediction-label");
const probabilityValue = document.getElementById("probability-value");
const predictAgainButton = document.getElementById("predict-again");

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
}

function clearError() {
    errorMessage.textContent = "";
    errorMessage.classList.add("hidden");
}

function getFormPayload() {
    return {
        tenure: Number(document.getElementById("tenure").value),
        citytier: Number(document.getElementById("citytier").value),
        warehousetohome: Number(document.getElementById("warehousetohome").value),
        gender: document.getElementById("gender").value,
        hourspendonapp: Number(document.getElementById("hourspendonapp").value),
        numberofdeviceregistered: Number(document.getElementById("numberofdeviceregistered").value),
        satisfactionscore: Number(document.getElementById("satisfactionscore").value),
        maritalstatus: document.getElementById("maritalstatus").value,
        numberofaddress: Number(document.getElementById("numberofaddress").value),
        complain: Number(document.getElementById("complain").value),
        orderamounthikefromlastyear: Number(document.getElementById("orderamounthikefromlastyear").value),
        couponused: Number(document.getElementById("couponused").value),
        ordercount: Number(document.getElementById("ordercount").value),
        daysincelastorder: Number(document.getElementById("daysincelastorder").value),
        cashbackamount: Number(document.getElementById("cashbackamount").value),
    };
}

async function predictChurn(event) {
    event.preventDefault();
    clearError();
    loadingMessage.classList.remove("hidden");
    resultSection.classList.add("hidden");

    try {
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(getFormPayload()),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Prediction request failed.");
        }

        predictionLabel.textContent = data.prediction;
        predictionLabel.className = data.prediction === "Churn" ? "churn" : "not-churn";
        probabilityValue.textContent = data.churn_probability;
        resultSection.classList.remove("hidden");
    } catch (error) {
        showError(
            `Unable to get prediction. Make sure the FastAPI backend is running on ${API_BASE_URL}. ` +
            `Details: ${error.message}`
        );
    } finally {
        loadingMessage.classList.add("hidden");
    }
}

form.addEventListener("submit", predictChurn);

predictAgainButton.addEventListener("click", () => {
    resultSection.classList.add("hidden");
    clearError();
    window.scrollTo({ top: 0, behavior: "smooth" });
});
