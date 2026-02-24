// Policy Sensitivity Analysis — frontend
(function () {
    "use strict";

    let metadata = null;
    let obsValues = [];
    let stepCount = 0;
    let debounceTimer = null;

    const DEBOUNCE_MS = 30;

    // DOM refs
    const slidersContainer = document.getElementById("sliders-container");
    const envLabel = document.getElementById("env-label");
    const canvas = document.getElementById("action-chart");
    const ctx = canvas.getContext("2d");
    const continuousOutput = document.getElementById("continuous-output");
    const valueDisplay = document.getElementById("value-display");
    const bestActionDisplay = document.getElementById("best-action-display");
    const resetObsBtn = document.getElementById("reset-obs-btn");
    const resetRnnBtn = document.getElementById("reset-rnn-btn");
    const rnnModeSelect = document.getElementById("rnn-mode-select");
    const stepCounter = document.getElementById("step-counter");

    // ---- Init ----

    async function init() {
        const resp = await fetch("/api/metadata");
        metadata = await resp.json();

        envLabel.textContent = metadata.env_name || `obs=${metadata.obs_dim}`;

        if (!metadata.is_recurrent) {
            rnnModeSelect.parentElement.querySelector("select").disabled = true;
        }

        buildSliders();
        bindControls();
        doInfer();
    }

    // ---- Build sliders ----

    function buildSliders() {
        slidersContainer.innerHTML = "";
        const dim = metadata.obs_dim || 0;
        obsValues = new Array(dim).fill(0);

        for (let i = 0; i < dim; i++) {
            const name = (metadata.obs_names && metadata.obs_names[i]) || `Feature ${i}`;
            const isBinary = metadata.obs_binary && metadata.obs_binary[i];
            const range = (metadata.obs_ranges && metadata.obs_ranges[i]) || [-1, 1];

            if (isBinary) {
                buildToggle(i, name);
            } else {
                buildSlider(i, name, range[0], range[1]);
            }
        }
    }

    function buildSlider(index, name, min, max) {
        const mid = (min + max) / 2;
        obsValues[index] = mid;

        const row = document.createElement("div");
        row.className = "slider-row";

        const label = document.createElement("span");
        label.className = "slider-label";
        label.textContent = name;
        label.title = name;

        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = min;
        slider.max = max;
        slider.step = ((max - min) / 200).toFixed(6);
        slider.value = mid;
        slider.dataset.index = index;

        const numInput = document.createElement("input");
        numInput.type = "number";
        numInput.min = min;
        numInput.max = max;
        numInput.step = "any";
        numInput.value = mid.toFixed(2);
        numInput.dataset.index = index;

        slider.addEventListener("input", () => {
            const val = parseFloat(slider.value);
            obsValues[index] = val;
            numInput.value = val.toFixed(2);
            scheduleInfer();
        });

        numInput.addEventListener("change", () => {
            let val = parseFloat(numInput.value);
            if (isNaN(val)) val = (min + max) / 2;
            val = Math.max(min, Math.min(max, val));
            obsValues[index] = val;
            slider.value = val;
            numInput.value = val.toFixed(2);
            scheduleInfer();
        });

        row.appendChild(label);
        row.appendChild(slider);
        row.appendChild(numInput);
        slidersContainer.appendChild(row);
    }

    function buildToggle(index, name) {
        obsValues[index] = 0;

        const row = document.createElement("div");
        row.className = "toggle-row";

        const label = document.createElement("span");
        label.className = "slider-label";
        label.textContent = name;
        label.title = name;

        const toggle = document.createElement("label");
        toggle.className = "toggle-switch";
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.dataset.index = index;
        const sliderEl = document.createElement("span");
        sliderEl.className = "toggle-slider";
        toggle.appendChild(cb);
        toggle.appendChild(sliderEl);

        const valDisplay = document.createElement("span");
        valDisplay.className = "toggle-value";
        valDisplay.textContent = "0";

        cb.addEventListener("change", () => {
            const val = cb.checked ? 1 : 0;
            obsValues[index] = val;
            valDisplay.textContent = val;
            scheduleInfer();
        });

        row.appendChild(label);
        row.appendChild(toggle);
        row.appendChild(valDisplay);
        slidersContainer.appendChild(row);
    }

    // ---- Controls ----

    function bindControls() {
        resetObsBtn.addEventListener("click", () => {
            buildSliders();
            doInfer();
        });

        rnnModeSelect.addEventListener("change", () => {
            const isStateful = rnnModeSelect.value === "stateful";
            resetRnnBtn.classList.toggle("hidden", !isStateful);
            stepCounter.classList.toggle("hidden", !isStateful);
            if (!isStateful) {
                stepCount = 0;
                stepCounter.textContent = "Step: 0";
                fetch("/api/reset_state", { method: "POST" });
            }
        });

        resetRnnBtn.addEventListener("click", async () => {
            await fetch("/api/reset_state", { method: "POST" });
            stepCount = 0;
            stepCounter.textContent = "Step: 0";
            doInfer();
        });
    }

    // ---- Inference ----

    function scheduleInfer() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(doInfer, DEBOUNCE_MS);
    }

    async function doInfer() {
        const stateful = rnnModeSelect.value === "stateful";
        const resp = await fetch("/api/infer", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ obs: obsValues, stateful }),
        });
        const result = await resp.json();

        if (stateful) {
            stepCount++;
            stepCounter.textContent = `Step: ${stepCount}`;
        }

        renderOutput(result);
    }

    // ---- Render output ----

    function renderOutput(result) {
        // Value
        if (result.value !== undefined) {
            valueDisplay.textContent = `Value: ${result.value.toFixed(4)}`;
        }

        if (result.action_type === "discrete") {
            renderDiscrete(result);
        } else if (result.action_type === "continuous") {
            renderContinuous(result);
        }
    }

    function renderDiscrete(result) {
        canvas.classList.remove("hidden");
        continuousOutput.classList.add("hidden");

        const probs = result.probabilities || [];
        const bestIdx = result.best_action;
        const names = [];
        for (let i = 0; i < probs.length; i++) {
            names.push((metadata.action_names && metadata.action_names[i]) || `Action ${i}`);
        }

        // Best action label
        bestActionDisplay.textContent = `Best: ${names[bestIdx]} (${(probs[bestIdx] * 100).toFixed(1)}%)`;

        // Draw bar chart
        const dpr = window.devicePixelRatio || 1;
        const barH = 26;
        const gap = 6;
        const labelW = 90;
        const pctW = 50;
        const padX = 10;
        const padY = 10;
        const chartH = probs.length * (barH + gap) - gap + padY * 2;

        canvas.style.height = chartH + "px";
        canvas.width = canvas.clientWidth * dpr;
        canvas.height = chartH * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        const w = canvas.clientWidth;
        const barMaxW = w - labelW - pctW - padX * 2;

        ctx.clearRect(0, 0, w, chartH);
        ctx.font = "12px monospace";
        ctx.textBaseline = "middle";

        for (let i = 0; i < probs.length; i++) {
            const y = padY + i * (barH + gap);
            const p = probs[i];
            const isBest = i === bestIdx;

            // Label
            ctx.fillStyle = isBest ? "#0ea5e9" : "#8899aa";
            ctx.textAlign = "right";
            ctx.fillText(names[i], labelW, y + barH / 2);

            // Bar bg
            const bx = labelW + padX;
            ctx.fillStyle = "#233554";
            roundRect(ctx, bx, y + 2, barMaxW, barH - 4, 3);
            ctx.fill();

            // Bar fill
            const fillW = Math.max(1, p * barMaxW);
            ctx.fillStyle = isBest ? "#0ea5e9" : "#e94560";
            roundRect(ctx, bx, y + 2, fillW, barH - 4, 3);
            ctx.fill();

            // Percentage
            ctx.fillStyle = "#e0e0e0";
            ctx.textAlign = "left";
            ctx.fillText(`${(p * 100).toFixed(1)}%`, bx + barMaxW + 6, y + barH / 2);
        }
    }

    function renderContinuous(result) {
        canvas.classList.add("hidden");
        continuousOutput.classList.remove("hidden");

        const mu = result.mu || [];
        const sigma = result.sigma || [];
        const names = [];
        for (let i = 0; i < mu.length; i++) {
            names.push((metadata.action_names && metadata.action_names[i]) || `Action ${i}`);
        }

        bestActionDisplay.textContent = `Mu: [${mu.map(v => v.toFixed(3)).join(", ")}]`;

        // Build / update rows
        continuousOutput.innerHTML = "";
        for (let i = 0; i < mu.length; i++) {
            const row = document.createElement("div");
            row.className = "mu-row";

            const label = document.createElement("span");
            label.className = "mu-label";
            label.textContent = names[i];

            const barContainer = document.createElement("div");
            barContainer.className = "mu-bar-container";

            // Determine display range — use [-2, 2] as default, expand if needed
            const lo = -2, hi = 2;
            const range = hi - lo;

            // Sigma band
            if (sigma.length > i) {
                const s = sigma[i];
                const bandLeft = Math.max(0, ((mu[i] - s) - lo) / range) * 100;
                const bandRight = Math.min(100, ((mu[i] + s) - lo) / range) * 100;
                const band = document.createElement("div");
                band.className = "mu-sigma-band";
                band.style.left = bandLeft + "%";
                band.style.width = (bandRight - bandLeft) + "%";
                barContainer.appendChild(band);
            }

            // Mu marker
            const marker = document.createElement("div");
            marker.className = "mu-marker";
            const pos = Math.max(0, Math.min(100, ((mu[i] - lo) / range) * 100));
            marker.style.left = `calc(${pos}% - 1.5px)`;
            barContainer.appendChild(marker);

            const val = document.createElement("span");
            val.className = "mu-value";
            val.textContent = mu[i].toFixed(3);

            row.appendChild(label);
            row.appendChild(barContainer);
            row.appendChild(val);
            continuousOutput.appendChild(row);
        }
    }

    // ---- Helpers ----

    function roundRect(ctx, x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.arcTo(x + w, y, x + w, y + r, r);
        ctx.lineTo(x + w, y + h - r);
        ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
        ctx.lineTo(x + r, y + h);
        ctx.arcTo(x, y + h, x, y + h - r, r);
        ctx.lineTo(x, y + r);
        ctx.arcTo(x, y, x + r, y, r);
        ctx.closePath();
    }

    // ---- Start ----
    init();
})();
