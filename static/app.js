const textInput = document.getElementById("text-input");
const voiceSelect = document.getElementById("voice-select");
const presetSelect = document.getElementById("preset-select");
const rateSlider = document.getElementById("rate-slider");
const pitchSlider = document.getElementById("pitch-slider");
const volumeSlider = document.getElementById("volume-slider");
const exaggerationSlider = document.getElementById("exaggeration-slider");
const rateValue = document.getElementById("rate-value");
const pitchValue = document.getElementById("pitch-value");
const volumeValue = document.getElementById("volume-value");
const exaggerationValue = document.getElementById("exaggeration-value");
const generateBtn = document.getElementById("generate-btn");
const playBtn = document.getElementById("play-btn");
const downloadBtn = document.getElementById("download-btn");
const status = document.getElementById("status");
const audioPlayer = document.getElementById("audio-player");
const referenceAudioInput = document.getElementById("reference-audio");
const progressWrapper = document.getElementById("progress-wrapper");
const progressBar = document.getElementById("progress-bar");
const progressLabel = document.getElementById("progress-label");

let audioUrl = "";
let generationInFlight = false;

function updateProgress(percent) {
  progressBar.style.width = `${percent}%`;
  progressBar.setAttribute("aria-valuenow", String(percent));
  progressLabel.textContent = `${Math.round(percent)}%`;
}

function setIndeterminateProgress(active, label = "") {
  progressBar.classList.toggle("indeterminate", active);
  if (active && label) {
    progressLabel.textContent = label;
  }
}

function showProgress(show) {
  progressWrapper.hidden = !show;
  if (!show) {
    updateProgress(0);
  }
}

function startProgress() {
  showProgress(true);
  setIndeterminateProgress(true, "Preparing generation...");
  updateProgress(8);
}

function stopProgress() {
  setIndeterminateProgress(false);
  updateProgress(100);
  window.setTimeout(() => showProgress(false), 250);
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    status.textContent = data.message || "Ready.";
    return data.ready;
  } catch (error) {
    status.textContent = "Model status unavailable.";
    return false;
  }
}

const presets = {
  neutral: { rate: 150, volume: 1.0, exaggeration: 0.5 },
  warm: { rate: 135, volume: 0.95, exaggeration: 0.35 },
  dramatic: { rate: 125, volume: 0.9, exaggeration: 0.85 },
  energetic: { rate: 180, volume: 1.0, exaggeration: 0.7 }
};

function applyPreset(name) {
  const preset = presets[name] || presets.neutral;
  rateSlider.value = preset.rate;
  volumeSlider.value = preset.volume;
  exaggerationSlider.value = preset.exaggeration;
  syncLabels();
}

function syncLabels() {
  rateValue.textContent = rateSlider.value;
  pitchValue.textContent = Number(pitchSlider.value).toFixed(1);
  volumeValue.textContent = Number(volumeSlider.value).toFixed(1);
  exaggerationValue.textContent = Number(exaggerationSlider.value).toFixed(1);
}

pitchSlider.disabled = true;

async function loadVoices() {
  try {
    const response = await fetch("/api/voices");
    if (!response.ok) {
      throw new Error("Voice discovery is unavailable in this runtime.");
    }

    const voices = await response.json();
    voiceSelect.innerHTML = "";

    const fallback = document.createElement("option");
    fallback.value = "";
    fallback.textContent = "Default voice";
    voiceSelect.appendChild(fallback);

    voices.forEach((voice) => {
      const option = document.createElement("option");
      option.value = voice.id;
      option.textContent = voice.name || voice.id;
      voiceSelect.appendChild(option);
    });
  } catch (error) {
    const fallback = document.createElement("option");
    fallback.value = "";
    fallback.textContent = "Default voice";
    voiceSelect.appendChild(fallback);
  }
}

async function generateAudio() {
  if (generationInFlight) {
    return;
  }

  const ready = await refreshStatus();
  if (!ready) {
    status.textContent = "Chatterbox is still loading. Please wait a moment and try again.";
    generateBtn.disabled = false;
    return;
  }

  const payload = {
    text: textInput.value.trim(),
    rate: Number(rateSlider.value),
    pitch: Number(pitchSlider.value),
    volume: Number(volumeSlider.value),
    exaggeration: Number(exaggerationSlider.value),
    voice: voiceSelect.value
  };

  if (!payload.text) {
    status.textContent = "Please enter some text first.";
    return;
  }

  generationInFlight = true;
  generateBtn.disabled = true;
  status.textContent = "Generating audio...";
  startProgress();

  try {
    const formData = new FormData();
    formData.append("text", payload.text);
    formData.append("rate", String(payload.rate));
    formData.append("pitch", String(payload.pitch));
    formData.append("volume", String(payload.volume));
    formData.append("exaggeration", String(payload.exaggeration));
    formData.append("voice", payload.voice);

    if (referenceAudioInput.files[0]) {
      formData.append("reference_audio", referenceAudioInput.files[0]);
    }

    const response = await fetch("/api/tts", {
      method: "POST",
      body: formData
    });

    let data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Generation failed.");
    }

    const statusUrl = data.status_url;
    while (data.status !== "complete") {
      await new Promise((resolve) => window.setTimeout(resolve, 1000));
      const jobResponse = await fetch(statusUrl);
      data = await jobResponse.json();
      if (!jobResponse.ok || data.status === "failed") {
        throw new Error(data.error || "Generation failed.");
      }
      if (data.total_chunks) {
        setIndeterminateProgress(false);
        updateProgress(data.progress || 0);
        const section = Math.min((data.completed_chunks || 0) + 1, data.total_chunks);
        const stage = data.stage === "decoding" ? "Decoding audio" : "Generating audio";
        const stepPercent = data.total_steps
          ? ` (${Math.round((data.step / data.total_steps) * 100)}%)`
          : "";
        const message = `${stage}: section ${section} of ${data.total_chunks}${stepPercent}`;
        status.textContent = message;
      } else {
        setIndeterminateProgress(true, "Preparing generation...");
        status.textContent = "Preparing generation...";
      }
    }

    updateProgress(100);
    setIndeterminateProgress(false);
    audioUrl = data.audio_url;
    audioPlayer.src = audioUrl;
    audioPlayer.hidden = false;
    playBtn.disabled = false;
    downloadBtn.disabled = false;
    status.textContent = data.chunk_count > 1
      ? `Audio ready (${data.chunk_count} sections combined).`
      : "Audio ready.";
    await audioPlayer.play();
  } catch (error) {
    if (audioUrl) {
      status.textContent += " Press Play if your browser blocked automatic playback.";
    } else {
      status.textContent = error.message;
    }
  } finally {
    stopProgress();
    generationInFlight = false;
    generateBtn.disabled = false;
  }
}

playBtn.addEventListener("click", () => {
  audioPlayer.play();
});

downloadBtn.addEventListener("click", () => {
  if (!audioUrl) return;
  const link = document.createElement("a");
  link.href = audioUrl;
  link.download = "chatterbox-tts.wav";
  link.click();
});

[rateSlider, pitchSlider, volumeSlider, exaggerationSlider].forEach((input) => {
  input.addEventListener("input", syncLabels);
});

presetSelect.addEventListener("change", (event) => {
  applyPreset(event.target.value);
});

generateBtn.addEventListener("click", generateAudio);

syncLabels();
applyPreset("neutral");
loadVoices();
refreshStatus();
