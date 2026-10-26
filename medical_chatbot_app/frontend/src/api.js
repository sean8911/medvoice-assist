async function request(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || "The application could not complete the request.");
  }
  return payload;
}

export function checkHealth() {
  return request("/api/health");
}

export function analyzeText(text) {
  return request("/api/analyze/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
}

export function analyzeClarification(originalNarrative, answer) {
  return request("/api/analyze/clarification", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      original_narrative: originalNarrative,
      answer,
    }),
  });
}

export function transcribeAudio(file, preprocess = true) {
  const form = new FormData();
  form.append("audio", file, file.name || "recording.webm");
  form.append("preprocess", String(preprocess));
  form.append("language", "en");
  return request("/api/transcribe", {
    method: "POST",
    body: form,
  });
}
