import {
  Activity,
  ChevronDown,
  Clock3,
  Download,
  FlaskConical,
  Server,
  X,
} from "lucide-react";

function downloadResult(analysis) {
  const blob = new Blob([JSON.stringify(analysis, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `medvoice-result-${analysis.request_id || "session"}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

function Metric({ label, value }) {
  return (
    <div className="research-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default function ResearchPanel({ open, onClose, analysis, transcriptResult }) {
  if (!open) return null;
  const extraction = analysis?.symptom_extraction;
  const classification = analysis?.classification;

  return (
    <aside className="research-drawer" aria-label="Research details">
      <div className="research-drawer__top">
        <div>
          <span className="section-kicker">Thesis evidence</span>
          <h2>Research details</h2>
        </div>
        <button className="icon-button" onClick={onClose} aria-label="Close research details">
          <X size={18} />
        </button>
      </div>

      {!analysis && !transcriptResult ? (
        <div className="research-empty">
          <FlaskConical size={28} />
          <h3>No pipeline result yet</h3>
          <p>Run a voice or text scenario to inspect each processing stage.</p>
        </div>
      ) : (
        <>
          <div className="research-metrics">
            <Metric
              label="Total time"
              value={`${analysis?.processing_time_seconds ?? transcriptResult?.processing_time_seconds ?? 0}s`}
            />
            <Metric label="Device" value={analysis?.model_runtime?.device || transcriptResult?.device || "-"} />
            <Metric label="Classes" value={classification?.number_of_disease_classes || 773} />
          </div>

          {transcriptResult && (
            <details open className="research-section">
              <summary>
                <Activity size={16} /> Audio and Whisper <ChevronDown size={15} />
              </summary>
              <dl>
                <dt>Whisper model</dt>
                <dd>{transcriptResult.whisper_model}</dd>
                <dt>Original duration</dt>
                <dd>{transcriptResult.audio_preprocessing?.original_duration_seconds}s</dd>
                <dt>Processed duration</dt>
                <dd>{transcriptResult.audio_preprocessing?.processed_duration_seconds}s</dd>
                <dt>Transcript</dt>
                <dd>{transcriptResult.transcript}</dd>
              </dl>
            </details>
          )}

          {extraction && (
            <details open className="research-section">
              <summary>
                <FlaskConical size={16} /> Extracted evidence <ChevronDown size={15} />
              </summary>
              <dl>
                <dt>Current</dt>
                <dd>{extraction.present_symptoms.join(", ") || "None"}</dd>
                <dt>Negated</dt>
                <dd>{extraction.negated_symptoms.join(", ") || "None"}</dd>
                <dt>Uncertain</dt>
                <dd>{extraction.uncertain_symptoms.join(", ") || "None"}</dd>
                <dt>BioBERT input</dt>
                <dd>{analysis.biobert_input || "Classification bypassed"}</dd>
              </dl>
            </details>
          )}

          {classification && (
            <details className="research-section">
              <summary>
                <Server size={16} /> BioBERT output <ChevronDown size={15} />
              </summary>
              <dl>
                <dt>Status</dt>
                <dd>{classification.status}</dd>
                <dt>Top score</dt>
                <dd>{(classification.model_score * 100).toFixed(2)}%</dd>
                <dt>Top-two margin</dt>
                <dd>
                  {classification.top_two_score_margin == null
                    ? "Not available"
                    : `${(classification.top_two_score_margin * 100).toFixed(2)}%`}
                </dd>
              </dl>
              {classification.top_predictions?.length > 0 && (
                <div className="score-list">
                  {classification.top_predictions.map((item) => (
                    <div key={item.label_id}>
                      <span>{item.disease}</span>
                      <strong>{(item.model_score * 100).toFixed(1)}%</strong>
                    </div>
                  ))}
                </div>
              )}
            </details>
          )}

          {analysis?.stage_timings_seconds && (
            <details className="research-section">
              <summary>
                <Clock3 size={16} /> Stage timings <ChevronDown size={15} />
              </summary>
              <div className="timing-list">
                {Object.entries(analysis.stage_timings_seconds).map(([name, value]) => (
                  <div key={name}>
                    <span>{name.replaceAll("_", " ")}</span>
                    <strong>{value}s</strong>
                  </div>
                ))}
              </div>
            </details>
          )}

          {analysis && (
            <button className="download-button" onClick={() => downloadResult(analysis)}>
              <Download size={16} /> Download complete JSON
            </button>
          )}
        </>
      )}
    </aside>
  );
}
