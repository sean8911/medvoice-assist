import {
  AudioWaveform,
  Check,
  CircleDashed,
  FileText,
  HelpCircle,
  MessageSquareText,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

const DEFINITIONS = [
  { id: "transcription", label: "Whisper transcript", icon: AudioWaveform },
  { id: "text_preprocessing", label: "Text preparation", icon: FileText },
  { id: "symptom_extraction", label: "Symptom extraction", icon: Sparkles },
  { id: "urgency_detection", label: "Safety screening", icon: ShieldCheck },
  { id: "clarification", label: "Clarification", icon: HelpCircle },
  { id: "biobert_classification", label: "BioBERT classification", icon: CircleDashed },
  { id: "rule_based_response", label: "Guided response", icon: MessageSquareText },
];

export default function PipelineSteps({ analysis, hasTranscript, busy }) {
  const trace = new Map(
    (analysis?.pipeline_trace || []).map((item) => [item.stage, item.status]),
  );

  return (
    <div className="pipeline-list">
      {DEFINITIONS.map(({ id, label, icon: Icon }) => {
        let status = trace.get(id) || "waiting";
        if (id === "transcription" && hasTranscript) status = "completed";
        if (id === "transcription" && analysis && !hasTranscript) status = "not used";
        if (busy && status === "waiting") status = "processing";
        return (
          <div className={`pipeline-step pipeline-step--${status}`} key={id}>
            <span className="pipeline-step__icon">
              {status === "completed" ? <Check size={14} /> : <Icon size={15} />}
            </span>
            <span>{label}</span>
            <small>{status}</small>
          </div>
        );
      })}
    </div>
  );
}
