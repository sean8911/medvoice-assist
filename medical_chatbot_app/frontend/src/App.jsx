import { useEffect, useRef, useState } from "react";
import {
  Activity,
  AudioLines,
  Check,
  ChevronRight,
  CircleStop,
  FileAudio,
  FlaskConical,
  Info,
  LockKeyhole,
  Menu,
  MessageCircleMore,
  Mic,
  Paperclip,
  Plus,
  RotateCcw,
  Send,
  ShieldCheck,
  Sparkles,
  Volume2,
  X,
} from "lucide-react";
import { analyzeClarification, analyzeText, checkHealth, transcribeAudio } from "./api";
import BrandMark from "./components/BrandMark";
import PipelineSteps from "./components/PipelineSteps";
import ResearchPanel from "./components/ResearchPanel";
import ResponseCard from "./components/ResponseCard";

const WELCOME_MESSAGE = {
  id: "welcome",
  role: "assistant",
  kind: "welcome",
};

const DEMO_PROMPTS = [
  "I have a headache every morning.",
  "I have a cough, sore throat and blocked nose.",
  "I have crushing chest pain and shortness of breath.",
];

function makeId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function WelcomeMessage({ onPrompt }) {
  return (
    <div className="welcome-card">
      <div className="welcome-card__glow" />
      <span className="assistant-avatar">
        <AudioLines size={20} />
      </span>
      <div className="welcome-card__content">
        <span className="section-kicker">Voice-first symptom support</span>
        <h2>Tell me what you are experiencing.</h2>
        <p>
          Speak naturally or type what you are experiencing. No special format is needed;
          I will show what I understood and ask one short follow-up when necessary.
        </p>
        <div className="suggestion-row">
          {DEMO_PROMPTS.map((prompt) => (
            <button key={prompt} onClick={() => onPrompt(prompt)}>
              {prompt}
              <ChevronRight size={14} />
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function TranscriptReview({ review, onChange, onConfirm, onCancel, busy }) {
  return (
    <div className="transcript-review">
      <div className="transcript-review__head">
        <span className="transcript-review__icon">
          <Volume2 size={18} />
        </span>
        <div>
          <span className="section-kicker">Whisper transcription</span>
          <h3>Confirm what you said</h3>
        </div>
        <span className="confidence-note">Edit any transcription errors</span>
      </div>
      {review.audioUrl && <audio controls src={review.audioUrl} />}
      <textarea
        value={review.text}
        onChange={(event) => onChange(event.target.value)}
        aria-label="Confirmed transcript"
        rows={3}
      />
      <div className="transcript-review__meta">
        <span>{review.result?.audio_preprocessing?.processed_duration_seconds}s processed audio</span>
        <span>Temporary file deleted</span>
      </div>
      <div className="transcript-review__actions">
        <button className="button button--ghost" onClick={onCancel} disabled={busy}>
          <X size={15} /> Cancel
        </button>
        <button className="button button--primary" onClick={onConfirm} disabled={busy || !review.text.trim()}>
          <Check size={16} /> Confirm and analyse
        </button>
      </div>
    </div>
  );
}

function ConversationMessage({ message, onPrompt }) {
  if (message.kind === "welcome") return <WelcomeMessage onPrompt={onPrompt} />;
  if (message.role === "user") {
    return (
      <div className="chat-row chat-row--user">
        <div className="user-bubble">
          {message.source === "voice" && (
            <span className="message-source"><Mic size={12} /> Confirmed voice transcript</span>
          )}
          <p>{message.text}</p>
        </div>
      </div>
    );
  }
  if (message.kind === "error") {
    return (
      <div className="chat-row chat-row--assistant">
        <span className="assistant-avatar assistant-avatar--error"><Info size={18} /></span>
        <div className="error-bubble">
          <strong>Something needs attention</strong>
          <p>{message.text}</p>
        </div>
      </div>
    );
  }
  return (
    <div className="chat-row chat-row--assistant">
      <span className="assistant-avatar"><AudioLines size={18} /></span>
      <ResponseCard response={message.response} extraction={message.extraction} />
    </div>
  );
}

function InsightsPanel({ analysis, health, busy, hasTranscript }) {
  const extraction = analysis?.symptom_extraction;
  const urgency = analysis?.urgency?.urgency_level || "not assessed";
  return (
    <aside className="insights-panel">
      <section className="insight-card system-card">
        <div className="insight-card__head">
          <div>
            <span className="section-kicker">System status</span>
            <h3>Local AI pipeline</h3>
          </div>
          <span className={`status-dot ${health?.status === "ok" ? "status-dot--online" : ""}`} />
        </div>
        <div className="system-row">
          <span>Trained BioBERT</span>
          <strong>{health?.runtime?.biobert_model_valid ? "Verified" : "Checking"}</strong>
        </div>
        <div className="system-row">
          <span>Inference device</span>
          <strong>{health?.runtime?.device?.toUpperCase() || "-"}</strong>
        </div>
        <div className="system-row">
          <span>Disease classes</span>
          <strong>{health?.runtime?.disease_classes || 773}</strong>
        </div>
      </section>

      <section className="insight-card">
        <div className="insight-card__head">
          <div>
            <span className="section-kicker">Current evidence</span>
            <h3>Detected symptoms</h3>
          </div>
          <Sparkles size={17} />
        </div>
        {!extraction ? (
          <p className="empty-copy">Symptoms will appear here after you confirm your description.</p>
        ) : (
          <div className="symptom-groups">
            <div>
              <span>Current</span>
              <div className="tag-row">
                {extraction.present_symptoms.length ? extraction.present_symptoms.map((item) => (
                  <span className="symptom-tag symptom-tag--present" key={item}>{item}</span>
                )) : <small>None detected</small>}
              </div>
            </div>
            {extraction.uncertain_symptoms.length > 0 && (
              <div>
                <span>Uncertain</span>
                <div className="tag-row">
                  {extraction.uncertain_symptoms.map((item) => (
                    <span className="symptom-tag symptom-tag--uncertain" key={item}>{item}</span>
                  ))}
                </div>
              </div>
            )}
            {extraction.negated_symptoms.length > 0 && (
              <div>
                <span>Excluded</span>
                <div className="tag-row">
                  {extraction.negated_symptoms.map((item) => (
                    <span className="symptom-tag symptom-tag--negated" key={item}>{item}</span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
        <div className={`urgency-chip urgency-chip--${urgency}`}>
          <ShieldCheck size={14} />
          Urgency: {urgency}
        </div>
      </section>

      <section className="insight-card">
        <div className="insight-card__head">
          <div>
            <span className="section-kicker">Processing</span>
            <h3>Pipeline stages</h3>
          </div>
          <Activity size={17} />
        </div>
        <PipelineSteps analysis={analysis} hasTranscript={hasTranscript} busy={busy} />
      </section>
    </aside>
  );
}

export default function App() {
  const [messages, setMessages] = useState([WELCOME_MESSAGE]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(null);
  const [health, setHealth] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [transcriptResult, setTranscriptResult] = useState(null);
  const [transcriptReview, setTranscriptReview] = useState(null);
  const [originalNarrative, setOriginalNarrative] = useState("");
  const [awaitingClarification, setAwaitingClarification] = useState(false);
  const [researchOpen, setResearchOpen] = useState(false);
  const [preprocessAudio, setPreprocessAudio] = useState(true);
  const [recording, setRecording] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const fileInputRef = useRef(null);
  const recorderRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const chunksRef = useRef([]);
  const chatEndRef = useRef(null);

  useEffect(() => {
    checkHealth().then(setHealth).catch(() => setHealth({ status: "offline" }));
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, transcriptReview, busy]);

  useEffect(() => () => {
    if (transcriptReview?.audioUrl) URL.revokeObjectURL(transcriptReview.audioUrl);
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
  }, [transcriptReview?.audioUrl]);

  const addError = (error) => {
    setMessages((items) => [...items, {
      id: makeId(),
      role: "assistant",
      kind: "error",
      text: error.message || String(error),
    }]);
  };

  const processNarrative = async (text, source = "text") => {
    const cleanText = text.trim();
    if (!cleanText || busy) return;
    setMessages((items) => [...items, { id: makeId(), role: "user", text: cleanText, source }]);
    setDraft("");
    setBusy("Analysing symptoms and safety signals");
    try {
      const result = awaitingClarification && originalNarrative
        ? await analyzeClarification(originalNarrative, cleanText)
        : await analyzeText(cleanText);
      setAnalysis(result);
      setMessages((items) => [...items, {
        id: result.request_id,
        role: "assistant",
        kind: "response",
        response: result.rule_based_response,
        extraction: result.symptom_extraction,
      }]);
      const needsMore = result.rule_based_response?.response_type === "clarification";
      setAwaitingClarification(needsMore);
      if (needsMore) {
        setOriginalNarrative(result.conversation?.combined_narrative || result.transcript);
      } else {
        setOriginalNarrative("");
      }
    } catch (error) {
      addError(error);
    } finally {
      setBusy(null);
    }
  };

  const handleAudio = async (file, audioUrl) => {
    setBusy("Cleaning audio and transcribing with Whisper");
    try {
      const result = await transcribeAudio(file, preprocessAudio);
      setTranscriptResult(result);
      setTranscriptReview({ text: result.transcript, audioUrl, result });
    } catch (error) {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      addError(error);
    } finally {
      setBusy(null);
    }
  };

  const handleFile = (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    handleAudio(file, URL.createObjectURL(file));
  };

  const startRecording = async () => {
    if (busy) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      chunksRef.current = [];
      const recorder = new MediaRecorder(stream);
      recorderRef.current = recorder;
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        const mimeType = recorder.mimeType || "audio/webm";
        const blob = new Blob(chunksRef.current, { type: mimeType });
        const extension = mimeType.includes("ogg") ? "ogg" : "webm";
        const file = new File([blob], `voice-symptoms.${extension}`, { type: mimeType });
        const audioUrl = URL.createObjectURL(blob);
        stream.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;
        handleAudio(file, audioUrl);
      };
      recorder.start();
      setRecording(true);
    } catch {
      addError(new Error("Microphone access was unavailable. You can upload audio or type instead."));
    }
  };

  const stopRecording = () => {
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
    setRecording(false);
  };

  const confirmTranscript = () => {
    const confirmed = transcriptReview.text;
    const audioUrl = transcriptReview.audioUrl;
    setTranscriptReview(null);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    processNarrative(confirmed, "voice");
  };

  const cancelTranscript = () => {
    if (transcriptReview?.audioUrl) URL.revokeObjectURL(transcriptReview.audioUrl);
    setTranscriptReview(null);
  };

  const resetSession = () => {
    setMessages([WELCOME_MESSAGE]);
    setDraft("");
    setAnalysis(null);
    setTranscriptResult(null);
    setTranscriptReview(null);
    setOriginalNarrative("");
    setAwaitingClarification(false);
    setResearchOpen(false);
  };

  const submitDraft = (event) => {
    event?.preventDefault();
    processNarrative(draft);
  };

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileMenuOpen ? "sidebar--open" : ""}`}>
        <div className="sidebar__top">
          <BrandMark />
          <button className="icon-button sidebar__close" onClick={() => setMobileMenuOpen(false)} aria-label="Close menu">
            <X size={18} />
          </button>
        </div>

        <button className="new-session-button" onClick={resetSession}>
          <Plus size={17} /> New consultation
        </button>

        <nav className="sidebar-nav" aria-label="Application sections">
          <button className="sidebar-nav__item sidebar-nav__item--active">
            <MessageCircleMore size={17} /> Consultation
          </button>
          <button className="sidebar-nav__item" onClick={() => setResearchOpen(true)}>
            <FlaskConical size={17} /> Research details
          </button>
        </nav>

        <div className="sidebar__spacer" />

        <label className="preprocess-toggle">
          <div>
            <span>Audio enhancement</span>
            <small>Denoise before Whisper</small>
          </div>
          <input
            type="checkbox"
            checked={preprocessAudio}
            onChange={(event) => setPreprocessAudio(event.target.checked)}
          />
          <span className="toggle-track"><span /></span>
        </label>

        <div className="privacy-card">
          <LockKeyhole size={17} />
          <div>
            <strong>Private by design</strong>
            <span>No database. Audio is temporary.</span>
          </div>
        </div>

        <div className="prototype-label">
          <span>THESIS 2</span>
          Research prototype v1.0
        </div>
      </aside>

      <main className="workspace">
        <header className="workspace-header">
          <div className="workspace-header__mobile">
            <button className="icon-button" onClick={() => setMobileMenuOpen(true)} aria-label="Open menu">
              <Menu size={20} />
            </button>
            <BrandMark compact />
          </div>
          <div>
            <span className="section-kicker">Guided consultation</span>
            <h1>Medical voice assistant</h1>
          </div>
          <div className="workspace-header__actions">
            <span className="research-badge"><FlaskConical size={14} /> Research use only</span>
            <button className="header-button" onClick={() => setResearchOpen(true)}>
              <FlaskConical size={16} /> Research details
            </button>
            <button className="icon-button" onClick={resetSession} aria-label="Reset session">
              <RotateCcw size={17} />
            </button>
          </div>
        </header>

        <div className="workspace-body">
          <section className="conversation" aria-label="Medical chatbot conversation">
            <div className="conversation-scroll">
              <div className="conversation-date"><span>Current private session</span></div>
              {messages.map((message) => (
                <ConversationMessage message={message} onPrompt={processNarrative} key={message.id} />
              ))}

              {busy && (
                <div className="chat-row chat-row--assistant">
                  <span className="assistant-avatar"><AudioLines size={18} /></span>
                  <div className="processing-bubble">
                    <span className="processing-orb"><span /><span /><span /></span>
                    <div>
                      <strong>{busy}</strong>
                      <small>Local models are processing this request</small>
                    </div>
                  </div>
                </div>
              )}

              {transcriptReview && (
                <TranscriptReview
                  review={transcriptReview}
                  onChange={(text) => setTranscriptReview((item) => ({ ...item, text }))}
                  onConfirm={confirmTranscript}
                  onCancel={cancelTranscript}
                  busy={busy}
                />
              )}
              <div ref={chatEndRef} />
            </div>

            <form className="composer" onSubmit={submitDraft}>
              {awaitingClarification && (
                <div className="composer__context">
                  <HelpContext />
                  Reply naturally to the short question above. No special format is required.
                </div>
              )}
              <div className={`composer__box ${recording ? "composer__box--recording" : ""}`}>
                <button
                  type="button"
                  className="composer__control"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={Boolean(busy) || recording}
                  aria-label="Upload audio"
                >
                  <Paperclip size={19} />
                </button>
                <input ref={fileInputRef} type="file" accept="audio/*" hidden onChange={handleFile} />
                <textarea
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      submitDraft();
                    }
                  }}
                  placeholder={awaitingClarification ? "Add anything I missed, or say 'no other symptoms'..." : "Describe what you feel in your own words..."}
                  rows={1}
                  disabled={Boolean(busy) || recording}
                  aria-label="Symptom description"
                />
                <button
                  type="button"
                  className={`composer__control composer__mic ${recording ? "composer__mic--active" : ""}`}
                  onClick={recording ? stopRecording : startRecording}
                  disabled={Boolean(busy)}
                  aria-label={recording ? "Stop recording" : "Record voice symptoms"}
                >
                  {recording ? <CircleStop size={20} /> : <Mic size={20} />}
                </button>
                <button className="composer__send" type="submit" disabled={Boolean(busy) || recording || !draft.trim()} aria-label="Send message">
                  <Send size={18} />
                </button>
              </div>
              <p>
                <ShieldCheck size={13} /> This tool provides preliminary guidance, not a medical diagnosis.
              </p>
            </form>
          </section>

          <InsightsPanel
            analysis={analysis}
            health={health}
            busy={Boolean(busy)}
            hasTranscript={Boolean(transcriptResult)}
          />
        </div>
      </main>

      <ResearchPanel
        open={researchOpen}
        onClose={() => setResearchOpen(false)}
        analysis={analysis}
        transcriptResult={transcriptResult}
      />
      {researchOpen && <button className="drawer-backdrop" onClick={() => setResearchOpen(false)} aria-label="Close research panel" />}
    </div>
  );
}

function HelpContext() {
  return <Info size={14} />;
}
