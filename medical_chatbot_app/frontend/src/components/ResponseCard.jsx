import {
  AlertCircle,
  ArrowUpRight,
  BookOpen,
  CheckCircle2,
  ClipboardList,
  HeartPulse,
  HelpCircle,
  ShieldAlert,
  Stethoscope,
} from "lucide-react";

const TYPE_META = {
  emergency: {
    icon: ShieldAlert,
    eyebrow: "Emergency action",
    tone: "danger",
  },
  urgent: {
    icon: AlertCircle,
    eyebrow: "Prompt assessment",
    tone: "urgent",
  },
  clarification: {
    icon: HelpCircle,
    eyebrow: "A few details first",
    tone: "clarify",
  },
  routine_prediction: {
    icon: HeartPulse,
    eyebrow: "Research model result",
    tone: "routine",
  },
  no_prediction: {
    icon: Stethoscope,
    eyebrow: "Professional assessment",
    tone: "neutral",
  },
};

function GuidanceList({ icon: Icon, title, items, className = "" }) {
  if (!items?.length) return null;
  return (
    <section className={`guidance-block ${className}`}>
      <h4>
        <Icon size={17} />
        {title}
      </h4>
      <ul>
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </section>
  );
}

function UnderstoodSymptoms({ extraction }) {
  if (!extraction) return null;
  const present = extraction.present_symptoms || [];
  const uncertain = extraction.uncertain_symptoms || [];
  const negated = extraction.negated_symptoms || [];
  if (!present.length && !uncertain.length && !negated.length) return null;

  return (
    <section className="understood-panel">
      <div className="understood-panel__head">
        <CheckCircle2 size={16} />
        <div>
          <strong>What I understood</strong>
          <span>Correct me naturally if anything below is wrong.</span>
        </div>
      </div>
      <div className="understood-panel__tags">
        {present.map((symptom) => (
          <span className="understood-tag" key={`present-${symptom}`}>{symptom}</span>
        ))}
        {uncertain.map((symptom) => (
          <span className="understood-tag understood-tag--uncertain" key={`uncertain-${symptom}`}>
            Maybe: {symptom}
          </span>
        ))}
      </div>
      {negated.length > 0 && (
        <p>Not currently reported: {negated.slice(0, 5).join(", ")}</p>
      )}
    </section>
  );
}

export default function ResponseCard({ response, extraction }) {
  if (!response) return null;
  const meta = TYPE_META[response.response_type] || TYPE_META.no_prediction;
  const Icon = meta.icon;
  const prediction = response.displayed_prediction;

  return (
    <article className={`response-card response-card--${meta.tone}`}>
      <header className="response-card__header">
        <span className="response-card__icon">
          <Icon size={21} />
        </span>
        <div>
          <span className="response-card__eyebrow">{meta.eyebrow}</span>
          <h3>{response.headline}</h3>
        </div>
      </header>

      <p className="response-card__message">{response.message}</p>

      <UnderstoodSymptoms extraction={extraction} />

      {prediction && (
        <div className="prediction-panel">
          <div>
            <span>Closest dataset class</span>
            <strong>{prediction.disease_class}</strong>
          </div>
          <div className="prediction-panel__score">
            <strong>{Math.round(prediction.model_score * 100)}%</strong>
            <span>model score</span>
          </div>
          <p>{prediction.interpretation}</p>
        </div>
      )}

      {response.clarification_questions?.length > 0 && (
        <section className="question-stack">
          {response.clarification_questions.map((question, index) => (
            <div className="question-card" key={question.question_id}>
              <span>{String(index + 1).padStart(2, "0")}</span>
              <div>
                <p>{question.question}</p>
                <small>
                  {question.safety_related ? "Safety check" : question.category.replaceAll("_", " ")}
                </small>
              </div>
            </div>
          ))}
        </section>
      )}

      <div className="guidance-grid">
        <GuidanceList
          icon={ClipboardList}
          title="Recommended next steps"
          items={response.recommended_actions}
        />
        <GuidanceList
          icon={CheckCircle2}
          title="General self-care"
          items={response.self_care_guidance}
        />
      </div>

      <GuidanceList
        icon={ShieldAlert}
        title="Get more help if"
        items={response.safety_net_guidance}
        className="guidance-block--safety"
      />

      {response.sources?.length > 0 && (
        <details className="source-list">
          <summary>
            <BookOpen size={16} />
            Guidance sources ({response.sources.length})
          </summary>
          <div>
            {response.sources.map((source) => (
              <a href={source.url} target="_blank" rel="noreferrer" key={source.source_id}>
                {source.title}
                <ArrowUpRight size={14} />
              </a>
            ))}
          </div>
        </details>
      )}

      <footer>{response.disclaimer}</footer>
    </article>
  );
}
