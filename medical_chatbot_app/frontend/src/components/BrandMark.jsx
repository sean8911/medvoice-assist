import { AudioLines, Plus } from "lucide-react";

export default function BrandMark({ compact = false }) {
  return (
    <div className={`brand-mark ${compact ? "brand-mark--compact" : ""}`}>
      <div className="brand-mark__icon" aria-hidden="true">
        <AudioLines size={21} strokeWidth={2.3} />
        <Plus className="brand-mark__plus" size={10} strokeWidth={3} />
      </div>
      <div>
        <strong>MedVoice</strong>
        {!compact && <span>Assist</span>}
      </div>
    </div>
  );
}
