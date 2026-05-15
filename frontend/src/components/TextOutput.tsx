import { EraserIcon } from "./icons";

interface Props {
  text: string;
  onClear: () => void;
  onSpeak: () => void;
  ttsEnabled: boolean;
  speaking: boolean;
}

export function TextOutput({ text, onClear, onSpeak, ttsEnabled, speaking }: Props) {
  return (
    <div className="text-output">
      <div className="text-output-header">
        <span className="text-output-label">OUTPUT</span>
        <div className="text-output-actions">
          {ttsEnabled && (
            <button className="btn-icon" onClick={onSpeak} disabled={!text || speaking} title="Speak">
              {speaking ? "⏸" : "🔊"}
            </button>
          )}
          <button className="btn-icon" onClick={onClear} disabled={!text} title="Clear" aria-label="Clear output">
            <EraserIcon />
          </button>
        </div>
      </div>
      <p className="text-output-body">{text || <span className="muted">-</span>}</p>
    </div>
  );
}
