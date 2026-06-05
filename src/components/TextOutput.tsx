import { EraserIcon } from "./icons";

interface Props {
  text: string;
  onClear: () => void;
}

export function TextOutput({ text, onClear }: Props) {
  return (
    <div className="border-border-app bg-bg flex w-full max-w-[min(760px,96%)] flex-col gap-2 rounded-xl border-[1.5px] px-[1.1rem] py-[0.85rem]">
      <div className="flex items-center justify-between">
        <span className="label-caps">OUTPUT</span>
        <button
          className="border-border-app text-muted hover:enabled:bg-surface hover:enabled:text-ink flex h-7 w-7 items-center justify-center rounded-md border-[1.5px] text-[0.8rem] transition-colors disabled:cursor-default disabled:opacity-35"
          onClick={onClear}
          disabled={!text}
          title="Clear"
          aria-label="Clear output"
        >
          <EraserIcon />
        </button>
      </div>
      <p className="min-h-[1.6em] text-[1.1rem] tracking-wider">
        {text || <span className="text-muted">-</span>}
      </p>
    </div>
  );
}
