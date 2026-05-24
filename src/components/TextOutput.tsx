import { EraserIcon } from "./icons";

interface Props {
  text: string;
  onClear: () => void;
}

export function TextOutput({ text, onClear }: Props) {
  return (
    <div className="flex w-full max-w-[min(760px,96%)] flex-col gap-2 rounded-xl border-[1.5px] border-border-app bg-bg px-[1.1rem] py-[0.85rem]">
      <div className="flex items-center justify-between">
        <span className="label-caps">OUTPUT</span>
        <button
          className="flex h-7 w-7 items-center justify-center rounded-md border-[1.5px] border-border-app text-[0.8rem] text-muted transition-colors hover:enabled:bg-surface hover:enabled:text-ink disabled:opacity-35 disabled:cursor-default"
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
