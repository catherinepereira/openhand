import { GitHubIcon } from "./icons";

interface Props {
  title: string;
  repo: string;
  modelRepo?: string;
}

export function SiteHeader({ title, repo, modelRepo }: Props) {
  return (
    <header className="flex h-16 items-baseline gap-4 border-b border-border-app bg-bg px-10 max-[700px]:px-5 max-[700px]:flex-col max-[700px]:items-start max-[700px]:py-3 max-[700px]:h-auto">
      <div className="flex items-baseline gap-2">
        <span className="text-[1.05rem] font-semibold tracking-tight whitespace-nowrap">
          {title}
        </span>
        <span aria-hidden="true">🤟</span>
      </div>
      <div className="ml-auto flex items-baseline gap-1.5 text-xs text-muted max-[700px]:ml-0">
        <GitHubIcon />
        <span>
          made by{" "}
          <Link href="https://github.com/catherinepereira">catherinepereira</Link>
          , code at <Link href={`https://github.com/catherinepereira/${repo}`}>{repo}</Link>
          {modelRepo && (
            <>
              , model at{" "}
              <Link href={`https://github.com/catherinepereira/${modelRepo}`}>{modelRepo}</Link>
            </>
          )}
        </span>
      </div>
    </header>
  );
}

function Link({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="underline underline-offset-2 hover:text-ink"
    >
      {children}
    </a>
  );
}
