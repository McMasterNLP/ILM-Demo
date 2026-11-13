import { Link, NavLink, useLocation } from "react-router-dom";

import Logo from "./Logo";
import { useLang, type Lang } from "../lib/i18n";

const linkClass = ({ isActive }: { isActive: boolean }) =>
  `text-sm transition-colors ${
    isActive ? "font-medium text-brand" : "text-ink-soft hover:text-ink"
  }`;

function LangToggle() {
  const { lang, setLang } = useLang();

  const segment = (value: Lang, label: string) => (
    <button
      key={value}
      type="button"
      onClick={() => setLang(value)}
      aria-pressed={lang === value}
      className={`rounded-full px-3 py-1 text-sm transition-colors ${
        lang === value
          ? "border border-brand bg-white font-medium text-brand"
          : "border border-transparent text-ink-soft hover:text-ink"
      }`}
    >
      {label}
    </button>
  );

  return (
    <div className="inline-flex items-center gap-0.5 rounded-full border border-hair bg-white p-0.5">
      {segment("en", "English")}
      {segment("ar", "العربية")}
    </div>
  );
}

export default function Header() {
  const { t } = useLang();
  const { pathname } = useLocation();

  const onAMap = pathname === "/map" || pathname.endsWith("/map");
  const onAQuiz = pathname === "/quiz" || pathname.startsWith("/quiz/");

  return (
    <header className="border-b border-hair bg-white">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between gap-4 px-6 py-4">
        <Link to="/" className="flex items-center gap-3">
          <Logo className="h-7 w-7" />
          <span className="text-base font-semibold tracking-tight">{t("brand")}</span>
        </Link>

        <div className="flex items-center gap-5">
          <nav className="flex items-center gap-5">
            <NavLink to="/" end className={linkClass}>
              {t("navStories")}
            </NavLink>
            <Link to="/map" className={linkClass({ isActive: onAMap })}>
              {t("navStoryMap")}
            </Link>
            <Link to="/quiz" className={linkClass({ isActive: onAQuiz })}>
              {t("navQuiz")}
            </Link>
            <NavLink to="/progress" className={linkClass}>
              {t("navProgress")}
            </NavLink>
          </nav>
          <LangToggle />
        </div>
      </div>
    </header>
  );
}
