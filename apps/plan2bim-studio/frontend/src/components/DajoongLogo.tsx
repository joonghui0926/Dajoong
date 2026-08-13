interface DajoongLogoProps {
  compact?: boolean;
  inverse?: boolean;
}

export function DajoongLogo({ compact = false, inverse = false }: DajoongLogoProps) {
  return (
    <span className={`${compact ? "dajoong-logo compact" : "dajoong-logo"}${inverse ? " inverse" : ""}`} aria-label="Dajoong">
      <img src="/brand/dajoong-logo-mark.svg" alt="" aria-hidden="true" width="580" height="651" decoding="async" />
      {!compact ? <span>Dajoong</span> : null}
    </span>
  );
}
