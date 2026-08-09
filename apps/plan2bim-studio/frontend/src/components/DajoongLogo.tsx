interface DajoongLogoProps {
  compact?: boolean;
  inverse?: boolean;
}

export function DajoongLogo({ compact = false, inverse = false }: DajoongLogoProps) {
  return (
    <span className={`${compact ? "dajoong-logo compact" : "dajoong-logo"}${inverse ? " inverse" : ""}`} aria-label="Dajoong">
      <img src="/brand/dajoong-logo-mark-512.png" alt="" aria-hidden="true" />
      {!compact ? <span>Dajoong</span> : null}
    </span>
  );
}
