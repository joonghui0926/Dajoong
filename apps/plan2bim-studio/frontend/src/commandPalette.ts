export interface StudioCommand {
  id: string;
  label: string;
  group: string;
  aliases?: string[];
  shortcut?: string;
  enabled?: boolean;
  disabledReason?: string;
  run: () => void;
}

export interface RankedStudioCommand {
  command: StudioCommand;
  recent: boolean;
  score: number;
}

const MAX_RECENT_COMMANDS = 7;

function normalize(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function subsequenceScore(haystack: string, needle: string): number | null {
  let cursor = 0;
  let first = -1;
  let last = -1;
  for (const character of needle) {
    const next = haystack.indexOf(character, cursor);
    if (next < 0) return null;
    if (first < 0) first = next;
    last = next;
    cursor = next + 1;
  }
  return 160 - first * 2 - (last - first - needle.length);
}

function commandSearchScore(command: StudioCommand, query: string): number | null {
  const normalizedQuery = normalize(query);
  if (!normalizedQuery) return 0;
  const label = normalize(command.label);
  const group = normalize(command.group);
  const aliases = normalize((command.aliases ?? []).join(" "));
  const searchable = `${label} ${group} ${aliases}`.trim();
  const searchableWords = searchable.split(" ").filter(Boolean);
  const fuzzyWordScore = (token: string) => {
    const scores = searchableWords
      .map((word) => subsequenceScore(word, token))
      .filter((score): score is number => score !== null);
    return scores.length ? Math.max(...scores) : null;
  };
  const tokens = normalizedQuery.split(" ").filter(Boolean);
  if (!tokens.every((token) => searchable.includes(token) || fuzzyWordScore(token) !== null)) {
    return null;
  }

  let score = 0;
  if (label === normalizedQuery) score += 1000;
  if (label.startsWith(normalizedQuery)) score += 700;
  if (label.includes(normalizedQuery)) score += 500;
  if (group.startsWith(normalizedQuery)) score += 180;
  if (aliases.includes(normalizedQuery)) score += 260;
  for (const token of tokens) {
    if (label.split(" ").some((word) => word.startsWith(token))) score += 140;
    else if (label.includes(token)) score += 95;
    else if (aliases.includes(token)) score += 70;
    else score += fuzzyWordScore(token) ?? 0;
  }
  return score;
}

export function sanitizeRecentCommandIds(value: unknown, knownIds?: Set<string>): string[] {
  if (!Array.isArray(value)) return [];
  const result: string[] = [];
  for (const item of value) {
    if (typeof item !== "string" || !item || result.includes(item)) continue;
    if (knownIds && !knownIds.has(item)) continue;
    result.push(item);
    if (result.length === MAX_RECENT_COMMANDS) break;
  }
  return result;
}

export function recordRecentCommand(recentIds: string[], commandId: string): string[] {
  return [commandId, ...recentIds.filter((id) => id !== commandId)].slice(0, MAX_RECENT_COMMANDS);
}

export function rankStudioCommands(
  commands: StudioCommand[],
  query: string,
  recentIds: string[],
): RankedStudioCommand[] {
  const recentIndex = new Map(recentIds.map((id, index) => [id, index]));
  const normalizedQuery = normalize(query);
  return commands
    .map((command, originalIndex) => {
      const searchScore = commandSearchScore(command, normalizedQuery);
      if (searchScore === null) return null;
      const recency = recentIndex.has(command.id) ? recentIndex.size - (recentIndex.get(command.id) ?? 0) : 0;
      return {
        command,
        recent: recentIndex.has(command.id),
        score: searchScore + (normalizedQuery ? recency * 2 : recency * 100) - originalIndex / 1000,
      };
    })
    .filter((item): item is RankedStudioCommand => item !== null)
    .sort((left, right) => {
      if (right.score !== left.score) return right.score - left.score;
      if ((left.command.enabled ?? true) !== (right.command.enabled ?? true)) {
        return (left.command.enabled ?? true) ? -1 : 1;
      }
      return left.command.label.localeCompare(right.command.label);
    });
}

export function nextCommandIndex(
  current: number,
  direction: "next" | "previous" | "first" | "last",
  length: number,
): number {
  if (length <= 0) return -1;
  if (direction === "first") return 0;
  if (direction === "last") return length - 1;
  if (direction === "next") return current < 0 ? 0 : (current + 1) % length;
  return current <= 0 ? length - 1 : current - 1;
}
