type IdleCapableWindow = Window & typeof globalThis;

export class DeferredJsonStorage<T> {
  private pending: T | null = null;
  private timer = 0;
  private idleCallback = 0;

  constructor(
    private readonly key: string,
    private readonly delayMs = 400,
  ) {}

  schedule(value: T) {
    this.pending = value;
    window.clearTimeout(this.timer);
    if (this.idleCallback && "cancelIdleCallback" in window) {
      window.cancelIdleCallback(this.idleCallback);
      this.idleCallback = 0;
    }
    this.timer = window.setTimeout(() => this.queueIdleWrite(), this.delayMs);
  }

  flush = () => {
    if (this.pending === null) return;
    const value = this.pending;
    this.pending = null;
    try {
      localStorage.setItem(this.key, JSON.stringify(value));
    } catch (error) {
      console.warn(`Could not persist ${this.key}`, error);
    }
  };

  private queueIdleWrite() {
    this.timer = 0;
    const idleWindow = window as IdleCapableWindow;
    if ("requestIdleCallback" in idleWindow) {
      this.idleCallback = idleWindow.requestIdleCallback(() => {
        this.idleCallback = 0;
        this.flush();
      }, { timeout: 1_200 });
      return;
    }
    this.flush();
  }
}
