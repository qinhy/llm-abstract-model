import { Model4LLMs } from "./LLMsModel";

// RegxExtractor class
export class RegxExtractor extends Model4LLMs.AbstractObj {
  regx!: string;
  isJson: boolean = false;

  constructor(regx: string, isJson: boolean = false) {
    super();
    this.regx = regx;
    this.isJson = isJson;
  }

  public async acall(text: Promise<string>): Promise<string> {
    return this.extract(await text);
  }
  public call(text: string): string {
    return this.extract(text);
  }

  private extract(text: string): string {
    const matches = text.match(new RegExp(this.regx)) || [];
    if (!this.tryBinaryError(() => matches[1])) {
      this.logError(new Error(`Cannot match ${this.regx} in text: ${text}`));
      return text;
    }
    return this.isJson ? JSON.parse(matches[1]) : matches[1];
  }

  private tryBinaryError(fn: () => any): boolean {
    try {
      fn();
      return true;
    } catch {
      return false;
    }
  }

  private logError(error: Error): void {
    console.error(error);
  }
}

// StringTemplate class
export class StringTemplate extends Model4LLMs.AbstractObj {
  string!: string;

  constructor(string: string) {
    super();
    this.string = string;
  }

  public call(args: string[]): string {
    return this.format(this.string, args);
  }

  private format(template: string, args: string[]): string {    
    for (let index = 0; index < args.length; index++) {
        const element = args[index];
        template = template.replace("{}",element);
    }
    return template;
  }
}

// Example usage:
// const regexExtractor = new RegxExtractor("regex pattern", false);
// const result = regexExtractor.call("input text");

// const stringTemplate = new StringTemplate("Hello, {0}!");
// const formatted = stringTemplate.call("World");