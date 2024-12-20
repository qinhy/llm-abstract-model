import fs from 'fs';
export class TextFile {
    filePath: string;
    chunkLines: number;
    overlapLines: number;
    currentPosition: number;
    lineCount?: number;
    private _file: fs.ReadStream | null = null;
    private _fileBuffer: string[] = [];
    private _currentChunk: string[] | null = null;

    constructor(filePath: string, chunkLines: number = 1000, overlapLines: number = 100) {
        if (chunkLines <= 0) {
            throw new Error("chunkLines must be greater than 0");
        }
        if (overlapLines < 0) {
            throw new Error("overlapLines must be non-negative");
        }

        this.filePath = filePath;
        this.chunkLines = chunkLines;
        this.overlapLines = overlapLines;
        this.currentPosition = 0;
        this.lineCount = 0;

        this._file = fs.createReadStream(this.filePath, { encoding: 'utf-8' });
        this._calculateTotalLines();
    }

    private _calculateTotalLines(): void {
        const fileContent = fs.readFileSync(this.filePath, 'utf-8');
        this.lineCount = fileContent.split('\n').length;
    }

    private _resetFile(): void {
        if (this._file) {
            this._file.close();
        }
        this._file = fs.createReadStream(this.filePath, { encoding: 'utf-8' });
        this.currentPosition = 0;
    }

    readChunk(): string[] | null {
        if (!this.lineCount || this.currentPosition >= this.lineCount) {
            return null; // End of file
        }

        const startLine = Math.max(this.currentPosition - this.overlapLines, 0);
        const endLine = Math.min(this.currentPosition + this.chunkLines, this.lineCount);

        if (startLine > 0) {
            this._resetFile();
            let lineCounter = 0;
            const reader = fs.createReadStream(this.filePath, { encoding: 'utf-8' });
            reader.on('data', (chunk:string) => {
                this._fileBuffer.push(chunk);
                const lines = chunk.split('\n');
                for (const line of lines) {
                    if (lineCounter >= startLine) break;
                    lineCounter++;
                }
            });
        }

        const chunk: string[] = [];
        let lineCounter = startLine;
        const fileContent = fs.readFileSync(this.filePath, 'utf-8');
        const lines = fileContent.split('\n');

        for (let i = startLine; i < endLine; i++) {
            if (!lines[i]) break;
            chunk.push(lines[i]);
            lineCounter++;
        }

        this.currentPosition = endLine;
        return chunk;
    }

    [Symbol.iterator]() {
        this._resetFile();
        this._currentChunk = this.readChunk();
        return this;
    }

    next(): IteratorResult<string[]> {
        if (!this._currentChunk) {
            this._currentChunk = this.readChunk();
        }

        if (!this._currentChunk || this._currentChunk.length === 0) {
            return { done: true, value: null as any };
        }

        const chunk = this._currentChunk;
        this._currentChunk = this.readChunk();
        return { done: false, value: chunk };
    }

    close(): void {
        if (this._file) {
            this._file.close();
        }
    }
}