// import * as fs from 'fs';
import { Buffer } from 'buffer';
import pako from 'pako';

export class PEMFileReader {
    private filePath: string;
    private keyBytes: Uint8Array;

    constructor(filePath: string) {
        this.filePath = filePath;
        this.keyBytes = this.readPemFile(filePath);
    }

    private readPemFile(path:string): Uint8Array {
        var content = path;
        // if(fs){
        //     content = fs.readFileSync(this.filePath, 'utf-8');
        // }
        const lines = content.split('\n');
        const keyData = lines
            .filter(line => !line.includes('BEGIN') && !line.includes('END'))
            .join('')
            .trim();
        return Buffer.from(keyData, 'base64');
    }

    private parseAsn1DerElement(data: Uint8Array, index: number): [number, number, Uint8Array, number] {
        const tag = data[index++];
        let length = data[index++];
        if (length & 0x80) {
            const numLengthBytes = length & 0x7F;
            length = Number(BigInt('0x' + Buffer.from(data.slice(index, index + numLengthBytes)).toString('hex')));
            index += numLengthBytes;
        }
        const value = data.slice(index, index + length);
        index += length;

        return [tag, length, value, index];
    }

    private parseAsn1DerInteger(data: Uint8Array, index: number): [BigInt, number] {
        const [tag, , value, nextIndex] = this.parseAsn1DerElement(data, index);
        if (tag !== 0x02) {
            throw new Error('Expected INTEGER');
        }
        const integer = BigInt('0x' + Buffer.from(value).toString('hex'));
        return [integer, nextIndex];
    }

    private parseAsn1DerSequence(data: Uint8Array, index: number): [Uint8Array, number] {
        const [tag, , value, nextIndex] = this.parseAsn1DerElement(data, index);
        if (tag !== 0x30) {
            throw new Error('Expected SEQUENCE');
        }
        return [value, nextIndex];
    }

    public loadPublicPkcs8Key(): [BigInt, BigInt] {
        const [data] = this.parseAsn1DerSequence(this.keyBytes, 0);
        let index = 0;

        // Skip algorithm identifier sequence
        [, index] = this.parseAsn1DerSequence(data, index);

        // Parse BIT STRING
        const [tag, , value, nextIndex] = this.parseAsn1DerElement(data, index);
        if (tag !== 0x03 || value[0] !== 0x00) {
            throw new Error('Invalid BIT STRING');
        }
        const publicKeyBytes = value.slice(1);

        // Parse RSAPublicKey sequence
        const [rsaKeyData] = this.parseAsn1DerSequence(publicKeyBytes, 0);
        index = 0;

        const [n, nextIndex2] = this.parseAsn1DerInteger(rsaKeyData, index);
        const [e] = this.parseAsn1DerInteger(rsaKeyData, nextIndex2);

        return [e, n];
    }

    public loadPrivatePkcs8Key(): [BigInt, BigInt] {
        const [data] = this.parseAsn1DerSequence(this.keyBytes, 0);
        let index = 0;

        // Skip version INTEGER
        [, index] = this.parseAsn1DerInteger(data, index);

        // Skip algorithm identifier sequence
        [, index] = this.parseAsn1DerSequence(data, index);

        // Parse privateKey OCTET STRING
        const [tag, , privateKeyBytes, nextIndex] = this.parseAsn1DerElement(data, index);
        if (tag !== 0x04) {
            throw new Error('Expected OCTET STRING');
        }

        // Parse RSAPrivateKey sequence
        const [rsaKeyData] = this.parseAsn1DerSequence(privateKeyBytes, 0);
        index = 0;

        // Skip version INTEGER
        [, index] = this.parseAsn1DerInteger(rsaKeyData, index);

        var n: BigInt;
        var e: BigInt;
        [n, index] = this.parseAsn1DerInteger(rsaKeyData, index);
        [e, index] = this.parseAsn1DerInteger(rsaKeyData, index);
        const [d] = this.parseAsn1DerInteger(rsaKeyData, index);

        return [d, n];
    }
}

export class SimpleRSAChunkEncryptor {
    private publicKey?: [BigInt, BigInt];
    private privateKey?: [BigInt, BigInt];
    private chunkSize?: number;

    constructor(publicKey?: [BigInt, BigInt], privateKey?: [BigInt, BigInt]) {
        this.publicKey = publicKey;
        this.privateKey = privateKey;
        if (publicKey) {
            this.chunkSize = Math.floor(publicKey[1].toString(2).length / 8) - 1;
            if (this.chunkSize <= 0) {
                throw new Error('The modulus "n" is too small. Please use a larger key size.');
            }
        }
    }

    private encryptChunk(chunk: Uint8Array): Uint8Array {
        if (!this.publicKey) {
            throw new Error('Public key is required for encryption.');
        }
        const [e, n] = this.publicKey;
        const chunkInt = BigInt('0x' + Buffer.from(chunk).toString('hex'));
        const encryptedChunkInt = chunkInt ** e % n;
        return Buffer.from(encryptedChunkInt.toString(16), 'hex');
    }

    powermod(base: BigInt, exp: BigInt | number, p: BigInt) {
        var result = 1n;
        while (exp !== 0n) {
          if (exp % 2n === 1n) result = result * base % p;
          base = base * base % p;
          exp >>= 1n;
        }
        return result;
    }

    private decryptChunk(encryptedChunk: Uint8Array): Uint8Array {

        if (!this.privateKey) {
            throw new Error('Private key is required for decryption.');
        }
        const [d, n] = this.privateKey;
        const encryptedChunkInt = BigInt('0x' + Buffer.from(encryptedChunk).toString('hex'));
        const decryptedChunkInt = this.powermod(encryptedChunkInt,d,n);//encryptedChunkInt ** d % n;
        return Buffer.from(decryptedChunkInt.toString(16), 'hex');
    }

    public encryptString(plaintext: string, compress: boolean = true): string {
        if (!this.chunkSize) {
            throw new Error('Public key required for encryption.');
        }

        // Compress the plaintext if requested
        const data = compress
            // ? zlib.deflateSync(Buffer.from(plaintext, 'utf-8'))
            ? Buffer.from(pako.deflate(Uint8Array.from(Buffer.from(plaintext, 'utf-8'))))
            : Buffer.from(plaintext, 'utf-8');
        
        // Split the data into chunks
        const chunks = Array.from({ length: Math.ceil(data.length / this.chunkSize) }, (_, i) =>
            data.slice(i * this.chunkSize, (i + 1) * this.chunkSize)
        );
    
        // Encrypt each chunk
        const encryptedChunks = chunks.map(chunk => this.encryptChunk(chunk));
    
        // Encode each encrypted chunk to Base64
        const encodedChunks = encryptedChunks.map(chunk => Buffer.from(chunk).toString('base64'));
    
        // Join the encoded chunks with a separator
        return encodedChunks.join('|');
    }
    

    public decryptString(encryptedData: string): string {
        if (!this.privateKey) {
            throw new Error('Private key required for decryption.');
        }
        const encryptedChunks = encryptedData.split('|');
        const decodedChunks = encryptedChunks.map(chunk => Buffer.from(chunk, 'base64'));
        const decryptedChunks = decodedChunks.map(chunk => this.decryptChunk(chunk));
        const data = Buffer.concat(decryptedChunks);
        try {
            // return zlib.inflateSync(data).toString('utf-8');
            return pako.inflate(Uint8Array.from(data), { to: 'string' });
        } catch {
            return data.toString('utf-8');
        }
    }
}

function ex3() {
    const publicKeyPath = '../tmp/public_key.pem';
    const privateKeyPath = '../tmp/private_key.pem';

    // Load keys from .pem files
    const publicKey = new PEMFileReader(publicKeyPath).loadPublicPkcs8Key();
    const privateKey = new PEMFileReader(privateKeyPath).loadPrivatePkcs8Key();
    
    // Instantiate the encryptor with the loaded keys
    const encryptor = new SimpleRSAChunkEncryptor(publicKey, privateKey);

    // Encrypt and decrypt a sample string
    const plaintext = "Hello, RSA encryption with .pem support!";
    console.log(`Original Plaintext: [${plaintext}]`);

    // Encrypt the plaintext
    const encryptedText = encryptor.encryptString(plaintext);
    console.log(`\nEncrypted (Base64 encoded): [${encryptedText}]`);

    // // Decrypt the encrypted text
    const decryptedText = encryptor.decryptString(encryptedText);
    console.log(`\nDecrypted Text: [${decryptedText}]`);
}

// ex3()