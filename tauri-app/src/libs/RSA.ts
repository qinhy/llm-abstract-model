// import * as fs from 'fs';
import { Buffer } from 'buffer';
import pako from 'pako';

export class PEMFileReader {
    // private filePath: string;
    private keyBytes: Uint8Array;

    constructor(filePath: string) {
        // this.filePath = filePath;
        this.keyBytes = this.readPemFile(filePath);
    }

    private readPemFile(path: string): Uint8Array {
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

    private parseAsn1DerInteger(data: Uint8Array, index: number): [bigint, number] {
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

    public loadPublicPkcs8Key(): [bigint, bigint] {
        const [data] = this.parseAsn1DerSequence(this.keyBytes, 0);
        let index = 0;

        // Skip algorithm identifier sequence
        [, index] = this.parseAsn1DerSequence(data, index);

        // Parse BIT STRING
        const [tag, , value, ] = this.parseAsn1DerElement(data, index);
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

    public loadPrivatePkcs8Key(): [bigint, bigint] {
        const [data] = this.parseAsn1DerSequence(this.keyBytes, 0);
        let index = 0;

        // Skip version INTEGER
        [, index] = this.parseAsn1DerInteger(data, index);

        // Skip algorithm identifier sequence
        [, index] = this.parseAsn1DerSequence(data, index);

        // Parse privateKey OCTET STRING
        const [tag, , privateKeyBytes,] = this.parseAsn1DerElement(data, index);
        if (tag !== 0x04) {
            throw new Error('Expected OCTET STRING');
        }

        // Parse RSAPrivateKey sequence
        const [rsaKeyData] = this.parseAsn1DerSequence(privateKeyBytes, 0);
        index = 0;

        // Skip version INTEGER
        [, index] = this.parseAsn1DerInteger(rsaKeyData, index);

        var n: bigint;
        [n, index] = this.parseAsn1DerInteger(rsaKeyData, index);
        [, index] = this.parseAsn1DerInteger(rsaKeyData, index);
        const [d] = this.parseAsn1DerInteger(rsaKeyData, index);

        return [d, n];
    }
}

export class SimpleRSAChunkEncryptor {
    private publicKey?: [bigint, bigint] | null;
    private privateKey?: [bigint, bigint] | null;
    private chunkSize: number = 0;

    constructor(publicKey?: [bigint, bigint] | null, privateKey?: [bigint, bigint] | null) {
        this.publicKey = publicKey;
        this.privateKey = privateKey;
        if (this.publicKey) {
            const n = this.publicKey[1];
            this.chunkSize = Math.floor(n.toString(2).length / 8);
            if (this.chunkSize <= 0) {
                throw new Error('The modulus "n" is too small. Please use a larger key size.');
            }
        }
    }

    powermod(base: bigint, exp: bigint | number, p: bigint): bigint {
        let e = typeof exp === 'number' ? BigInt(exp) : exp;
        let result = 1n;
        let b = base % p; // optional initial reduction

        while (e !== 0n) {
            if ((e & 1n) === 1n) {
                result = (result * b) % p;
            }
            b = (b * b) % p;
            e >>= 1n;
        }
        return result;
    }

    public encryptString(plaintext: string, compress: boolean = true): string {
        // Ensure the chunk size is defined
        if (!this.chunkSize) {
            throw new Error('Public key required for encryption.');
        }    
        const plainEncoder = new TextEncoder();
    
        // Step 1: Compress the plaintext if requested, otherwise encode it as-is
        const data = compress
            ? Buffer.from(pako.deflate(Uint8Array.from(Buffer.from(plaintext, 'utf-8'))))
            // ? zlib.deflateSync(Buffer.from(plaintext, 'utf-8'))
            : plainEncoder.encode(plaintext);
    
        const chunkSize = this.chunkSize - 1; // for making it starts without 0 !
    
        // Step 2: Split the data into chunks of the specified size
        const chunks = Array.from(
            { length: Math.ceil(data.length / chunkSize) },
            (_, i) => data.subarray(i * chunkSize, (i + 1) * chunkSize)
        );
    
        // Step 3: Encrypt each chunk using a series of transformation steps     
        if (!this.publicKey) {
            throw new Error('Public key required for encryption.');
        }   
        const [e, n] = this.publicKey;
        const encryptedChunks = chunks
            // a. Convert chunk to hex
            .map(chunk => Buffer.from(chunk).toString('hex'))
            // b. Convert hex string to bigint, make it starts without 0 !
            .map(chunkHex => BigInt('0x1' + chunkHex))
            // c. Encrypt the bigint using the public key
            .map(chunkInt => this.powermod(chunkInt, e, n))
            // d. Convert the encrypted bigint to a padded hex string
            .map(encryptedInt => encryptedInt.toString(16).padStart(this.chunkSize * 2, '0'))
            // e. Encode the hex string to Base64
            .map(encryptedHex => Buffer.from(encryptedHex, 'hex').toString('base64'));
    
        // Step 4: Join all the encrypted Base64-encoded chunks with a separator
        return encryptedChunks.join('|');
    }
    public decryptString(encryptedData: string): string {
        if (!this.privateKey) {
            throw new Error('Private key required for decryption.');
        }
        const [d, n] = this.privateKey; // Destructure private key components once
    
        const encryptedChunks = encryptedData.split('|');
    
        // Step 1: Decode Base64 chunks to Buffers
        const decryptedChunks = encryptedChunks
                .map(chunk => Buffer.from(chunk, 'base64'))
                // Step 2: Convert Buffers to hex strings
                .map(buffer => buffer.toString('hex'))
                // Step 3: Convert hex strings to bigints
                .map(hex => BigInt('0x' + hex))
                // Step 4: Decrypt bigints using the private key
                .map(chunkInt => this.powermod(chunkInt, d, n))
                // Step 5: Convert decrypted bigints to hex strings
                .map(chunkInt => chunkInt.toString(16))            
                // Step 6: Verify and slice hex strings, then convert to Buffers
                .map(hex => (hex.at(0) === '1' ? hex.slice(1) : 
                        (() => { throw new Error('decryptChunkHex must start with 0x1!'); })()))
                .map(slicedHex => Buffer.from(slicedHex, 'hex'));
    
        // Step 7: Concatenate Buffers
        const data = Buffer.concat(decryptedChunks);
    
        // Step 8: Decode the concatenated data
        const plainDecoder = new TextDecoder('utf-8', { fatal: true });
        try {
            return plainDecoder.decode(data); // Try decoding as UTF-8
        } catch {
            try {
                return pako.inflate(Uint8Array.from(data), { to: 'string' });
                // return zlib.inflateSync(data).toString('utf-8'); // Attempt decompression if decoding fails
            } catch {
                throw new Error('Failed to decode data after all attempts.');
            }
        }
    }    
}