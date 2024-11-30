# from https://github.com/qinhy/singleton-key-value-storage.git
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

class SimpleRSAChunkEncryptor:
    def __init__(self, public_key=None, private_key=None):
        self.public_key = public_key
        self.private_key = private_key
        self.chunk_size = self.calculate_chunk_size(public_key) if public_key else None

    @staticmethod
    def calculate_chunk_size(public_key):
        chunk_size = (public_key[1].bit_length() // 8) - 1
        if chunk_size <= 0:
            raise ValueError("The modulus 'n' is too small. Please use a larger key size.")
        return chunk_size

    @staticmethod
    def parse_asn1_der_element(data, index):
        tag = data[index]
        index += 1

        length_byte = data[index]
        index += 1
        if length_byte & 0x80 == 0:
            length = length_byte & 0x7F
        else:
            num_length_bytes = length_byte & 0x7F
            length = int.from_bytes(data[index:index+num_length_bytes], byteorder='big')
            index += num_length_bytes

        value = data[index:index+length]
        index += length

        return tag, length, value, index

    @staticmethod
    def load_public_key_from_pkcs8(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        key_data = ''.join(line.strip() for line in lines if "BEGIN" not in line and "END" not in line)
        key_bytes = base64.b64decode(key_data)

        index = 0
        tag, length, value, index = SimpleRSAChunkEncryptor.parse_asn1_der_element(key_bytes, index)
        if tag != 0x30:
            raise ValueError("Invalid PKCS#8 file: expected SEQUENCE")

        data = value
        index = 0
        tag, length, value, index = SimpleRSAChunkEncryptor.parse_asn1_der_element(data, index)
        if tag != 0x30:
            raise ValueError("Invalid PKCS#8 file: expected SEQUENCE for algorithm identifier")

        tag, length, value, index = SimpleRSAChunkEncryptor.parse_asn1_der_element(data, index)
        if tag != 0x03:
            raise ValueError("Invalid PKCS#8 file: expected BIT STRING")

        if value[0] != 0x00:
            raise ValueError("Invalid BIT STRING padding")
        public_key_bytes = value[1:]

        index = 0
        tag, length, value, index = SimpleRSAChunkEncryptor.parse_asn1_der_element(public_key_bytes, index)
        if tag != 0x30:
            raise ValueError("Invalid RSAPublicKey structure: expected SEQUENCE")

        rsa_key_data = value
        index = 0
        tag, length, n_bytes, index = SimpleRSAChunkEncryptor.parse_asn1_der_element(rsa_key_data, index)
        if tag != 0x02:
            raise ValueError("Invalid RSAPublicKey structure: expected INTEGER for modulus")
        n = int.from_bytes(n_bytes, byteorder='big')

        tag, length, e_bytes, index = SimpleRSAChunkEncryptor.parse_asn1_der_element(rsa_key_data, index)
        if tag != 0x02:
            raise ValueError("Invalid RSAPublicKey structure: expected INTEGER for exponent")
        e = int.from_bytes(e_bytes, byteorder='big')

        return (e, n)

    @staticmethod
    def load_private_key_from_pkcs8(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        key_data = ''.join(line.strip() for line in lines if "BEGIN" not in line and "END" not in line)
        key_bytes = base64.b64decode(key_data)

        index = 0
        tag, length, value, index = SimpleRSAChunkEncryptor.parse_asn1_der_element(key_bytes, index)
        if tag != 0x30:
            raise ValueError("Invalid PKCS#8 file: expected SEQUENCE")

        data = value
        index = 0
        tag, length, version_bytes, index = SimpleRSAChunkEncryptor.parse_asn1_der_element(data, index)
        if tag != 0x02:
            raise ValueError("Invalid PKCS#8 file: expected INTEGER for version")

        tag, length, value, index = SimpleRSAChunkEncryptor.parse_asn1_der_element(data, index)
        if tag != 0x30:
            raise ValueError("Invalid PKCS#8 file: expected SEQUENCE for algorithm identifier")

        tag, length, private_key_bytes, index = SimpleRSAChunkEncryptor.parse_asn1_der_element(data, index)
        if tag != 0x04:
            raise ValueError("Invalid PKCS#8 file: expected OCTET STRING for privateKey")

        index = 0
        tag, length, value, index = SimpleRSAChunkEncryptor.parse_asn1_der_element(private_key_bytes, index)
        if tag != 0x30:
            raise ValueError("Invalid RSAPrivateKey structure: expected SEQUENCE")

        rsa_key_data = value
        index = 0
        tag, length, version_bytes, index = SimpleRSAChunkEncryptor.parse_asn1_der_element(rsa_key_data, index)
        if tag != 0x02:
            raise ValueError("Invalid RSAPrivateKey structure: expected INTEGER for version")

        tag, length, n_bytes, index = SimpleRSAChunkEncryptor.parse_asn1_der_element(rsa_key_data, index)
        if tag != 0x02:
            raise ValueError("Invalid RSAPrivateKey structure: expected INTEGER for modulus")
        n = int.from_bytes(n_bytes, byteorder='big')

        tag, length, e_bytes, index = SimpleRSAChunkEncryptor.parse_asn1_der_element(rsa_key_data, index)
        if tag != 0x02:
            raise ValueError("Invalid RSAPrivateKey structure: expected INTEGER for publicExponent")
        e = int.from_bytes(e_bytes, byteorder='big')

        tag, length, d_bytes, index = SimpleRSAChunkEncryptor.parse_asn1_der_element(rsa_key_data, index)
        if tag != 0x02:
            raise ValueError("Invalid RSAPrivateKey structure: expected INTEGER for privateExponent")
        d = int.from_bytes(d_bytes, byteorder='big')

        return (d, n)

    def encrypt_chunk(self, chunk):
        if not self.public_key:
            raise ValueError("Public key is required for encryption.")
        e, n = self.public_key
        chunk_int = int.from_bytes(chunk, byteorder='big')
        encrypted_chunk_int = pow(chunk_int, e, n)
        return encrypted_chunk_int.to_bytes((n.bit_length() + 7) // 8, byteorder='big')

    def decrypt_chunk(self, encrypted_chunk):
        if not self.private_key:
            raise ValueError("Private key is required for decryption.")
        d, n = self.private_key
        encrypted_chunk_int = int.from_bytes(encrypted_chunk, byteorder='big')
        decrypted_chunk_int = pow(encrypted_chunk_int, d, n)
        decrypted_chunk = decrypted_chunk_int.to_bytes((n.bit_length() + 7) // 8, byteorder='big')
        return decrypted_chunk.lstrip(b'\x00')

    def encrypt_string(self, plaintext):
        if not self.chunk_size:
            raise ValueError("Public key required for encryption.")
        encrypted_chunks = []
        plaintext_bytes = plaintext.encode('utf-8')

        for i in range(0, len(plaintext_bytes), self.chunk_size):
            chunk = plaintext_bytes[i:i + self.chunk_size]
            encrypted_chunk = self.encrypt_chunk(chunk)
            encrypted_chunks.append(base64.b64encode(encrypted_chunk))

        return b'|'.join(encrypted_chunks).decode('utf-8')

    def decrypt_string(self, encrypted_data):
            if not self.private_key:
                raise ValueError("Private key is required for decryption.")
            encrypted_chunks = encrypted_data.split('|')
            decrypted_chunks = [
                self.decrypt_chunk(base64.b64decode(chunk))
                for chunk in encrypted_chunks
            ]
            return b''.join(decrypted_chunks).decode('utf-8')
    
    def encrypt_string(self, plaintext: str, max_workers: int = 8) -> str:
        if not self.chunk_size:
            raise ValueError("Public key required for encryption.")
        text_bytes = plaintext.encode('utf-8')
        chunks = [text_bytes[i:i + self.chunk_size] for i in range(0, len(text_bytes), self.chunk_size)]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map futures with their original index
            futures = {executor.submit(self.encrypt_chunk, chunk): index for index, chunk in enumerate(chunks)}
            encrypted_chunks = [None] * len(chunks)
            for future in futures:
                index = futures[future]
                encrypted_chunks[index] = future.result()

        # Encode and join encrypted chunks
        encoded_chunks = [base64.b64encode(chunk) for chunk in encrypted_chunks]
        return '|'.join(chunk.decode('utf-8') for chunk in encoded_chunks)

    def decrypt_string(self, encrypted_data: str, max_workers: int = 8) -> str:
        if not self.private_key:
            raise ValueError("Private key required for decryption.")
        encoded_chunks = encrypted_data.split('|')
        decoded_chunks = [base64.b64decode(chunk) for chunk in encoded_chunks]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map futures with their original index
            futures = {executor.submit(self.decrypt_chunk, chunk): index for index, chunk in enumerate(decoded_chunks)}
            decrypted_chunks = [None] * len(decoded_chunks)
            for future in futures:
                index = futures[future]
                decrypted_chunks[index] = future.result()

        # Combine decrypted chunks
        return b''.join(decrypted_chunks).decode('utf-8')

# Example Usage
def ex1():
    # Example RSA key components (these are just sample values, not secure for actual use)
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization

    # Generate a 2048-bit RSA private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Get public key from the private key
    public_key = private_key.public_key()

    # Extract public exponent (e) and modulus (n) from public key
    public_numbers = public_key.public_numbers()
    e = public_numbers.e
    n = public_numbers.n

    # Extract private exponent (d) and modulus (n) from private key
    private_numbers = private_key.private_numbers()
    d = private_numbers.d

    # Now we have public and private key tuples as (e, n) and (d, n)
    public_key_tuple = (e, n)
    private_key_tuple = (d, n)

    print("Public Key:", public_key_tuple)
    print("Private Key:", private_key_tuple)

    # Instantiate the encryptor with the public and private key
    encryptor = SimpleRSAChunkEncryptor(public_key_tuple, private_key_tuple)

    # Encrypt a sample plaintext
    plaintext = "Hello, RSA encryption with chunking and Base64!"
    print(f"Original Plaintext:[{plaintext}]")

    # Encrypt the plaintext
    encrypted_text = encryptor.encrypt_string(plaintext)
    print(f"\nEncrypted (Base64 encoded):[{encrypted_text}]")

    # Decrypt the encrypted text
    decrypted_text = encryptor.decrypt_string(encrypted_text)
    print(f"\nDecrypted Text:[{decrypted_text}]")


def ex2():
    from Crypto.PublicKey import RSA

    # Generate a 2048-bit RSA key pair
    key = RSA.generate(2048)

    # Export the private key in PKCS#8 format
    private_key_pem = key.export_key(format="PEM", pkcs=8)
    with open("private_key.pem", "wb") as private_file:
        private_file.write(private_key_pem)

    # Export the public key in PKCS#8 format
    public_key_pem = key.publickey().export_key(format="PEM", pkcs=8)
    with open("public_key.pem", "wb") as public_file:
        public_file.write(public_key_pem)

    print("Keys have been generated and saved as 'private_key.pem' and 'public_key.pem'.")


def ex3():
    # Load keys from .pem files
    public_key_path = './tmp/public_key.pem'
    private_key_path = './tmp/private_key.pem'

    public_key = SimpleRSAChunkEncryptor.load_public_key_from_pkcs8(public_key_path)
    private_key = SimpleRSAChunkEncryptor.load_private_key_from_pkcs8(private_key_path)

    # Instantiate the encryptor with the loaded keys
    encryptor = SimpleRSAChunkEncryptor(public_key, private_key)

    # Encrypt and decrypt a sample string
    plaintext = "Hello, RSA encryption with .pem support!"
    print(f"Original Plaintext:[{plaintext}]")

    # Encrypt the plaintext
    encrypted_text = encryptor.encrypt_string(plaintext)
    print(f"\nEncrypted (Base64 encoded):[{encrypted_text}]")

    # Decrypt the encrypted text
    decrypted_text = encryptor.decrypt_string(encrypted_text)
    print(f"\nDecrypted Text:[{decrypted_text}]")