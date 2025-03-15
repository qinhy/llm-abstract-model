
import base64

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
        length = (length_byte & 0x7F) if (length_byte & 0x80 == 0) else int.from_bytes(data[index:index+(length_byte & 0x7F)], 'big')
        index += (length_byte & 0x7F) if (length_byte & 0x80 != 0) else 0
        value = data[index:index+length]
        index += length
        return tag, length, value, index

    @staticmethod
    def load_key_from_pkcs8(file_path):
        with open(file_path, 'r') as file:
            key_data = ''.join(line.strip() for line in file if "BEGIN" not in line and "END" not in line)
        return base64.b64decode(key_data)

    @classmethod
    def load_public_key_from_pkcs8(cls, file_path):
        key_bytes = cls.load_key_from_pkcs8(file_path)
        index = 0
        tag, _, value, _ = cls.parse_asn1_der_element(key_bytes, index)
        if tag != 0x30:
            raise ValueError("Invalid PKCS#8 file: expected SEQUENCE")
        data = value
        index = 0
        tag, _, value, index = cls.parse_asn1_der_element(data, index)
        if tag != 0x30:
            raise ValueError("Invalid PKCS#8 file: expected SEQUENCE for algorithm identifier")
        tag, _, value, index = cls.parse_asn1_der_element(data, index)
        if tag != 0x03 or value[0] != 0x00:
            raise ValueError("Invalid PKCS#8 file: expected BIT STRING with padding")
        public_key_bytes = value[1:]
        index = 0
        tag, _, value, index = cls.parse_asn1_der_element(public_key_bytes, index)
        if tag != 0x30:
            raise ValueError("Invalid RSAPublicKey structure: expected SEQUENCE")
        rsa_key_data = value
        index = 0
        tag, _, n_bytes, index = cls.parse_asn1_der_element(rsa_key_data, index)
        n = int.from_bytes(n_bytes, 'big') if tag == 0x02 else cls.raise_invalid_structure("modulus")
        tag, _, e_bytes, _ = cls.parse_asn1_der_element(rsa_key_data, index)
        e = int.from_bytes(e_bytes, 'big') if tag == 0x02 else cls.raise_invalid_structure("exponent")
        return e, n

    @staticmethod
    def raise_invalid_structure(field):
        raise ValueError(f"Invalid RSAPublicKey structure: expected INTEGER for {field}")
    
    @classmethod
    def load_private_key_from_pkcs8(cls, file_path):
        key_bytes = cls.load_key_from_pkcs8(file_path)
        index = 0
        tag, _, value, _ = cls.parse_asn1_der_element(key_bytes, index)
        if tag != 0x30:
            raise ValueError("Invalid PKCS#8 file: expected SEQUENCE")
        data = value
        index = 0
        for expected_tag, description in [(0x02, "version"), (0x30, "algorithm identifier"), (0x04, "privateKey")]:
            tag, _, value, index = cls.parse_asn1_der_element(data, index)
            if tag != expected_tag:
                raise ValueError(f"Invalid PKCS#8 file: expected {description}")
        private_key_bytes = value
        index = 0
        tag, _, value, index = cls.parse_asn1_der_element(private_key_bytes, index)
        if tag != 0x30:
            raise ValueError("Invalid RSAPrivateKey structure: expected SEQUENCE")
        rsa_key_data = value
        index = 0
        for expected_tag, field in [(0x02, "version"), (0x02, "modulus"), (0x02, "publicExponent"), (0x02, "privateExponent")]:
            tag, _, component_bytes, index = cls.parse_asn1_der_element(rsa_key_data, index)
            if tag != expected_tag:
                cls.raise_invalid_structure(field)
            if field == "modulus":
                n = int.from_bytes(component_bytes, 'big')
            elif field == "publicExponent":
                e = int.from_bytes(component_bytes, 'big')
            elif field == "privateExponent":
                d = int.from_bytes(component_bytes, 'big')
        return d, n

    def encrypt_chunk(self, chunk):
        if not self.public_key:
            raise ValueError("Public key is required for encryption.")
        e, n = self.public_key
        chunk_int = int.from_bytes(chunk, 'big')
        encrypted_chunk_int = pow(chunk_int, e, n)
        return encrypted_chunk_int.to_bytes((n.bit_length() + 7) // 8, 'big')

    def decrypt_chunk(self, encrypted_chunk):
        if not self.private_key:
            raise ValueError("Private key is required for decryption.")
        d, n = self.private_key
        encrypted_chunk_int = int.from_bytes(encrypted_chunk, 'big')
        decrypted_chunk_int = pow(encrypted_chunk_int, d, n)
        return decrypted_chunk_int.to_bytes((n.bit_length() + 7) // 8, 'big').lstrip(b'\x00')

    def encrypt_string(self, plaintext):
        if not self.chunk_size:
            raise ValueError("Public key required for encryption.")
        plaintext_bytes = plaintext.encode('utf-8')
        encrypted_chunks = [
            base64.b64encode(self.encrypt_chunk(plaintext_bytes[i:i + self.chunk_size]))
            for i in range(0, len(plaintext_bytes), self.chunk_size)
        ]
        return b'|'.join(encrypted_chunks).decode('utf-8')

    def decrypt_string(self, encrypted_data):
        if not self.private_key:
            raise ValueError("Private key is required for decryption.")
        decrypted_chunks = [
            self.decrypt_chunk(base64.b64decode(chunk))
            for chunk in encrypted_data.split('|')
        ]
        return b''.join(decrypted_chunks).decode('utf-8')

def ex1():
    from cryptography.hazmat.primitives.asymmetric import rsa
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    public_numbers = public_key.public_numbers()
    e, n = public_numbers.e, public_numbers.n
    private_numbers = private_key.private_numbers()
    d = private_numbers.d
    public_key_tuple = (e, n)
    private_key_tuple = (d, n)
    print("Public Key:", public_key_tuple)
    print("Private Key:", private_key_tuple)
    encryptor = SimpleRSAChunkEncryptor(public_key_tuple, private_key_tuple)
    plaintext = "Hello, RSA encryption with chunking and Base64!"
    print(f"Original Plaintext:[{plaintext}]")
    encrypted_text = encryptor.encrypt_string(plaintext)
    print(f"\nEncrypted (Base64 encoded):[{encrypted_text}]")
    decrypted_text = encryptor.decrypt_string(encrypted_text)
    print(f"\nDecrypted Text:[{decrypted_text}]")

def ex2():
    from Crypto.PublicKey import RSA
    key = RSA.generate(2048)
    private_key_pem = key.export_key(format="PEM", pkcs=8)
    with open("private_key.pem", "wb") as private_file:
        private_file.write(private_key_pem)
    public_key_pem = key.publickey().export_key(format="PEM", pkcs=8)
    with open("public_key.pem", "wb") as public_file:
        public_file.write(public_key_pem)
    print("Keys have been generated and saved as 'private_key.pem' and 'public_key.pem'.")

def ex3():
    public_key_path = './tmp/public_key.pem'
    private_key_path = './tmp/private_key.pem'
    public_key = SimpleRSAChunkEncryptor.load_public_key_from_pkcs8(public_key_path)
    private_key = SimpleRSAChunkEncryptor.load_private_key_from_pkcs8(private_key_path)
    encryptor = SimpleRSAChunkEncryptor(public_key, private_key)
    plaintext = "Hello, RSA encryption with .pem support!"
    print(f"Original Plaintext:[{plaintext}]")
    encrypted_text = encryptor.encrypt_string(plaintext)
    print(f"\nEncrypted (Base64 encoded):[{encrypted_text}]")
    decrypted_text = encryptor.decrypt_string(encrypted_text)
    print(f"\nDecrypted Text:[{decrypted_text}]")

ex3()
import base64

class PEMFileReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.key_bytes = self._read_pem_file()

    def _read_pem_file(self):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
        key_data = ''.join(line.strip() for line in lines if "BEGIN" not in line and "END" not in line)
        return base64.b64decode(key_data)

    def _parse_asn1_der_element(self, data, index):
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

    def _parse_asn1_der_integer(self, data, index):
        tag, _, value, index = self._parse_asn1_der_element(data, index)
        if tag != 0x02:
            raise ValueError("Expected INTEGER")
        integer = int.from_bytes(value, byteorder='big')
        return integer, index

    def _parse_asn1_der_sequence(self, data, index):
        tag, length, value, index = self._parse_asn1_der_element(data, index)
        if tag != 0x30:
            raise ValueError("Expected SEQUENCE")
        return value, index

    def load_public_pkcs8_key(self):
        data, _ = self._parse_asn1_der_sequence(self.key_bytes, 0)
        index = 0
        _, index = self._parse_asn1_der_sequence(data, index)
        tag, _, value, index = self._parse_asn1_der_element(data, index)
        if tag != 0x03:
            raise ValueError("Expected BIT STRING")
        if value[0] != 0x00:
            raise ValueError("Invalid BIT STRING padding")
        public_key_bytes = value[1:]
        rsa_key_data, _ = self._parse_asn1_der_sequence(public_key_bytes, 0)
        index = 0
        n, index = self._parse_asn1_der_integer(rsa_key_data, index)
        e, _ = self._parse_asn1_der_integer(rsa_key_data, index)
        return e, n

    def load_private_pkcs8_key(self):
        data, _ = self._parse_asn1_der_sequence(self.key_bytes, 0)
        index = 0
        _, index = self._parse_asn1_der_integer(data, index)
        _, index = self._parse_asn1_der_sequence(data, index)
        tag, _, private_key_bytes, index = self._parse_asn1_der_element(data, index)
        if tag != 0x04:
            raise ValueError("Expected OCTET STRING")
        rsa_key_data, _ = self._parse_asn1_der_sequence(private_key_bytes, 0)
        index = 0
        _, index = self._parse_asn1_der_integer(rsa_key_data, index)
        # Continue with further parsing logic
n, index = self._parse_asn1_der_integer(rsa_key_data, index)
        e, index = self._parse_asn1_der_integer(rsa_key_data, index)
        d, _ = self._parse_asn1_der_integer(rsa_key_data, index)
        return d, n

class SimpleRSAChunkEncryptor:
    def __init__(self, public_key: tuple[int, int] = None, private_key: tuple[int, int] = None):
        self.public_key = public_key
        self.private_key = private_key
        if public_key:
            self.chunk_size = (public_key[1].bit_length() // 8)
            if self.chunk_size <= 0:
                raise ValueError("The modulus 'n' is too small. Please use a larger key size.")

    def encrypt_chunk(self, chunk: bytes):
        if not self.public_key:
            raise ValueError("Public key is required for encryption.")
        e, n = self.public_key
        chunk_int = int.from_bytes(chunk, byteorder='big')
        encrypted_chunk_int = pow(chunk_int, e, n)
        return encrypted_chunk_int.to_bytes((n.bit_length() + 7) // 8, byteorder='big')

    def decrypt_chunk(self, encrypted_chunk: bytes):
        if not self.private_key:
            raise ValueError("Private key is required for decryption.")
        d, n = self.private_key
        encrypted_chunk_int = int.from_bytes(encrypted_chunk, byteorder='big')
        decrypted_chunk_int = pow(encrypted_chunk_int, d, n)
        decrypted_chunk = decrypted_chunk_int.to_bytes((n.bit_length() + 7) // 8, byteorder='big')
        return decrypted_chunk.lstrip(b'\x00')

    def encrypt_string(self, plaintext: str, compress: bool = True) -> str:
        if not self.chunk_size:
            raise ValueError("Public key required for encryption.")
        
        data = zlib.compress(plaintext.encode('utf-8')) if compress else plaintext.encode('utf-8')
        chunk_size = self.chunk_size - 1
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        e, n = self.public_key

        encrypted_chunks = []
        for chunk in chunks:
            chunk_hex = chunk.hex()
            chunk_int = int('0x1' + chunk_hex, 16)
            encrypted_int = pow(chunk_int, e, n)
            encrypted_hex = encrypted_int.to_bytes((self.chunk_size * 2) * 4 // 8, 'big').hex()
            encrypted_base64 = base64.b64encode(bytes.fromhex(encrypted_hex)).decode('utf-8')
            encrypted_chunks.append(encrypted_base64)

        return '|'.join(encrypted_chunks)

    def decrypt_string(self, encrypted_data: str):
        if not self.private_key:
            raise ValueError("Private key required for decryption.")
        
        encrypted_chunks = encrypted_data.split('|')
        decoded_chunks = [base64.b64decode(chunk) for chunk in encrypted_chunks]
        decrypted_chunks = [self.decrypt_chunk(chunk) for chunk in decoded_chunks]
        data = b''.join(decrypted_chunks)
        
        try:
            return zlib.decompress(data).decode('utf-8')
        except zlib.error:
            return data.decode('utf-8')
def _parse_asn1_der_integer(self, data: bytes, start_index: int):
        if data[start_index] != 0x02:
            raise ValueError("Expected ASN.1 integer")
        length = data[start_index + 1]
        integer = int.from_bytes(data[start_index + 2:start_index + 2 + length], byteorder='big')
        return integer, start_index + 2 + length

    def _split_data_into_chunks(self, data: bytes, chunk_size: int):
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    def _convert_chunk_to_int(self, chunk: bytes):
        return int.from_bytes(chunk, byteorder='big')

    def _convert_int_to_chunk(self, integer: int, chunk_size: int):
        return integer.to_bytes(chunk_size, byteorder='big')