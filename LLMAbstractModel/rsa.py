# from https://github.com/qinhy/singleton-key-value-storage.git
import base64
import zlib

class PEMFileReader:    
    def __init__(self, file_path):
        self.file_path = file_path
        self.key_bytes = self._read_pem_file()

    def _read_pem_file(self):
        """Read and decode a PEM file."""
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
        key_data = ''.join(line.strip() for line in lines if "BEGIN" not in line and "END" not in line)
        return base64.b64decode(key_data)

    def _parse_asn1_der_element(self, data, index):
        """Parse an ASN.1 DER element starting at the given index."""
        tag = data[index]
        index += 1

        # Parse length
        length_byte = data[index]
        index += 1
        if length_byte & 0x80 == 0:
            # Short form length
            length = length_byte & 0x7F
        else:
            # Long form length
            num_length_bytes = length_byte & 0x7F
            length = int.from_bytes(data[index:index+num_length_bytes], byteorder='big')
            index += num_length_bytes

        value = data[index:index+length]
        index += length

        return tag, length, value, index

    def _parse_asn1_der_integer(self, data, index):
        """Parse an ASN.1 DER INTEGER starting at the given index."""
        tag, _, value, index = self._parse_asn1_der_element(data, index)
        if tag != 0x02:
            raise ValueError("Expected INTEGER")
        integer = int.from_bytes(value, byteorder='big')
        return integer, index

    def _parse_asn1_der_sequence(self, data, index):
        """Parse an ASN.1 DER SEQUENCE starting at the given index."""
        tag, length, value, index = self._parse_asn1_der_element(data, index)
        if tag != 0x30:
            raise ValueError("Expected SEQUENCE")
        return value, index

    def load_public_pkcs8_key(self):
        """Load an RSA public key from a PKCS#8 PEM file."""
        data, _ = self._parse_asn1_der_sequence(self.key_bytes, 0)
        index = 0

        # Parse algorithm identifier SEQUENCE (skip it)
        _, index = self._parse_asn1_der_sequence(data, index)

        # Parse BIT STRING
        tag, _, value, index = self._parse_asn1_der_element(data, index)
        if tag != 0x03:
            raise ValueError("Expected BIT STRING")
        if value[0] != 0x00:
            raise ValueError("Invalid BIT STRING padding")
        public_key_bytes = value[1:]  # Skip the first byte

        # Parse the RSAPublicKey SEQUENCE
        rsa_key_data, _ = self._parse_asn1_der_sequence(public_key_bytes, 0)
        index = 0

        # Parse modulus (n) and exponent (e)
        n, index = self._parse_asn1_der_integer(rsa_key_data, index)
        e, _ = self._parse_asn1_der_integer(rsa_key_data, index)

        return e, n

    def load_private_pkcs8_key(self):
        """Load an RSA private key from a PKCS#8 PEM file."""
        data, _ = self._parse_asn1_der_sequence(self.key_bytes, 0)
        index = 0

        # Parse version INTEGER (skip it)
        _, index = self._parse_asn1_der_integer(data, index)

        # Parse algorithm identifier SEQUENCE (skip it)
        _, index = self._parse_asn1_der_sequence(data, index)

        # Parse privateKey OCTET STRING
        tag, _, private_key_bytes, index = self._parse_asn1_der_element(data, index)
        if tag != 0x04:
            raise ValueError("Expected OCTET STRING")

        # Parse RSAPrivateKey SEQUENCE
        rsa_key_data, _ = self._parse_asn1_der_sequence(private_key_bytes, 0)
        index = 0

        # Parse version INTEGER (skip it)
        _, index = self._parse_asn1_der_integer(rsa_key_data, index)

        # Parse modulus (n), publicExponent (e), and privateExponent (d)
        n, index = self._parse_asn1_der_integer(rsa_key_data, index)
        e, index = self._parse_asn1_der_integer(rsa_key_data, index)
        d, _ = self._parse_asn1_der_integer(rsa_key_data, index)

        return d, n

class SimpleRSAChunkEncryptor:
    def __init__(self, public_key:tuple[int,int]=None, private_key:tuple[int,int]=None):
        self.public_key = public_key
        self.private_key = private_key
        if public_key:
            self.chunk_size = (public_key[1].bit_length() // 8)
            if self.chunk_size <= 0:
                raise ValueError("The modulus 'n' is too small. Please use a larger key size.")

    # def encrypt_chunk(self, chunk:bytes):
    #     """Encrypt a single chunk using RSA public key."""
    #     if not self.public_key: raise ValueError("Public key is required for encryption.")
    #     e, n = self.public_key
    #     chunk_int = int.from_bytes(chunk, byteorder='big')
    #     encrypted_chunk_int = pow(chunk_int, e, n)
    #     return encrypted_chunk_int.to_bytes((n.bit_length() + 7) // 8, byteorder='big')

    # def decrypt_chunk(self, encrypted_chunk:bytes):
    #     """Decrypt a single chunk using RSA private key."""
    #     if not self.private_key: raise ValueError("Private key is required for decryption.")
    #     d, n = self.private_key
    #     encrypted_chunk_int = int.from_bytes(encrypted_chunk, byteorder='big')
    #     decrypted_chunk_int:int = pow(encrypted_chunk_int, d, n)
    #     decrypted_chunk = decrypted_chunk_int.to_bytes((n.bit_length() + 7) // 8, byteorder='big')
    #     return decrypted_chunk.lstrip(b'\x00')

    # def encrypt_string(self, plaintext: str, compress: bool = False):
    #     if not self.chunk_size: raise ValueError("Public key required for encryption.")
    #     data = zlib.compress(plaintext.encode('utf-8')) if compress else plaintext.encode('utf-8')
    #     chunk_indices = range(0, len(data), self.chunk_size)
    #     chunks = [data[i:i + self.chunk_size] for i in chunk_indices]
    #     encrypted_chunks = [self.encrypt_chunk(chunk) for chunk in chunks]
    #     encoded_chunks = [base64.b64encode(chunk) for chunk in encrypted_chunks]
    #     encrypted_string = b'|'.join(encoded_chunks).decode('utf-8')
    #     return encrypted_string

    # def decrypt_string(self, encrypted_data: str):
    #     if not self.private_key: raise ValueError("Private key required for decryption.")
    #     encrypted_chunks = encrypted_data.split('|')
    #     decoded_chunks = [base64.b64decode(chunk) for chunk in encrypted_chunks]
    #     decrypted_chunks = [self.decrypt_chunk(chunk) for chunk in decoded_chunks]
    #     data = b''.join(decrypted_chunks)
    #     try:
    #         return zlib.decompress(data).decode('utf-8')
    #     except zlib.error:
    #         return data.decode('utf-8')

    
    def encrypt_string(self, plaintext: str, compress: bool = True) -> str:
        if not self.chunk_size:
            raise ValueError("Public key required for encryption.")
        
        # Step 1: Compress or encode the plaintext
        if compress:
            data = zlib.compress(plaintext.encode('utf-8'))
        else:
            data = plaintext.encode('utf-8')
        
        chunk_size = self.chunk_size - 1

        # Step 2: Split the data into chunks
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

        e, n = self.public_key


        # Step 3: Encrypt each chunk, step by step
        encrypted_chunks = []
        for chunk in chunks:
            # a. Convert chunk to hex
            chunk_hex = chunk.hex()
            # print(f"Chunk Hex: {chunk_hex}")  # Debug: Log chunk as hex

            # b. Convert hex string to BigInt, ensuring it starts without 0
            chunk_int = int('0x1' + chunk_hex, 16)
            # print(f"Chunk Int: {chunk_int}")  # Debug: Log chunk as BigInt

            # c. Encrypt the BigInt using the public key
            encrypted_int = pow(chunk_int, e, n)
            # print(f"Encrypted Int: {encrypted_int}")  # Debug: Log encrypted BigInt

            # d. Convert the encrypted BigInt to a padded hex string
            encrypted_hex = encrypted_int.to_bytes((self.chunk_size*2) *4//8, 'big').hex()
            # print(f"Encrypted Hex: {encrypted_hex}")  # Debug: Log encrypted hex

            # e. Encode the hex string to Base64
            encrypted_base64 = base64.b64encode(bytes.fromhex(encrypted_hex)).decode('utf-8')
            # print(f"Encrypted Base64: {encrypted_base64}")  # Debug: Log Base64

            # Add the final encrypted Base64 string to the list
            encrypted_chunks.append(encrypted_base64)

        # Step 4: Join encrypted chunks with a separator
        return '|'.join(encrypted_chunks)

    def decrypt_string(self, encrypted_data: str) -> str:
        if not self.private_key:
            raise ValueError("Private key required for decryption.")

        d, n = self.private_key

        # Step 1: Decode and decrypt each chunk
        decrypted_chunks = [
            bytes.fromhex(
                hex(pow(int.from_bytes(base64.b64decode(chunk), 'big'), d, n))[3:]  # Remove '0x1' prefix
            )
            for chunk in encrypted_data.split('|')
        ]

        # Step 2: Concatenate decrypted chunks
        data = b''.join(decrypted_chunks)

        # Step 3: Decode the concatenated data
        try:
            return data.decode('utf-8')  # Attempt decoding as UTF-8
        except UnicodeDecodeError:
            try:
                return zlib.decompress(data).decode('utf-8')  # Attempt decompression if decoding fails
            except Exception as e:
                raise ValueError("Failed to decode data after all attempts.") from e
    