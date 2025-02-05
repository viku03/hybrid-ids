import hashlib
import secrets

class RollingKeyEncryptor:
    def __init__(self):
        # Generate initial key
        self.current_key = secrets.randbits(256)
    
    def generate_key_hash(self):
        """
        Generate SHA-256 hash of current key
        
        :return: SHA-256 hash of current key
        """
        return hashlib.sha256(
            str(self.current_key).encode()
        ).hexdigest()
    
    def derive_next_key(self):
        """
        Derive next key by using last 5 bits of current key
        
        :return: Decimal representation of last 5 bits
        """
        # Get last 5 bits
        last_5_bits = self.current_key & 0b11111
        
        # Generate new key using hash and last 5 bits
        new_key_hasher = hashlib.sha256()
        new_key_hasher.update(str(self.current_key).encode())
        new_key_hasher.update(str(last_5_bits).encode())
        
        # Update and return new key
        self.current_key = int(new_key_hasher.hexdigest(), 16)
        return last_5_bits
    
    def encrypt(self, plaintext):
        """
        Encrypt plaintext using current key
        
        :param plaintext: Text to encrypt
        :return: Encrypted text and key hash for first transaction
        """
        # XOR encryption with current key
        encrypted = ''.join([
            chr(ord(char) ^ (self.current_key & 0xFF)) 
            for char in plaintext
        ])
        
        # For first transaction, return key hash
        key_hash = self.generate_key_hash()
        
        return encrypted, key_hash
    
    def decrypt(self, ciphertext):
        """
        Decrypt ciphertext
        
        :param ciphertext: Text to decrypt
        :return: Decrypted text
        """
        # XOR decryption with current key
        decrypted = ''.join([
            chr(ord(char) ^ (self.current_key & 0xFF)) 
            for char in ciphertext
        ])
        
        return decrypted

def simulate_communication():
    """
    Simulate communication between two devices
    """
    # Sender and receiver with same initial setup
    sender = RollingKeyEncryptor()
    receiver = RollingKeyEncryptor()
    
    # First transaction
    messages = [
        "Hello, secure communication!",
        "Next message with rolling key",
        "Demonstrating key derivation"
    ]
    
    for message in messages:
        # Sender encrypts
        encrypted, key_hash = sender.encrypt(message)
        print(f"\nSent: {message}")
        print(f"Encrypted: {encrypted}")
        print(f"Key Hash: {key_hash}")
        
        # Receiver verifies and decrypts
        print("Receiver Side:")
        
        # Verify key hash (simulated)
        receiver_hash = receiver.generate_key_hash()
        print(f"Receiver Key Hash: {receiver_hash}")
        
        # Derive next key (last 5 bits)
        sender_next_key_bits = sender.derive_next_key()
        receiver_next_key_bits = receiver.derive_next_key()
        
        print(f"Sender Next Key Bits: {sender_next_key_bits}")
        print(f"Receiver Next Key Bits: {receiver_next_key_bits}")
        
        # Decrypt
        decrypted = receiver.decrypt(encrypted)
        print(f"Decrypted: {decrypted}")
        
        # Verify keys are synchronized
        assert sender_next_key_bits == receiver_next_key_bits, "Keys out of sync!"
        print("Keys synchronized successfully!")

if __name__ == "__main__":
    simulate_communication()