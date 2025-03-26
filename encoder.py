import time
import base64
import pickle  # <-- for serialization
from phe import paillier
from tqdm import tqdm

def generate_paillier_keypair():
    return paillier.generate_paillier_keypair()

def encode_and_encrypt_image(image_path, public_key):
    start_time = time.time()
    
    # 1. Read raw bytes from image
    t0 = time.time()
    with open(image_path, "rb") as f:
        img_data = f.read()
    t1 = time.time()
    print(f"[encoder] Reading image took {t1 - t0:.4f} seconds.")

    # 2. Convert the image bytes to a Base64-encoded string
    t0 = time.time()
    b64_string = base64.b64encode(img_data).decode('utf-8')
    t1 = time.time()
    print(f"[encoder] Base64 encoding took {t1 - t0:.4f} seconds.")
    
    # 3. Chunk the Base64 string into smaller pieces
    t0 = time.time()
    chunk_size = 256
    chunks = []
    start_time = time.time()
    print("[encoder] Starting encryption...")
    for i in range(0, len(b64_string), chunk_size):
        chunk = b64_string[i:i + chunk_size]
        chunks.append(chunk)
    t1 = time.time()
    print(f"[encoder] Splitting into {len(chunks)} chunks took {t1 - t0:.4f} seconds.")

    # 4. Encrypt each chunk
    t0 = time.time()
    encrypted_chunks = []
    print("[encoder] Starting encryption...")

    for chunk in tqdm(chunks, desc="Encrypting chunks", unit="chunk"):
        chunk_int = int.from_bytes(chunk.encode('utf-8'), byteorder='big')
        encrypted_val = public_key.encrypt(chunk_int)
        encrypted_chunks.append(encrypted_val)

    t1 = time.time()
    print(f"[encoder] Finished encryption in {t1 - t0:.4f} seconds.")

    total_time = time.time() - start_time
    print(f"[encoder] Total encoding & encryption time: {total_time:.4f} seconds.")

    return encrypted_chunks

if __name__ == "__main__":
    # Example usage
    image_path = "your_image.png"  # Update with a valid image path

    image_path = "Five_Faces\gates\gates0.jpg"
    
    # 1. Generate keys
    start_keygen = time.time()
    pub_key, pri_key = generate_paillier_keypair()
    end_keygen = time.time()
    print(f"[encoder] Key generation took {end_keygen - start_keygen:.4f} seconds.")
    
    # 2. Encrypt the image
    ciphertexts = encode_and_encrypt_image(image_path, pub_key)
    print(f"[encoder] Number of encrypted chunks: {len(ciphertexts)}")

    # 3. Save ciphertexts + keys to disk
    #    In reality, you might want to store private_key separately and more securely.
    with open("encrypted_data_.pkl", "wb") as f:
        pickle.dump({
            "public_key": pub_key,
            "private_key": pri_key,
            "ciphertexts": ciphertexts
        }, f)
    print("[encoder] Encrypted data and keys saved to 'encrypted_data.pkl'")
