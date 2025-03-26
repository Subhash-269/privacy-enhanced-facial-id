import time
import base64
import pickle

def decrypt_and_decode_image(encrypted_chunks, private_key, output_image_path):
    start_time = time.time()

    # 1. Decrypt each chunk
    t0 = time.time()
    decrypted_b64_parts = []
    
    for enc_val in encrypted_chunks:
        # Decrypt the integer
        decrypted_int = private_key.decrypt(enc_val)
        
        # Convert int back to original chunk of bytes
        byte_length = (decrypted_int.bit_length() + 7) // 8
        chunk_bytes = decrypted_int.to_bytes(byte_length, byteorder='big')
        
        # Convert bytes to string
        chunk_str = chunk_bytes.decode('utf-8')
        
        # Accumulate
        decrypted_b64_parts.append(chunk_str)
    t1 = time.time()
    print(f"[decoder] Decrypting {len(encrypted_chunks)} chunks took {t1 - t0:.4f} seconds.")

    # 2. Combine all parts into one Base64 string
    t0 = time.time()
    complete_b64_string = "".join(decrypted_b64_parts)
    t1 = time.time()
    print(f"[decoder] Concatenating Base64 string took {t1 - t0:.4f} seconds.")

    # 3. Decode from Base64 back to raw image bytes
    t0 = time.time()
    img_data = base64.b64decode(complete_b64_string)
    t1 = time.time()
    print(f"[decoder] Base64 decoding to bytes took {t1 - t0:.4f} seconds.")
    
    # 4. Write to disk as the output image
    t0 = time.time()
    with open(output_image_path, "wb") as f:
        f.write(img_data)
    t1 = time.time()
    print(f"[decoder] Writing image to disk took {t1 - t0:.4f} seconds.")

    total_time = time.time() - start_time
    print(f"[decoder] Total decryption & decode time: {total_time:.4f} seconds.")

if __name__ == "__main__":
    # Example usage: load the pickle file, decrypt, and restore the image
    pkl_file = "encrypted_data_.pkl"
    restored_image_path = "restored_image.png"
    
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    # Extract keys and ciphertext
    pub_key = data["public_key"]       # not used in decoder, but available
    pri_key = data["private_key"]      # needed for decryption
    ciphertexts = data["ciphertexts"]  # the encrypted data

    # Perform the decryption and write the resulting image
    decrypt_and_decode_image(ciphertexts, pri_key, restored_image_path)
    print(f"[decoder] Decrypted image saved as '{restored_image_path}'.")
