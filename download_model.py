"""
Download the large safetensors shard using HTTP Range requests to resume after connection resets.
"""
import os
import time
import urllib.request
import ssl

SNAPSHOT = r"C:\Users\Lenovo\.cache\huggingface\hub\models--Qwen--Qwen2-VL-2B-Instruct\snapshots\895c3a49bc3fa70a340399125c650a463535e71c"
BASE_URL = "https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/resolve/895c3a49bc3fa70a340399125c650a463535e71c"
FILENAME = "model-00001-of-00002.safetensors"

DEST = os.path.join(SNAPSHOT, FILENAME)
URL = f"{BASE_URL}/{FILENAME}"
CHUNK_SIZE = 512 * 1024  # 512 KB read chunks
MAX_RETRIES = 50  # many retries since we resume

def download_with_resume():
    ctx = ssl.create_default_context()
    
    # Delete the corrupt partial file
    if os.path.exists(DEST) and os.path.getsize(DEST) < 1_000_000_000:
        print(f"Removing corrupt partial file ({os.path.getsize(DEST)} bytes)")
        os.remove(DEST)
    
    # Use a .part file for in-progress download
    part_file = DEST + ".part"
    
    for attempt in range(1, MAX_RETRIES + 1):
        # Check how much we already have
        existing_size = 0
        if os.path.exists(part_file):
            existing_size = os.path.getsize(part_file)
        
        print(f"\nAttempt {attempt}/{MAX_RETRIES} — resuming from {existing_size / 1e6:.1f} MB")
        
        try:
            headers = {"User-Agent": "SATQuery/1.0"}
            if existing_size > 0:
                headers["Range"] = f"bytes={existing_size}-"
            
            req = urllib.request.Request(URL, headers=headers)
            resp = urllib.request.urlopen(req, timeout=120, context=ctx)
            
            # Check if server supports range
            status = resp.status
            content_length = resp.headers.get("Content-Length")
            content_range = resp.headers.get("Content-Range")
            
            if existing_size > 0 and status == 200:
                # Server ignored Range header, starting from scratch
                print("  Server does not support Range. Starting from 0.")
                existing_size = 0
                mode = "wb"
            elif status == 206:
                print(f"  Resuming. Content-Range: {content_range}")
                mode = "ab"
            else:
                mode = "wb"
                existing_size = 0
            
            total = existing_size
            with open(part_file, mode) as f:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
                    if total % (10 * 1024 * 1024) < CHUNK_SIZE:
                        print(f"  ... {total / 1e6:.1f} MB")
            
            # Check if download seems complete
            final_size = os.path.getsize(part_file)
            print(f"  Downloaded to {final_size / 1e6:.1f} MB")
            
            if final_size > 3_000_000_000:  # shard 1 should be ~4GB
                os.replace(part_file, DEST)
                print(f"\nSUCCESS: {FILENAME} ({final_size / 1e6:.1f} MB)")
                return True
            else:
                print(f"  File seems incomplete ({final_size / 1e6:.1f} MB), will retry...")
                
        except Exception as e:
            current_size = os.path.getsize(part_file) if os.path.exists(part_file) else 0
            print(f"  FAIL at {current_size / 1e6:.1f} MB: {e}")
            wait = min(30, 3 * attempt)
            print(f"  Waiting {wait}s before retry...")
            time.sleep(wait)
    
    print(f"\nFAILED after {MAX_RETRIES} attempts")
    return False


if __name__ == "__main__":
    download_with_resume()
