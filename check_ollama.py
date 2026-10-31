import urllib.request, json, subprocess, sys, os

print("="*50)
print("OLLAMA STATUS CHECK")
print("="*50)

# 1. Check if ollama binary exists anywhere
paths_to_check = [
    r"C:\Users\YASHIKA\AppData\Local\Programs\Ollama\ollama.exe",
    r"C:\Program Files\Ollama\ollama.exe",
    r"C:\Users\YASHIKA\AppData\Local\Ollama\ollama.exe",
]
found_binary = None
for p in paths_to_check:
    if os.path.exists(p):
        found_binary = p
        break

if found_binary:
    print(f"Ollama binary found: {found_binary}")
else:
    print("Ollama binary NOT found in common locations")

# 2. Check if port 11434 is open (server running)
try:
    url = "http://localhost:11434/api/tags"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=3) as response:
        data = json.loads(response.read())
        models = data.get("models", [])
        print(f"\nOllama server: RUNNING on http://localhost:11434")
        print(f"Models installed: {len(models)}")
        for m in models:
            name = m.get("name", "unknown")
            size_gb = round(m.get("size", 0) / 1e9, 1)
            print(f"  - {name}  ({size_gb} GB)")
except Exception as e:
    print(f"\nOllama server: NOT RUNNING  ({e})")
    print("To start: run 'ollama serve' in a terminal")

# 3. Try sending a test message if server is up
print("\n" + "-"*50)
print("LIVE CHAT TEST (mistral / llama3.1)")
print("-"*50)
try:
    # Try to detect which model is available
    url = "http://localhost:11434/api/tags"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=3) as response:
        data = json.loads(response.read())
        models = data.get("models", [])
        model_names = [m.get("name", "") for m in models]
        
        # Pick a model to test with
        test_model = None
        preferred = ["mistral:7b", "mistral", "llama3.1:8b", "llama3.1", "llama3:8b", "llama3"]
        for p in preferred:
            for name in model_names:
                if p in name.lower():
                    test_model = name
                    break
            if test_model:
                break
        
        if not test_model and model_names:
            test_model = model_names[0]
        
        if test_model:
            print(f"Testing with model: {test_model}")
            payload = json.dumps({
                "model": test_model,
                "messages": [{"role": "user", "content": "Say hello in one sentence."}],
                "stream": False
            }).encode("utf-8")
            
            import time
            t0 = time.perf_counter()
            req2 = urllib.request.Request(
                "http://localhost:11434/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req2, timeout=60) as r2:
                result = json.loads(r2.read())
                latency = round((time.perf_counter() - t0) * 1000, 1)
                reply = result.get("message", {}).get("content", "").strip()
                print(f"Response: {reply}")
                print(f"Latency:  {latency} ms")
                print("LIVE TEST: PASS")
        else:
            print("No models installed yet. Run: ollama pull mistral:7b")

except Exception as e:
    print(f"Live test skipped (server not running): {e}")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print("Ollama installed:", "YES" if found_binary else "NO - download from https://ollama.com")
