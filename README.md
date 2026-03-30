# My Coqui API using `xtts_v2`

This is the backend for the workshop [Build Your First AI Skill](https://github.com/codebar-shanghai/workshop-agents-skills).

## How to Use

1. Put reference audio files (`*.wav`) under the folder `voices`.
2. Start the container.
3. Run `curl -X POST http://127.0.0.1:5002/tts -F "text=hello world" -F "language=en" --output en.wav`

```bash
# Use `ghcr.io/coqui-ai/tts` if you want to use CUDA
docker run --rm -it \
  -v /path/to/models:/root/.local/share/tts \
  -v /path/to/this/repo:/workspace \
  -p 5002:5002 \
  ghcr.io/coqui-ai/tts-cpu \
  python3 /workspace/app.py
```

## License

MIT
