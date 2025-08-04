import whisper
import pyaudio
import numpy as np
import wave

# === CONFIG ===
MODEL_SIZE = "base"
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024

# === INIT ===
model = whisper.load_model(MODEL_SIZE)
pa = pyaudio.PyAudio()
device_index = pa.get_default_input_device_info()['index']

stream = pa.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=CHUNK_SIZE,
    input_device_index=device_index
)

print("\n✅ Recording... Speak your answers continuously.\n")
print("Press Ctrl+C to stop.\n")

# === STATE ===
all_transcript = ""
buffer = bytes()
max_record_time = 30  # seconds buffer before flush
chunk_frames = SAMPLE_RATE // CHUNK_SIZE * max_record_time

try:
    while True:
        data = stream.read(CHUNK_SIZE)
        buffer += data

        if len(buffer) > chunk_frames * CHUNK_SIZE:
            # Write buffer to WAV
            wf = wave.open("temp.wav", "wb")
            wf.setnchannels(1)
            wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(buffer)
            wf.close()

            # Transcribe
            result = model.transcribe("temp.wav")
            text = result["text"].strip()
            all_transcript += " " + text

            print("\n📄 Running Transcript:")
            print(all_transcript)

            buffer = bytes()

except KeyboardInterrupt:
    print("\n\n✅ Final Transcript Saved!\n")
    with open("final_transcript.txt", "w", encoding="utf-8") as f:
        f.write(all_transcript)
    print("➡️ Saved to: final_transcript.txt")

    stream.stop_stream()
    stream.close()
    pa.terminate()
