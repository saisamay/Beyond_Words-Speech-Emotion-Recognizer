import subprocess
import os
import sys

input_file = sys.argv[1]
output_file = "converted_audio.wav"

command = [
    "ffmpeg",
    "-y",
    "-i", input_file,
    "-vn",
    "-ac", "1",
    "-ar", "16000",
    "-c:a", "pcm_s16le",
    output_file
]

subprocess.run(command, check=True)

print("Conversion successful!")
print("Output:", os.path.abspath(output_file))
print("Size:", os.path.getsize(output_file), "bytes")