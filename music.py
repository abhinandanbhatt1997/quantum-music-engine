from scipy.io import arff
import pandas as pd
from mido import Message, MidiFile, MidiTrack
from random import choices

# Load the EEG ARFF file
data, meta = arff.loadarff("/home/abhinandan/Downloads/eeg+eye+state/eeg-eye-state.arff")
df = pd.DataFrame(data)
df['eyeDetection'] = df['eyeDetection'].astype(int)

# Melody DNA map
QMDNA = {
    "melody_seeds": [
        [60, 62, 64],    # calm
        [67, 69, 71],    # energetic
        [64, 67, 72],    # focus
    ]
}

# Collapse brain state into melody
def collapse_melody(af3_value, eye_state):
    if eye_state == 1:
        weights = [0.7, 0.2, 0.1]
    elif af3_value > 500:
        weights = [0.2, 0.7, 0.1]
    else:
        weights = [0.3, 0.3, 0.4]
    return choices(QMDNA["melody_seeds"], weights=weights, k=1)[0]

# Generate MIDI
mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

for _, row in df.sample(n=100, random_state=42).iterrows():
    af3 = row['AF3']
    eye = row['eyeDetection']
    melody = collapse_melody(af3, eye)
    
    for note in melody:
        track.append(Message('note_on', note=int(note), velocity=64, time=0))
        track.append(Message('note_off', note=int(note), velocity=64, time=480))

mid.save("eeg_eyes_quantum_music.mid")
print("✅ Saved: eeg_eyes_quantum_music.mid")
