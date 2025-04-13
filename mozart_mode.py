import os
import subprocess
import numpy as np
from scipy.io import arff
import pandas as pd
from mido import Message, MidiFile, MidiTrack, MetaMessage
from random import choice, randint, random, sample, uniform

# --- Configuration ---
INPUT_PATH = "/home/abhinandan/Downloads/eeg+eye+state/eeg-eye-state.arff"
OUTPUT_DIR = "/home/abhinandan/Desktop/today/music-engine/output"
os.makedirs(f"{OUTPUT_DIR}/mozart", exist_ok=True)

# --- Mozartian Music Engine ---
class MozartComposer:
    def __init__(self):
        self.scales = {
            "major": {"notes": [0, 2, 4, 5, 7, 9, 11], "chords": [[0, 4, 7], [2, 5, 9], [4, 7, 11]]},
            "minor": {"notes": [0, 2, 3, 5, 7, 8, 10], "chords": [[0, 3, 7], [2, 5, 8], [4, 7, 10]]}
        }
        self.progressions = [
            [0, 4, 5, 0],    # I-V-vi-IV (Pop)
            [0, 3, 4, 0],     # I-iv-V-I (Classical)
            [0, 5, 3, 4]      # I-vi-iv-V (50s)
        ]
        self.phrases = ["A1", "A2", "B1", "A3"]  # Mozartian phrase structure
        
    def select_scale(self, af3_value, eye_state):
        """Choose scale based on EEG data with musical intent"""
        if eye_state == 1 and af3_value > 500:
            return "minor", 60  # A minor (darker)
        else:
            return "major", 60  # C major (brighter)
            
    def generate_motif(self, scale, root_note, phrase_type):
        """Create Mozart-style motifs with proper voice leading"""
        # Core melodic shapes (Mozart lexicon)
        shapes = {
            "A1": [0, 2, 4, 2, 0],                   # Arch
            "A2": [0, 4, 7, 4, 2, 0],                # Arpeggio + resolution
            "B1": [7, 6, 5, 4, 2, 0],                # Descending line
            "A3": [0, 2, 4, 5, 4, 2, 0]              # Balanced phrase
        }
        
        # Select and transpose shape
        motif = [root_note + scale["notes"][i % len(scale["notes"])] for i in shapes[phrase_type]]
        
        # Add ornamentation (30% chance of mordent/grace note)
        if random() < 0.3:
            motif.insert(2, motif[1] + 1)  # Upper mordent
        
        return motif
        
    def harmonize(self, melody, scale, current_chord):
        """Create 3-voice harmony following classical rules"""
        # Make sure chord indices are within range of scale notes
        chord_indices = [i % len(scale["notes"]) for i in current_chord]
        chord_tones = [scale["notes"][i] + 60 for i in chord_indices]
        
        harmony = []
        for note in melody:
            # Find closest chord tone (voice leading)
            distances = [abs(note - ct) for ct in chord_tones]
            harmony.append(chord_tones[np.argmin(distances)])
            # Move chord tones smoothly
            chord_tones = [ct + (note - ct) // 2 for ct in chord_tones]
        return harmony

# --- EEG Processing ---
def load_eeg_data():
    try:
        data, meta = arff.loadarff(INPUT_PATH)
        df = pd.DataFrame(data)
        df['eyeDetection'] = df['eyeDetection'].astype(int)
        return df.sample(n=16, random_state=42)  # 16 phrases = 32 bars
    except Exception as e:
        raise Exception(f"EEG data error: {e}")

# --- MIDI Generation ---
def create_mozartian_midi(composer, df):
    mid = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    mid.tracks.append(track)
    
    # Set tempo (Allegro)
    track.append(MetaMessage('set_tempo', tempo=500000))
    
    # Add instrument (Piano)
    track.append(Message('program_change', program=0, time=0))
    
    scale_type, root = composer.select_scale(df.iloc[0]['AF3'], df.iloc[0]['eyeDetection'])
    scale = composer.scales[scale_type]
    progression = choice(composer.progressions)
    
    # Generate 32-bar structure (8 phrases)
    for i, phrase in enumerate(composer.phrases * 2):
        # Get EEG data for this phrase
        row = df.iloc[i % len(df)]
        velocity = min(max(int(row['AF3'] / 8), 110), 127)  # Fixed velocity calculation
        velocity = max(velocity, 30)  # Ensure minimum velocity
        
        # Generate melody and harmony
        melody = composer.generate_motif(scale, root, phrase)
        chord_index= progression[i % len(progression)] % len(scale["chords"])
        chord = scale["chords"][chord_index]
        harmony = composer.harmonize(melody, scale, chord)
        
        # Add expressive timing (swing eighths)
        for j, (note, chord_note) in enumerate(zip(melody, harmony)):
            duration = 240 if j % 2 == 0 else 360  # Swing rhythm
            
            # Melody with expression
            track.append(Message('note_on', note=note, velocity=velocity, time=0))
            track.append(Message('note_off', note=note, velocity=0, time=duration))
            
            # Harmony with softer touch
            track.append(Message('note_on', note=chord_note, velocity=max(velocity-20, 10), time=0))
            track.append(Message('note_off', note=chord_note, velocity=0, time=duration))
        
        # Phrase ending (cadence)
        if i % 4 == 3:
            track.append(Message('note_on', note=root, velocity=velocity, time=120))
            track.append(Message('note_off', note=root, velocity=0, time=480))
    
    return mid

# --- Main ---
if __name__ == "__main__":
    print("🎼 Composing Mozartian EEG Music...")
    
    # Initialize
    mozart = MozartComposer()
    try:
        eeg_data = load_eeg_data()
    except Exception as e:
        print(f"❌ Error loading EEG data: {e}")
        exit(1)
    
    # Create MIDI
    midi = create_mozartian_midi(mozart, eeg_data)
    midi_path = f"{OUTPUT_DIR}/mozart/eeg_mozart.mid"
    midi.save(midi_path)
    print(f"✅ Saved Mozartian MIDI to {midi_path}")
    
    # Convert to audio
    try:
        wav_path = f"{OUTPUT_DIR}/mozart/eeg_mozart.wav"
        subprocess.run([
            "fluidsynth", "-ni", 
            "/usr/share/sounds/sf2/FluidR3_GM.sf2", 
            midi_path, 
            "-F", wav_path,
            "-g", "0.8"
        ], check=True)
        print(f"🎧 Audio rendered to {wav_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Audio rendering failed. Install fluidsynth: sudo apt-get install fluidsynth fluid-soundfont-gm")
    except FileNotFoundError:
        print("❌ fluidsynth not found. Please install it first.")
    
    print("✨ Composition complete! Listen to your brain's Mozart!")
