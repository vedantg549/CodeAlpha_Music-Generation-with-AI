pip install tensorflow tensorflow-datasets magenta

import tensorflow as tf
import tensorflow_datasets as tfds
import magenta
import numpy as np

# 1. Data Preparation
# Use pre-processed MIDI datasets from Magenta or create your own.
# Example using MAESTRO dataset:
dataset = tfds.load('maestro', split='train', as_supervised=True)

# Preprocess MIDI data into suitable numerical representations.
# This could involve converting MIDI events to sequences of notes,
# durations, velocities, or other relevant musical features.

# Example (simplified):
# Assume 'notes' represent a sequence of MIDI note numbers.
def preprocess_midi(example):
    notes = example[0] # Extract notes from example
    # ... further preprocessing (e.g., quantization, padding, feature extraction)
    return notes

dataset = dataset.map(preprocess_midi).batch(32)

# 2. Model Building (RNN Example)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=128, output_dim=128),  # Vocabulary size, embedding dim
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.Dense(128, activation='softmax') # Output probabilities for next note
])

# 3. Model Training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(dataset, epochs=10)  # Adjust epochs and other hyperparameters as needed

# 4. Music Generation

def generate_music(model, seed_sequence, length):
  generated_notes = list(seed_sequence)
  for _ in range(length):
    input_sequence = np.array([generated_notes[-1]])
    predicted_probs = model.predict(input_sequence)
    predicted_note = np.random.choice(128, p=predicted_probs[0])
    generated_notes.append(predicted_note)

  return generated_notes

# Seed music
seed_notes = [60,62,64,60]

generated_notes = generate_music(model, seed_notes, 50) # Generate 50 more notes


# 5. Post-processing and Output
# Convert numerical sequence to MIDI format using libraries like pretty_midi.

# ... (Code to convert generated_notes to a MIDI file)
# Example (Requires pretty_midi installation)
# import pretty_midi
# # Convert numerical representation back to MIDI data.

# midi_data = pretty_midi.PrettyMIDI()
# # ... create an instrument
# # ... add notes to the instrument based on generated_notes
# midi_data.write("generated_music.mid")

print("Music Generation complete")
