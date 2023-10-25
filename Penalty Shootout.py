import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding

class Goalkeeper:
    def __init__(self):
        # Define side names and create a mapping from sides to indices
        self.side_names = ["top left", "bottom left", "center", "top right", "bottom right"]
        self.side_to_index = {side: i for i, side in enumerate(self.side_names)}
        
        # Define the RNN model
        self.model = Sequential([
            Embedding(input_dim=len(self.side_names), output_dim=10, input_length=3),
            SimpleRNN(32, activation='relu'),
            Dense(len(self.side_names), activation='softmax')
        ])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Initialize the shot history
        self.shot_history = []

        # Initialize the pattern-based decisions
        self.patterns = {}

        # Initialize penalty scores
        self.penalties_scored = 0
        self.penalties_saved = 0

    def predict_next_shot(self):
        if len(self.shot_history) >= 3:
            # Check if there's a known pattern in the shot history
            for pattern, predicted_shot in self.patterns.items():
                if self.shot_history[-3:] == list(pattern):
                    return predicted_shot

            # If no pattern, use the RNN to predict the next shot
            input_sequence = [self.side_to_index[side] for side in self.shot_history[-3:]]
            input_sequence = np.array(input_sequence).reshape(1, -1)
            predicted_index = np.argmax(self.model.predict(input_sequence))
            return self.side_names[predicted_index]
        
        # If not enough history, choose a random side
        return random.choice(self.side_names)

    def save_shot(self, side):
        # Update the shot history
        self.shot_history.append(side)

    def update_patterns(self):
        if len(self.shot_history) >= 4:
            pattern = tuple(self.shot_history[-4:-1])
            next_shot = self.shot_history[-1]
            self.patterns[pattern] = next_shot

    def train_model(self, num_training_rounds=1000):
        for _ in range(num_training_rounds):
            # Pick a random side
            user_shot = random.choice(self.side_names)

            # Save the shot
            self.save_shot(user_shot)

            # Update patterns based on user's shots
            self.update_patterns()

            # If there are enough shots, train the RNN
            if len(self.shot_history) >= 4:
                input_sequence = [self.side_to_index[side] for side in self.shot_history[-4:-1]]
                output_index = self.side_to_index[self.shot_history[-1]]
                input_sequence = np.array(input_sequence).reshape(1, -1)
                output = np.zeros((1, len(self.side_names)))
                output[0, output_index] = 1
                self.model.fit(input_sequence, output, epochs=1, verbose=0)

    def play(self):
        while True:
            # Get the user's shot
            user_shot = input("Enter a side (top left, bottom left, center, top right, bottom right), 'stop or end' to close the program: ")

            # Check if the user wants to stop
            if user_shot.lower() == 'stop' or user_shot.lower() == 'end':
                break

            # Validate the user's shot
            if user_shot not in self.side_names:
                print("Invalid input. Choose from the provided sides.")
                continue

            # Predict the user's next shot
            predicted_shot = self.predict_next_shot()

            # Save the shot
            self.save_shot(user_shot)

            # Update patterns based on user's shots
            self.update_patterns()

            # Print the result and update penalty scores
            if predicted_shot == user_shot:
                print("Goal Saved! You shot {} and the keeper saved it.".format(user_shot))
                self.penalties_saved += 1
            elif predicted_shot == 'center' and predicted_shot != user_shot:
                print("Goal Scored! You shot {} and the keeper stood still.".format(user_shot))
                self.penalties_scored += 1
            else:
                print("Goal Scored! You shot {} and the keeper dived {}.".format(user_shot, predicted_shot))
                self.penalties_scored += 1
            print("Penalties Scored: {}, Penalties Saved: {}\n".format(self.penalties_scored, self.penalties_saved))

if __name__ == "__main__":
    goalkeeper = Goalkeeper()

    # Train the model for a set number of rounds
    goalkeeper.train_model(num_training_rounds=1000)

    # Play the game
    goalkeeper.play()
