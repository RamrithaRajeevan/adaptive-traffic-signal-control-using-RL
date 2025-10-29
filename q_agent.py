import random
import pickle

class QLearningAgent:
    def __init__(self, actions):
        self.actions = actions
        self.alpha = 0.1      # Learning rate
        self.gamma = 0.9      # Discount factor
        self.epsilon = 0.1    # Exploration rate
        self.q_table = {}     # Q-table stored as a dictionary

    def get_q_value(self, state, action):
        """Return Q-value for a given state-action pair."""
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        """Choose an action using epsilon-greedy strategy."""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Explore
        q_values = [self.get_q_value(state, a) for a in self.actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
        return random.choice(best_actions)  # Exploit

    def learn(self, state, action, reward, next_state):
        """Update Q-table using the Q-learning formula."""
        old_q = self.get_q_value(state, action)
        next_q = max([self.get_q_value(next_state, a) for a in self.actions])
        new_q = old_q + self.alpha * (reward + self.gamma * next_q - old_q)
        self.q_table[(state, action)] = new_q

    def save_q_table(self, filename="results/q_table.pkl"):
        """Save the learned Q-table."""
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"💾 Q-table saved to {filename}")

    def load_q_table(self, filename="results/q_table.pkl"):
        """Load an existing Q-table if it exists."""
        try:
            with open(filename, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"📂 Loaded Q-table from {filename}")
        except FileNotFoundError:
            print("⚠️ No saved Q-table found — starting fresh.")

