import os
import sys
import csv
import matplotlib.pyplot as plt
import traci

from env.traffic_env import TrafficEnv
from agent.q_agent import QLearningAgent

# SUMO setup
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui')
else:
    sys.exit(" Please declare environment variable SUMO_HOME.")

sumoCmd = [
    sumoBinary,
    "-n", "net.net.xml",
    "-r", "Route.rou.xml",
    "--start",
    "--quit-on-end",
    "--step-length", "1.0"
]

# Training 
def train():
    episodes = 100
    env = TrafficEnv("J0")
    agent = QLearningAgent(actions=env.action_space)

    agent.load_q_table()

    rewards = []
    os.makedirs("results/plots", exist_ok=True)
    print(" Starting Q-Learning traffic signal training...")

    for episode in range(episodes):
        print(f"\nEpisode {episode+1}/{episodes} started...")
        traci.start(sumoCmd)
        state = env.get_state()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        traci.close()
        print(f"Episode {episode+1}/{episodes} finished | Total Reward: {total_reward:.2f}")

    
    agent.save_q_table()

   #save reward
    csv_path = "results/rewards.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward"])
        for i, r in enumerate(rewards):
            writer.writerow([i+1, r])
    print(f" Rewards saved to: {csv_path}")

    # Plot 
    plt.figure(figsize=(7,4))
    plt.plot(range(1, episodes+1), rewards, marker="o", color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve: Q-Learning Traffic Agent")
    plt.grid(True)
    plot_path = "results/plots/learning_curve.png"
    plt.savefig(plot_path)
    plt.show()
    print(f" Learning curve saved to: {plot_path}")

    print("\n Training complete — rewards and plot generated successfully!")


if __name__ == "__main__":
    train()
