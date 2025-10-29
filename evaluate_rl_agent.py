import traci
from env.traffic_env import TrafficEnv
from agent.q_agent import QLearningAgent
import matplotlib.pyplot as plt


def evaluate_rl_agent():
    """Run the trained RL agent once and return average waiting time and queue length."""
    env = TrafficEnv("J0")
    agent = QLearningAgent(actions=env.action_space)
    agent.load_q_table("results/q_table.pkl")

    total_wait_time = 0
    total_queue = 0
    steps = 0

    # SUMO command
    sumoCmd = ["sumo", "-n", "net.net.xml", "-r", "Route.rou.xml", "--no-step-log", "true"]
    traci.start(sumoCmd)

    state = env.get_state()
    done = False
    incoming_edges = ["E1", "E2", "E3", "E4"]  # Adjust to match your network

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        state = next_state
        steps += 1
        total_wait_time += sum(traci.edge.getWaitingTime(e) for e in incoming_edges)
        total_queue += sum(traci.edge.getLastStepHaltingNumber(e) for e in incoming_edges)

    traci.close()

    # Calculate averages
    avg_wait = total_wait_time / steps if steps > 0 else 0
    avg_queue = total_queue / steps if steps > 0 else 0

    print(f"🤖 RL Agent Results → Avg Wait Time: {avg_wait:.2f}, Avg Queue: {avg_queue:.2f}")

    # Return the results so we can use them later
    return avg_wait, avg_queue


if __name__ == "__main__":
    # Run RL evaluation
    avg_wait, avg_queue = evaluate_rl_agent()

    # Baseline values from your fixed-timer evaluation
    fixed_wait = 437.37
    fixed_queue = 26.99

    # Prepare data for comparison
    methods = ['Fixed Timer', 'RL Agent']
    wait_times = [fixed_wait, avg_wait]
    queues = [fixed_queue, avg_queue]

    # Plot 1: Waiting time
    plt.figure(figsize=(6, 4))
    plt.bar(methods, wait_times, color=['gray', 'orange'])
    plt.title('Average Waiting Time Comparison')
    plt.ylabel('Seconds')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Plot 2: Queue length
    plt.figure(figsize=(6, 4))
    plt.bar(methods, queues, color=['gray', 'green'])
    plt.title('Average Queue Length Comparison')
    plt.ylabel('Vehicles')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
