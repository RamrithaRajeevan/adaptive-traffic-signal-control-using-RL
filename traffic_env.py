import traci

class TrafficEnv:
    def __init__(self, junction_id="J0"):
        self.junction_id = junction_id
        self.action_space = [0, 1, 2, 3]  # four light phases
        self.prev_wait = 0

    def get_state(self):
        """Return queue lengths for each incoming edge."""
        incoming_edges = ["-E3", "-E1", "E4", "E2"]
        queue_lengths = [traci.edge.getLastStepHaltingNumber(e) for e in incoming_edges]
        return tuple(queue_lengths)

    def get_reward(self):
      incoming_edges = ["E1", "E2", "E3", "E4"]
      total_wait = sum(traci.edge.getWaitingTime(e) for e in incoming_edges)
      total_queue = sum(traci.edge.getLastStepHaltingNumber(e) for e in incoming_edges)
      reward = - (0.5 * total_wait + 2.0 * total_queue)
      print(f"Wait={total_wait:.2f}, Queue={total_queue}, Reward={reward:.2f}")
      return reward




    def step(self, action):
        """Apply selected phase and advance simulation."""
        traci.trafficlight.setPhase(self.junction_id, action)
        for _ in range(30):
            traci.simulationStep()
        next_state = self.get_state()
        reward = self.get_reward()
        done = traci.simulation.getMinExpectedNumber() <= 0
        return next_state, reward, done

    def reset(self):
        """Reload SUMO simulation for a new episode."""
        traci.load(["-n", "net.net.xml", "-r", "Route.rou.xml", "--start"])
        self.prev_wait = 0
        return self.get_state()
