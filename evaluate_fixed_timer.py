import traci
import sumolib

def evaluate_fixed_timer():
    sumoCmd = ["sumo", "-n", "net.net.xml", "-r", "Route.rou.xml", "--no-step-log", "true"]
    traci.start(sumoCmd)
    
    step = 0
    total_wait_time = 0
    total_queue = 0

    incoming_edges = ["E1", "E2", "E3", "E4"]  # adjust to your actual edges

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
        total_wait_time += sum(traci.edge.getWaitingTime(e) for e in incoming_edges)
        total_queue += sum(traci.edge.getLastStepHaltingNumber(e) for e in incoming_edges)

    traci.close()
    avg_wait = total_wait_time / step
    avg_queue = total_queue / step
    print(f"📊 Fixed-Timer Results → Avg Wait Time: {avg_wait:.2f}, Avg Queue: {avg_queue:.2f}")

if __name__ == "__main__":
    evaluate_fixed_timer()
