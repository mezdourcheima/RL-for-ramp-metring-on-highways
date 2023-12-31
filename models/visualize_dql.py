import traci
import pickle
import time
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("trained_traffic_light_model_dql.h5")

# Load the test data
with open("/Users/cheimamezdour/Projects/RLOC-SUMO/dql_test_data.pkl", "rb") as f:
    test_data = pickle.load(f)

# Connect to SUMO with GUI
try:
    traci.start(["sumo-gui", "-c", "/Users/cheimamezdour/Projects/RLOC-SUMO/mynet.sumocfg"])

    # Loop through the test data and apply actions
    for entry in test_data:
        state = entry["state"]

        # Use the trained model to choose an action
        q_values = model.predict(np.array(state).reshape(1, -1))[0]
        action = np.argmax(q_values)

        # Apply the chosen action to the traffic light in SUMO
        traci.trafficlight.setPhase("n2", action)

        # Step the simulation
        traci.simulationStep()

        # Optional: Add a delay to slow down the visualization
        time.sleep(0.1)  # Adjust the delay as needed

except traci.exceptions.FatalTraCIError as e:
    print(f"Error: {e}")

finally:
    # Close connection to SUMO in the finally block
    try:
        traci.close()
    except traci.exceptions.FatalTraCIError:
        pass  # Ignore if there is no active connection
