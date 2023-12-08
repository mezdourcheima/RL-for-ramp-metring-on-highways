# Number of vehicles to generate
num_vehicles = 100

# Start building the XML string
xml_string = '<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n'

# Definitions for Car and Bus types
xml_string += '\t<vType accel="1.0" id="Car" decel="5.0" sigma="0.0" length="2.0" maxSpeed="100.0" />\n'
xml_string += '\t<vType accel="1.0" id="Bus" decel="5.0" sigma="0.0" length="12.0" maxSpeed="1.0" />\n'

# Adding routes and initial vehicles
xml_string += "\t<!-- Route 0 -->\n"
for i in range(num_vehicles):
    xml_string += f'\t<vehicle id="veh{i}" type="Car" route="route0" depart="{i}" color="1,1,0" departLane="{i % 3}"/>\n'

# Route 1
xml_string += "\t<!-- Route 1 -->\n"
for i in range(num_vehicles, 2 * num_vehicles):
    xml_string += f'\t<vehicle id="veh{i}" type="Car" route="route1" depart="{i}" color="1,1,0" departLane="{i % 3}"/>\n'

# Route 2
xml_string += "\t<!-- Route 2 -->\n"
for i in range(2 * num_vehicles, 3 * num_vehicles):
    xml_string += f'\t<vehicle id="veh{i}" type="Car" route="route2" depart="{i}" color="1,1,0" departLane="{i % 3}"/>\n'

# Closing the XML file
xml_string += "</routes>"

# Save the XML string to a file
with open("additional_vehicles.xml", "w") as file:
    file.write(xml_string)
