import numpy as np
import math
import os

import xml.etree.ElementTree as ET
from xml.dom import minidom

# 00000000000000000000000000000000000000000000000
#               Inhance XML
# 00000000000000000000000000000000000000000000000
def prettify(elem):
    """
    Make xml file to look prity.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


# 11111111111111111111111111111111111111111111111
#           Generating Trip File
# 11111111111111111111111111111111111111111111111
def gen_trip_file(pos, SUMO_FILES_PATH):
    """
    make all posible trips.
    """
    TRIP_FILE_PATH = os.path.join(SUMO_FILES_PATH, 'trips.xml')
    trips = ET.Element('trips')
    for p in pos:
        for i in range(len(pos)):
            if p != pos[i]:
                data = {'id':'{s}_{e}'.format(s=p, e=pos[i]), 'depart':'0', 'from':'{}2TL'.format(p), 'to':'TL2{}'.format(pos[i])}
                ET.SubElement(trips, 'trip', data)
    with open(TRIP_FILE_PATH, 'w') as f:
        print(prettify(trips), file=f)


# 22222222222222222222222222222222222222222222222
#            Valid Paths
# 22222222222222222222222222222222222222222222222
def gen_valid_paths(path=None):
    """
    generate and read result.rou.xml file.
    """
    TRIP_FILE_PATH = os.path.join(path, 'trips.xml')
    # GENERATING result.rou.xml
    NET_FILE_PATH = os.path.join(path, 'tlc.net.xml')
    RESULT_FILE_PATH = os.path.join(path, 'result.rou.xml')
    os.system('duarouter --route-files {} --net-file {} --output-file {}'.format(TRIP_FILE_PATH, NET_FILE_PATH, RESULT_FILE_PATH))
    
    # READING id AND edge
    tree = ET.parse(RESULT_FILE_PATH)
    root = tree.getroot()
    name = []
    route = []
    for child in root:
        attrb = child.attrib['id']
        if attrb == "E0_W0":
            attrb = "E_W"
        elif attrb == "E0_N0":
            attrb = "E_N"
        elif attrb == "E0_S0":
            attrb = "E_S"

        elif attrb == "W0_E0":
            attrb = "W_E"
        elif attrb == "W0_N0":
            attrb = "W_N"
        elif attrb == "W0_S0":
            attrb = "W_S"

        elif attrb == "N0_E0":
            attrb = "N_E"
        elif attrb == "N0_W0":
            attrb = "N_W"
        elif attrb == "N0_S0":
            attrb = "N_S"

        elif attrb == "S0_E0":
            attrb = "S_E"
        elif attrb == "S0_W0":
            attrb = "S_W"
        elif attrb == "S0_N0":
            attrb = "S_N"

       # print("child.attrib['id']: ", child.attrib['id'])
        name.append(attrb)
        for child2 in child:
#            rslt = child2.attrib['edges']
            route.append(child2.attrib['edges'])
   # print({'id':name, 'edges':route})
    return {'id':name, 'edges':route}


# 33333333333333333333333333333333333333333333333
#              Car Generation steps
# 33333333333333333333333333333333333333333333333
def car_gen_step(seed, car_gen, time):
    """
    return a steps on which car is generated.
    """
    np.random.seed(seed)
    min_new = 0
    max_new = time
 
    timings = np.random.weibull(2, car_gen)
    timings = np.sort(timings)
    car_dep_time = []
    min_old = math.floor(timings[1])
    max_old = math.ceil(timings[-1])
    for value in timings: #dep time gen logic?
        car_dep_time = np.append(car_dep_time, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)
        # weibul distribution implementation
    car_dep_time = np.rint(car_dep_time)
    return car_dep_time


# 44444444444444444444444444444444444444444444444
#           Route File Generator
# 44444444444444444444444444444444444444444444444
def gen_rout_file(seed, car_gen, sim_time, ACC_CAR_GEN_PROB, STER_DIR_PROB, DEP_SPEED):
    """
    generate tlc.rou.xml file.
    """
    # INITALIZE
    SUMO_FILES_PATH = 'includes/sumo'
    POSITIONS = ['E', 'W', 'N', 'S']
    
    # GENERATING trips.xml
    gen_trip_file(POSITIONS, SUMO_FILES_PATH)

    # GETING VALID PATHS
    routes = gen_valid_paths(SUMO_FILES_PATH)

    # GETTING DEPARTURE TIME
    car_dep_time = car_gen_step(seed, car_gen, sim_time)

    # WRITING tlc.rou.xml FILE
    root = ET.Element('routes')
    ET.SubElement(root, 'vType', {"accel":"1.0", "decel":"4.5", "id":"standard_car", "length":"5.0", "minGap":"2.5", "maxSpeed":"25", "sigma":"0.5"})
    for i in range(len(routes['id'])):
#        ET.SubElement(root, 'route', {"id":routes['id'][i], "edges":routes['edges'][i]})
        ET.SubElement(root, 'route', {"id":routes['id'][i], "edges":routes['edges'][i]})

    # FINDING SOURCE AND DESTINATION
    np.random.seed(seed)
    for c in range(len(car_dep_time)):
    # SELECTING CAR DEPLOY POSITION
        gen_prob = np.random.uniform()
        for i in range(len(POSITIONS)):
            lower_lim = ACC_CAR_GEN_PROB[i]
            upper_lim = ACC_CAR_GEN_PROB[i+1]
            if (lower_lim<=gen_prob) and (gen_prob<upper_lim):
                sour = POSITIONS[i]
        
        # SELECTING STREIGHT AND TURN
        dir_prob = np.random.uniform()
        if dir_prob < STER_DIR_PROB:
            if 'E' in sour:
                dest = sour.replace('E', 'W')
            elif 'W' in sour:
                dest = sour.replace('W', 'E')
            elif 'N' in sour:
                dest = sour.replace('N', 'S')
            elif 'S' in sour:
                dest = sour.replace('S', 'N')
        else:
            dest = sour
            while dest == sour:
                dest = np.random.choice(POSITIONS)

        ET.SubElement(root, 'vehicle', {"id":"{}_{}_{}".format(sour,dest, c), "type":"standard_car", "route":"{}_{}".format(sour,dest), "depart":str(car_dep_time[c]), "departLane":"random", "departSpeed":str(DEP_SPEED)})
    ROUTE_FILE_PATH = os.path.join(SUMO_FILES_PATH, 'tlc.rou.xml')
    with open(ROUTE_FILE_PATH, 'w') as f:
        print(prettify(root), file=f) 


# 55555555555555555555555555555555555555555555555
#           All Direction Traffic Flow
# 55555555555555555555555555555555555555555555555

class GenerateTraffic:
    def __init__(self, sim_time):
        self.sim_time = sim_time
        print("Traffic reset")

    def set_traffic_flow(self, seed, turn_count):
        #used in agent training, 
        # for generating better training samples
        """
        change traffic flow from low, high, ew_high and ns_high.
        """

        # INITALIZE
        STER_DIR_PROB = 0.6
        DEP_SPEED = 10 

        if turn_count == 0: # LOW TRAFFIC
            car_gen = 600
            ACC_CAR_GEN_PROB = [0, 0.25, 0.50, 0.75, 1.00]  # [0.25, 0.25, 0.25, 0.25]

        elif turn_count == 1: # HIGH TRAFFIC
            car_gen = 3000
            ACC_CAR_GEN_PROB = [0, 0.25, 0.50, 0.75, 1.00]  # [0.25, 0.25, 0.25, 0.25]

        elif turn_count == 2: # EW HIGH TRAFFIC
            car_gen = 1500
            ACC_CAR_GEN_PROB = [0, 0.40, 0.80, 0.90, 1.00]  # [0.40, 0.40, 0.10, 0.10]

        elif turn_count == 3: # NS HIGH TRAFFIC
            car_gen = 1500
            ACC_CAR_GEN_PROB = [0, 0.10, 0.20, 0.60, 1.00]  # [0.10, 0.10, 0.40, 0.40]
        
        gen_rout_file(seed, car_gen, self.sim_time, ACC_CAR_GEN_PROB, STER_DIR_PROB, DEP_SPEED)

if __name__ == '__main__':
    traffic = GenerateTraffic(6400)
    traffic.set_traffic_flow(5, 3)
