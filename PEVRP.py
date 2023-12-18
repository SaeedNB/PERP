import copy
import datetime
import math
import pickle
import sys
from multiprocessing import Pool
import numpy as np
from search_array_jit import find_first_occurrence
from pympler import asizeof
import sys, getopt

class PEVRP:
    def __init__(self):
        with open("../../graphs/new_graph_below_175_upper_0_with_points_whole_graph.pickle", "rb") as a:
            self.graph = pickle.load(a)
        self.all_shortest_path_length = dict()

        # for i in range(800):
        #     res = nx.single_source_dijkstra_path_length(self.graph, i, weight='travel_time')
        #     self.all_shortest_path_length[i] = res

        # with open("all_shortest_path_length.pkl", "wb") as f:
        #     pickle.dump(self.all_shortest_path_length, f)

        with open("all_shortest_path_length.pkl", "rb") as f:
            self.all_shortest_path_length = pickle.load(f)
        self.powers = dict()
        for node in self.graph.nodes:
            self.powers[node] = self.graph.nodes[node].get("power", 0)
        self.station_state = np.zeros((800, 1440))
        # with open("station_status.pkl", "rb") as f:
        #     self.station_state = pickle.load(f)
        self.buckets = [50,75, 100]
        self.route_table = dict()
        self.current_bound = dict()
        self.empty_route_table = dict()
        self.empty_current_bound = dict()
        self.look_ahead_threshold = 10
        self.increase_threshold = 2
        self.outlet_count = 1

        for _id in range(800):
            for b in self.buckets:
                self.empty_route_table["{}-{}".format(_id, b)] = list()
                self.empty_current_bound["{}-{}".format(_id, b)] = -1

        self.row = {"from": None, "start_soc": None, "arrived_at": None, "leave_at": None, "arrived_soc": None,
                    "leave_soc": None, "charge_time": None, "wait_time": None}

    def run(self, requests, threads, threshold, look_ahead, outlet_count):
        self.look_ahead_threshold = look_ahead
        self.outlet_count = outlet_count
        total_travel_length = 0
        for _id, request in enumerate(requests):
            print("number of taken slots {}".format(np.count_nonzero(self.station_state != 0)))
            tot = np.count_nonzero(self.station_state == 1) + 2 * np.count_nonzero(
                self.station_state == 2) + 3 * np.count_nonzero(self.station_state == 3) + 4 * np.count_nonzero(
                self.station_state == 4)
            print("number of taken slots {}".format(tot))

            t = datetime.datetime.now()
            source = request.get("source")
            target = request.get("destination")
            soc = request.get("soc")
            start_time = request.get("start_time")
            max_receive_power = request.get("max_receive_power")
            print(datetime.datetime.now())
            found_paths = self.find_all_shortest_paths_with_current_station_status(source, target, soc, start_time,
                                                                                   max_receive_power)
            forward_look_ahead_upper_bounds, candidates = self.refine_solutions_for_selective_look_ahead(
                found_paths)
            influence_list = list()
            number_of_process = threads
            pool = Pool(processes=number_of_process)
            print("start to find influence path")
            if len(found_paths) > 1:
                thread_res_list = list()
                this_round_requests = requests[_id + 1: _id + self.look_ahead_threshold]
                divided_list_size = int((len(this_round_requests) + 1) / number_of_process)
                for n in range(1, number_of_process + 1):
                    path_list = this_round_requests[(n - 1) * divided_list_size: n * divided_list_size]

                    thread_res_list.append(pool.apply_async(self.find_all_paths_for_list_of_requests,
                                                            [path_list, forward_look_ahead_upper_bounds, candidates,None, threshold]))

                pool.close()
                pool.join()
                for thread_res in thread_res_list:
                    for item in thread_res.get():
                        influence_list.append(item)

            print("end of found influence path")
            self.calculate_influence_and_assign_best_route_v6(found_paths, influence_list, request)
            total_travel_length += int(found_paths[0]["path"][1].split("-")[1]) - request.get("start_time")
            print("route travel time : {}".format(int(found_paths[0]["path"][1].split("-")[1])))
            print("total travel length till now : {}".format(total_travel_length))
            print("request {} ended in {}".format(_id, datetime.datetime.now() - t))

    def find_all_paths_for_list_of_requests(self, requests, forward_look_ahead_upper_bounds, candidates, station_state=None, threshold = 0):
        if station_state is not None:
            self.station_state = station_state

        influence_list = list()
        for r in requests:
            source = r.get("source")
            target = r.get("destination")
            soc = r.get("soc")
            max_receive_power = r.get("max_receive_power")
            start_time = r.get("start_time")
            influence_found_paths = self.find_all_shortest_paths_with_current_station_status(source, target,
                                                                                             soc,
                                                                                             start_time,
                                                                                             max_receive_power,
                                                                                             forward_look_ahead_upper_bounds,
                                                                                             candidates,
                                                                                             threshold)
            influence_list.append(influence_found_paths)

        return influence_list

    def find_all_shortest_paths_with_current_station_status(self, source, target, soc, start_time, max_receive_power,
                                                            look_ahead_upper_bounds=None, candidates=None, threshold=0):
        self.route_table = copy.deepcopy(self.empty_route_table)
        self.current_bound = copy.deepcopy(self.empty_current_bound)
        open_nodes = list()
        possible_influence_path_counter = 0
        inf = 0
        if look_ahead_upper_bounds is not None:
            if start_time < look_ahead_upper_bounds[source]:
                possible_influence_path_counter += 1
                inf = 1

        open_nodes.append({"time": start_time, "node": "{}-{}".format(source, soc), "inf": inf, "always_one": False})
        open_nodes_set = set()
        open_nodes_set.add("{}-{}".format(source, soc))
        upper_bound = np.inf
        visited_node_list = list()
        while len(open_nodes):
            open_nodes = sorted(open_nodes, key=lambda d: d['time'])
            current_node = open_nodes.pop(0)
            open_nodes_set.remove(current_node["node"])
            if look_ahead_upper_bounds is not None:
                if possible_influence_path_counter == 0:
                    continue
                possible_influence_path_counter -= current_node["inf"]

            visited_node_list.append(current_node["node"])
            if current_node["time"] > upper_bound:
                continue

            node_name, charge_amount = current_node["node"].split("-")
            node_name = int(node_name)
            charge_amount = int(charge_amount)
            for neighbor in self.graph.neighbors(node_name):
                if neighbor < 8 and neighbor != target:
                    continue
                if neighbor == target:
                    this_round_bucket = [100]
                else:
                    this_round_bucket = self.buckets

                edge_data = self.graph.get_edge_data(node_name, neighbor)
                if edge_data.get("distance") < 50000:
                    continue
                if self.req_energy(edge_data.get("distance")) < charge_amount:
                    tt = current_node["time"] + int(edge_data.get("travel_time") / 60)

                    arrived_soc = charge_amount - self.req_energy(edge_data.get("distance"))

                    for charge_bucket in this_round_bucket:
                        if arrived_soc > charge_bucket:
                            continue
                        charge_time, wait_time = self.get_charge_at_station(neighbor, tt, charge_bucket - arrived_soc,
                                                                            max_receive_power)

                        charge_and_wait = charge_time + wait_time

                        new_node = "{}-{}".format(neighbor, charge_bucket)
                        new_inf = 0
                        always_one = current_node["always_one"]
                        if look_ahead_upper_bounds is not None:
                            if tt + charge_and_wait < look_ahead_upper_bounds[neighbor][0]:
                                new_inf = 1
                                if neighbor in candidates:
                                    always_one = True

                            if always_one:
                                new_inf = 1

                        my_row = {"from": node_name, "start_soc": charge_amount, "arrived_at": tt,
                                  "leave_at": tt + charge_and_wait,
                                  "arrived_soc": arrived_soc, "leave_soc": charge_bucket,
                                  "charge_time": charge_time, "wait_time": wait_time}
                        self.route_table["{}-{}".format(neighbor, charge_bucket)].append(my_row)
                        if neighbor != target:
                            if new_node not in open_nodes_set:
                                if new_node not in visited_node_list:
                                    open_nodes.append({"time": tt + charge_and_wait, "node": new_node, "inf": new_inf,
                                                       "always_one": always_one})
                                    possible_influence_path_counter += new_inf
                                    open_nodes_set.add(new_node)
                                    self.current_bound[new_node] = tt + charge_and_wait
                            else:
                                if self.current_bound[new_node] > tt + charge_and_wait:
                                    self.current_bound[new_node] = tt + charge_and_wait
                                    for item in open_nodes:
                                        if item["node"] == new_node:
                                            item["time"] = tt + charge_and_wait
                                            break
                        else:
                            if upper_bound > tt + charge_and_wait + threshold:
                                # print("new solution found: {}".format(datetime.datetime.now() - t))
                                upper_bound = tt + charge_and_wait+ threshold
        open_paths = list()
        complete_path_list = list()
        open_paths.append({"path": list(), "start": start_time, "end": 0})
        size = sys.getsizeof(self.route_table)
        list_sizes = [sys.getsizeof(lst) for lst in self.route_table.values()]
        total_size = asizeof.asizeof(self.route_table)
        # print("Size of dictionary:", total_size, "bytes")
        # node, arrive, leave, arrive_soc, leave_soc, charge_time, wait_time
        open_paths[0]["path"].append("{}-{}-{}-{}-{}-{}-{}".format(target, None, None, 1, 1, 0, 0))
        open_paths[0]["start"] = upper_bound
        open_paths[0]["end"] = upper_bound
        open_paths[0]["leave_soc"] = self.buckets

        while len(open_paths) > 0 :
            open_paths = sorted(open_paths, key=lambda d: d['end'], reverse=True)
            current_path = open_paths.pop(0)
            node_name = int(current_path.get("path")[-1].split("-")[0])
            all_path_to_dest = self.get_list_of_paths_by_node_and_time_range(node_name,
                                                                             current_path.get("start"),
                                                                             current_path.get("end"),
                                                                             current_path.get("leave_soc"))
            for item in all_path_to_dest:
                new_path = copy.deepcopy(current_path)
                new_node_name = "{}-{}-{}-{}-{}-{}-{}".format(item.get("from"), item.get("arrived_at"),
                                                              item.get("leave_at"), item.get("arrived_soc"),
                                                              item.get("leave_soc"), item.get("charge_time"),
                                                              item.get("wait_time"))
                new_path["path"].append(new_node_name)
                edge_data = self.graph.get_edge_data(item.get("from"), node_name)
                new_path["start"] -= (int(edge_data.get("travel_time") / 60) + item.get("charge_time"))
                new_path["end"] -= (int(edge_data.get("travel_time") / 60) + item.get("charge_time"))
                soc = self.req_energy(edge_data.get("distance")) + item["arrived_soc"]
                if soc > 75:
                    leave_soc = [100]
                elif soc > 50:
                    leave_soc = [75, 100]
                else:
                    leave_soc = [50, 75, 100]
                new_path["leave_soc"] = leave_soc
                if new_path["start"] >= start_time:
                    if item.get("from") != source:
                        open_paths.append(new_path)
                    else:
                        complete_path_list.append(new_path)
        return complete_path_list

    def get_list_of_paths_by_node_and_time_range(self, node, start, end, leave_soc_list):
        my_list = list()
        my_set = set()
        for b in self.buckets:
            node_id = "{}-{}".format(node, b)
            for item in self.route_table[node_id]:
                if item.get("leave_at") <= end and item.get("leave_soc") in leave_soc_list:
                    if str(item) not in my_set:
                        my_set.add(str(item))
                        my_list.append(item)
        return my_list

    def req_energy(self, distance):
        return int(distance / 1750)

    def calculate_influence_and_assign_best_route(self, found_path, influence_list):
        influence = np.zeros((800, 1440))
        path_counter = 0
        for paths in influence_list:
            if paths:
                path_counter += 1
            for path in paths:
                p = path.get("path")
                for _id, stop in enumerate(p[1:-1]):
                    node_id = int(stop.split("-")[0])
                    end_ = int(p[_id + 2].split("-")[2])
                    start = int(p[_id + 2].split("-")[1])
                    influence[node_id][start:end_] += 1
        print("number of influenced path: {}".format(path_counter))
        all_path_influence_count = list()
        for path in found_path:
            path_influence_for_this_path = 0
            p = path.get("path")
            for _id, stop in enumerate(p[1:-1]):
                node_id = int(stop.split("-")[0])
                end_ = int(p[_id + 2].split("-")[2])
                start = int(p[_id + 2].split("-")[1])
                path_influence_for_this_path += np.sum(influence[node_id][start:end_])
            all_path_influence_count.append(path_influence_for_this_path)

        print(all_path_influence_count)
        return all_path_influence_count

    def calculate_influence_and_assign_best_route_v6(self, found_paths, influence_list, req):
        total_path_lenghts = list()
        for vehicle_id, paths in enumerate(influence_list):
            inf_path_lengths = []
            for i_path_id, path in enumerate(paths):
                length = int(path["path"][1].split("-")[1])
                inf_path_lengths.append(length)
            total_path_lenghts.append(np.array(copy.deepcopy(inf_path_lengths)))
        number_of_destroyed_paths = list()
        inf_for_each_path = list()
        for found_path_id, found_path in enumerate(found_paths):
            total_path_lengths_for_this_round = copy.deepcopy(total_path_lenghts)
            p = found_path.get("path")
            conflict_for_this_path = set()
            for _id, stop in enumerate(p[1:-1]):
                node_id = int(stop.split("-")[0])
                end_ = int(p[_id + 2].split("-")[2])
                charge = int(p[_id + 2].split("-")[5])
                start = end_ - charge
                for vehicle_id, paths in enumerate(influence_list):
                    for i_path_id, path in enumerate(paths):
                        ip = path.get("path")
                        for i_id, stop in enumerate(ip[1:-1]):
                            i_node_id = int(stop.split("-")[0])
                            i_end_ = int(ip[i_id + 2].split("-")[2])
                            i_charge = int(ip[i_id + 2].split("-")[5])
                            i_start = end_ - i_charge
                            if node_id == i_node_id:
                                if start <= i_start <= end_ or start <= i_end_ <= end_:
                                    conflict_for_this_path.add("{}-{}".format(vehicle_id, i_path_id))
            min_before_assign = 0
            for item in total_path_lengths_for_this_round:
                if len(item):
                    min_before_assign += np.min(item)
            for conf in conflict_for_this_path:
                total_path_lengths_for_this_round[int(conf.split('-')[0])][int(conf.split('-')[1])] += threshold
            min_after_assign = 0
            for item in total_path_lengths_for_this_round:
                if len(item):
                    min_after_assign += np.min(item)

            number_of_destroyed_paths.append(len(conflict_for_this_path))
            inf_for_each_path.append(min_after_assign - min_before_assign)

        min_inf_path_id = np.argmin(np.array(inf_for_each_path))
        inf_path = inf_for_each_path[min_inf_path_id]
        locs = np.where(np.array(inf_for_each_path) <= inf_path)[0]
        charge_time_dict = dict()
        min_inf_path_id = -1
        current_charge_time_min = 99999
        for index_ in locs:
            if number_of_destroyed_paths[index_] < current_charge_time_min:
                current_charge_time_min = number_of_destroyed_paths[index_]
                min_inf_path_id = index_

        locs2 = np.where(np.array(number_of_destroyed_paths) <= current_charge_time_min)[0]
        list_of_influence = self.calculate_influence_and_assign_best_route(found_paths, influence_list)
        current_inf_min = 99999
        for index2 in locs2:
            if list_of_influence[index2] < current_inf_min:
                current_inf_min = list_of_influence[index2]
                min_inf_path_id = index2
        self.assign_route(found_paths[min_inf_path_id])

    def assign_route(self, path):
        p = path.get("path")
        for _id, stop in enumerate(p[1:-1]):
            node_id = int(stop.split("-")[0])
            end_ = int(p[_id + 2].split("-")[2])
            start = int(p[_id + 2].split("-")[1])
            charge_time = int(p[_id + 2].split("-")[5])
            wait_time = int(p[_id + 2].split("-")[6])
            print("assign route cs {}, start {}, charge {}, wait {}, end {}".format(node_id, start, charge_time, wait_time, end_))
            assert start + wait_time + charge_time == end_
            for i in range(start + wait_time, end_):
                self.station_state[int(node_id)][i] += 1

    def get_charge_at_station_old(self, node, start, amount, max_receive_power_for_this_vehicle):
        if amount <= 0:
            return 0, 0
        if node <= 7:
            return 0, 0

        kw = (amount / 100) * 50
        power = self.graph.nodes[node]["power"]
        if power > max_receive_power_for_this_vehicle:
            power = max_receive_power_for_this_vehicle

        charge_time = int((kw / power) * 60)
        charge_time = math.ceil(charge_time/5) * 5
        x = start - 1
        while True:
            x += 1
            if np.count_nonzero(self.station_state[node][x:x + charge_time] == self.outlet_count) ==0:
                break
        wait = x - start
        return charge_time, wait

    def get_charge_at_station(self, node, start, amount, max_receive_power_for_this_vehicle):
        graph_time = 0
        wait_time_ = 0
        xt = datetime.datetime.now()
        if amount <= 0:
            return 0, 0
        if node <= 7:
            return 0, 0

        kw = (amount / 100) * 50
        power = self.powers[node]
        if power > max_receive_power_for_this_vehicle:
            power = max_receive_power_for_this_vehicle

        charge_time = int((kw / power) * 60)
        charge_time = math.ceil(charge_time / 5) * 5
        charge_time +=2
        wait = find_first_occurrence(start, node, charge_time, 1440, self.station_state[node])
        return charge_time, wait

    def refine_solutions_for_selective_look_ahead(self, list_of_paths):
        forward_latest_leave_times = np.zeros((800, 1))
        for path in list_of_paths:
            p = path.get("path")
            for _id, stop in enumerate(p[1:-1]):
                node_id = int(stop.split("-")[0])
                end_ = int(p[_id + 2].split("-")[2])
                if forward_latest_leave_times[node_id][0] < end_:
                    forward_latest_leave_times[node_id][0] = end_

        open_list = list()
        candidate_list = list()
        for node_id in range(800):
            if forward_latest_leave_times[node_id][0] > 0:
                candidate_list.append(node_id)
                open_list.append({"node": node_id, "time": forward_latest_leave_times[node_id][0]})

        for item in open_list:
            for i in range(800):
                shortest_travel = int(self.all_shortest_path_length[i].get(item["node"], 0) / 60)
                if shortest_travel > 0:
                    if forward_latest_leave_times[i] < item["time"] - shortest_travel:
                        forward_latest_leave_times[i] = item["time"] - shortest_travel


            return forward_latest_leave_times, candidate_list


if __name__ == '__main__':
    threads = 1
    threshold = 10
    look_ahead = 1
    outlet_count = 1
    opts, args = getopt.getopt(sys.argv[1:], "-l:-t:-n:", ["lookahead=", "threshold=", "num="])
    for opt, arg in opts:
        if opt in ("-l", "--lookahead"):
            look_ahead = int(arg)
        elif opt in ("-t", "--threshold"):
            threshold = int(arg)
        elif opt in ("-n", "--num"):
            threads = int(arg)
    max_receive_power_list = [50, 60, 75, 80, 100]

    with open("./requests_dynamic_10_percent_with_bucket.pkl", "rb") as f:
        requests = pickle.load(f)
        requests = sorted(requests, key=lambda d: d['start_time'])
        for _id in range(len(requests)):
            requests[_id]["id"] = _id
            requests[_id]["max_receive_power"] = max_receive_power_list[_id % 5]
            requests[_id]["soc"] = 100

    t = datetime.datetime.now()
    s = PEVRP()
    s.run(requests, threads, threshold, look_ahead, outlet_count)
    print(datetime.datetime.now() - t)

