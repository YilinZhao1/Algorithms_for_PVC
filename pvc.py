import matplotlib.pyplot as plt
import networkx as nx
import threading
import time
import os
import pulp
import numpy as np
import random
import itertools
import pandas as pd
import copy

class PVC_Tester:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.txt_path = os.path.join(file_path, 'result.txt')
        #self.process_txt_path = os.path.join(file_path, 'result_table.txt')
        self.stop_flag = False
        self.timer_thread = None
        with open(self.txt_path, 'w') as f:
            f.write('vertices\tk\tepsilon\ttime(s)\ttype\tmax_w\tmax_v\n')

    #visulise the input graph
    def draw_input_graph(self, graph, weight, num):
        G = nx.Graph()
        n = len(graph)
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                if graph[i][j] == 1:
                    G.add_edge(i, j, weight=weight[i][j])

        pos = nx.spring_layout(G)
        plt.figure(figsize=(20, 20))
        plt.title(f'Number of Vertices: {num}', fontsize=16, fontweight='bold')
        nx.draw(G, pos, with_labels=True, node_color='white', edge_color='grey', node_size=800, font_size=20)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
        plt.savefig(os.path.join(self.file_path, f'graphs/InputSize{num}.jpg'))
        plt.close()
        return pos
        
    #visulise the output graph
    def draw_output_graph(self, algorithm, graph, weight, num, k, max_vertices, max_weight, pos):
        G = nx.Graph()
        n = len(graph)
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                if graph[i][j] == 1:
                    G.add_edge(i, j, weight=weight[i][j])

        color_nodes = max_vertices
        color_edges = [(m, n) for m in color_nodes for n in range(len(graph)) if graph[m][n] == 1]

        plt.figure(figsize=(20, 20))
        plt.title(f'Algorithm: {type(algorithm).__name__}, Number of Vertices: {num}, k: {k}, k_vertices: {max_vertices}, weight: {max_weight}', fontsize=16, fontweight='bold')
        vertex_colors = ['lightblue' if node in color_nodes else 'white' for node in G.nodes()]
        edge_colors = ['lightgreen' if (i, j) in color_edges or (j, i) in color_edges else 'grey' for i, j in G.edges()]
        nx.draw(G, pos, with_labels=True, node_color=vertex_colors, edge_color=edge_colors, node_size=600, font_size=20)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.savefig(os.path.join(self.file_path, f'graphs/Vertex{num}-k{k}-{type(algorithm).__name__}.jpg'))
        plt.close()

    #### TEST
    def test_algorithm(self, algorithm, graph, weight, num, k, pos):
        try:
            self.stop_flag = False
            self.timer_thread = threading.Timer(180, algorithm.stop_program)

            self.timer_thread.start()
            start_time = time.time()

            if isinstance(algorithm, PVC):
                max_vertices, max_weight = algorithm.maxk_vc(graph, weight, k)
            elif isinstance(algorithm, LP_PVC):
                max_vertices, max_weight = algorithm.maxk_vc_lp(graph, weight, k)
            elif isinstance(algorithm, C_PVC):
                vn_prime = algorithm.calculate_vn_prime(graph, weight, k)
                q_lst = algorithm.calculat_Q(graph, weight, vn_prime)
                weight_prime = algorithm.create_weight_prime(graph, weight, vn_prime)
                max_vertices, max_weight = algorithm.maxk_vc_combine(graph, weight_prime, vn_prime, q_lst, k)
            elif isinstance(algorithm, G_PVC):
                max_vertices, max_weight = algorithm.maxk_vc_greedy(graph, weight, k)
            else:
                raise ValueError("Unknown algorithm.")

            end_time = time.time()
            execution_time = end_time - start_time
            execution_time = round(execution_time, 3)

            self.timer_thread.cancel()
            if algorithm.stop_flag:
                print('Algorithm STOP')
                with open(self.txt_path, 'a') as f:
                    f.write(f'{num}\t{k}\t{algorithm.epsilon}\t{180}\t{type(algorithm).__name__}\t{None}\t{None}\n')
                return

            print(f'input size: {num}  k = {k}  time: {execution_time}s')

            with open(self.txt_path, 'a') as f:
                f.write(f'{num}\t{k}\t{algorithm.epsilon}\t{execution_time}\t{type(algorithm).__name__}\t{max_weight}\t{list(max_vertices)}\n')

            #self.draw_output_graph(algorithm, graph, weight, num, k, max_vertices, max_weight, pos)

        except KeyboardInterrupt:
            print("STOP")

    # creat graphs
    def generate_graph(self, num):
        graph = [[0]*num for _ in range(num)]
        weight = [[0]*num for _ in range(num)]

        for i in range(num):
            for j in range(i + 1, num):
                #if random.randint(0, 1) == 1:
                if random.randint(0, 4) == 1:
                    #w = random.randint(1, 6)
                    graph[i][j] = graph[j][i] = 1
                    weight[i][j] = weight[j][i] = 1

        return graph, weight
    
    def generate(self, test_size):
        graphs = []
        for num in test_size:
            graph, weight = self.generate_graph(num)
            pos = self.draw_input_graph(graph, weight, num)
            graphs.append((graph, weight, pos))
        return graphs

    def test_algorithm1(self, graphs, test_k, algorithm1):
        for graph, weight, pos in graphs:
            num = len(graph)
            for k in test_k:
                print(f"Testing with size {num} and k {k} using algorithm1")
                self.test_algorithm(algorithm1, graph, weight, num, k, pos)

    def test_algorithm2(self, graphs, test_k, algorithm2):
        for graph, weight, pos in graphs:
            num = len(graph)
            for k in test_k:
                print(f"Testing with size {num} and k {k} using algorithm2")
                self.test_algorithm(algorithm2, graph, weight, num, k, pos)
    
    def test_algorithm3(self, graphs, test_k, algorithm3):
        for graph, weight, pos in graphs:
            num = len(graph)
            for k in test_k:
                print(f"Testing with size {num} and k {k} using algorithm3")
                self.test_algorithm(algorithm3, graph, weight, num, k, pos)

    def test_algorithm4(self, graphs, test_k, algorithm4):
        for graph, weight, pos in graphs:
            num = len(graph)
            for k in test_k:
                print(f"Testing with size {num} and k {k} using algorithm4")
                self.test_algorithm(algorithm4, graph, weight, num, k, pos)

    # make the structure clear
    def process_file(self):
        df = pd.read_csv(self.txt_path, sep='\t')

        #df = df.drop_duplicates(subset=['vertices', 'k', 'epsilon', 'type'])

        # sort
        df_sorted = df.sort_values(by=['vertices', 'k'])

        df_sorted.to_csv(self.txt_path, sep='\t', index=False)

    # draw size-time plots
    def k_plot_withPVC(self):
        df = pd.read_csv(self.txt_path, sep='\t')

        # generate different k values
        k_values = df['k'].unique()

        # create plots
        for k in k_values:
            df_k = df[df['k'] == k]
            plt.figure(figsize=(10, 6))

            plt.plot(df_k[df_k['type'] == 'PVC']['vertices'], df_k[df_k['type'] == 'PVC']['time(s)'], marker='o', label='PVC(t)')
            plt.plot(df_k[df_k['type'] == 'LP_PVC']['vertices'], df_k[df_k['type'] == 'LP_PVC']['time(s)'], marker='s', label='LP_PVC(t)')
            plt.plot(df_k[df_k['type'] == 'C_PVC']['vertices'], df_k[df_k['type'] == 'C_PVC']['time(s)'], marker='^', label='C_PVC(t)')
            plt.plot(df_k[df_k['type'] == 'G_PVC']['vertices'], df_k[df_k['type'] == 'G_PVC']['time(s)'], marker='x', label='G_PVC(t)')

            plt.title(f'k={k}')
            plt.xlabel('Number of Vertices')
            plt.ylabel('Time (s)')
            plt.legend()
            plt.grid(True)

            output_file_path = os.path.join(self.file_path, f'k={k}.jpg')
            plt.savefig(output_file_path)
            plt.close()

    def k_plot_withoutPVC(self):
        df = pd.read_csv(self.txt_path, sep='\t')

        k_values = df['k'].unique()

        for k in k_values:
            df_k = df[df['k'] == k]
            plt.figure(figsize=(10, 6))

            #plt.plot(df_k[df_k['type'] == 'PVC']['vertices'], df_k[df_k['type'] == 'PVC']['time(s)'], marker='o', label='PVC(t)')
            plt.plot(df_k[df_k['type'] == 'LP_PVC']['vertices'], df_k[df_k['type'] == 'LP_PVC']['time(s)'], marker='s', label='LP_PVC(t)')
            plt.plot(df_k[df_k['type'] == 'C_PVC']['vertices'], df_k[df_k['type'] == 'C_PVC']['time(s)'], marker='^', label='C_PVC(t)')
            plt.plot(df_k[df_k['type'] == 'G_PVC']['vertices'], df_k[df_k['type'] == 'G_PVC']['time(s)'], marker='x', label='G_PVC(t)')

            plt.title(f'k={k}')
            plt.xlabel('Number of Vertices')
            plt.ylabel('Time (s)')
            plt.legend()
            plt.grid(True)

            output_file_path = os.path.join(self.file_path, f'k_algo234={k}.jpg')
            plt.savefig(output_file_path)
            plt.close()

    # create tables for solutions
    def solution_csv(self):

        df = pd.read_csv(self.txt_path, delimiter='\t')
        df.drop(columns=['epsilon', 'time(s)'], inplace=True)

        df_pivot = df.pivot_table(index=['vertices', 'k'], columns='type', values=['max_w', 'max_v'], aggfunc='first').reset_index()
        df_pivot.columns = ['_'.join(col).strip() if type(col) is tuple else col for col in df_pivot.columns.values]
        df_pivot.columns = [col.replace('max_w_', 'w_').replace('max_v_', 'v_').replace('_', '_') for col in df_pivot.columns]

        output_file_path = os.path.join(self.file_path, 'solution.csv')

        df_pivot.to_csv(output_file_path, index=False)

class PVC:
    def __init__(self, epsilon=0):
        self.epsilon = epsilon
        self.stop_flag = False

    def maxk_vc(self, graph, weight, k):
        num_vertices = len(graph)
        k_weights = []
        max_weight = 0
        max_vertex = 0
        max_vertices = []

        # n' = min(k + {k / 𝜀}, n)
        n_prime = min(k + int(np.ceil(k / self.epsilon)), num_vertices)

        # sort the vertices by weights
        vertices_weight = [sum(weight[i]) for i in range(num_vertices)]
        vertices_lst = list(range(num_vertices))
        vertices_lst.sort(key=lambda x: vertices_weight[x], reverse=True)

        # select the first n' vertices to get Vn' 
        vn_prime = vertices_lst[:n_prime]
        k_vertices = list(itertools.combinations(vn_prime, k))
        for vertices in k_vertices:
            if self.stop_flag:
                return max_vertices, max_weight
            
            k_weight = 0
            len_vertices = len(vertices)
            for m in range(len_vertices - 1):
                for n in range(m + 1, len_vertices):
                    if graph[vertices[m]][vertices[n]] == 1:
                        k_weight -= weight[vertices[m]][vertices[n]]
            for vertex in vertices:
                for i in range(num_vertices):
                    if graph[vertex][i] == 1:
                        k_weight += weight[vertex][i]
            k_weights.append(k_weight)

        for index, w in enumerate(k_weights):
            if w > max_weight:
                max_weight = w
                max_vertex = index
        max_vertices.append(k_vertices[max_vertex])

        for index, w in enumerate(k_weights):
            if w == max_weight and k_vertices[index] not in max_vertices:
                max_vertices.append(k_vertices[index])
        #ramdom select one group of vertices to return
        return random.choice(max_vertices), max_weight

    def stop_program(self):
        self.stop_flag = True
        #print("Algorithm STOP")
    
class LP_PVC:
    def __init__(self, epsilon=0):
        self.epsilon = epsilon
        self.stop_flag = False

    def maxk_vc_lp(self, graph, weight, k):
        m = len(graph)
        problem = pulp.LpProblem("LP_PVC", pulp.LpMaximize)

        # Define variables
        Y = pulp.LpVariable.dicts("Y", range(m), cat='Binary')
        Z = pulp.LpVariable.dicts("Z", (range(m), range(m)), cat='Binary')

        # Objective: maximize the sum of edges
        objective = pulp.lpSum([weight[i][j] * Z[i][j] for i in range(m) for j in range(i, m) if graph[i][j] == 1])
        problem += objective

        # Constraint: number of selected vertices is less than or equal to k
        problem += pulp.lpSum([Y[i] for i in range(m)]) <= k

        # Constraint: edge coverage
        #m
        for i in range(m):
            for j in range(i + 1, m):  # Avoid duplicate constraints
                if graph[i][j] == 1:
                    problem += Z[i][j] <= Y[i] + Y[j]
                    problem += Z[i][j] <= 1

        #mute
        problem.solve(pulp.PULP_CBC_CMD(msg=False))
        
        selected_vertices = [i for i in range(m) if pulp.value(Y[i]) == 1]
        max_weight = pulp.value(problem.objective)

        print('selected_vertices: ', selected_vertices)
        print('max_weight: ', max_weight)

        return selected_vertices, max_weight

    def stop_program(self):
        self.stop_flag = True
        print("10 minutes, STOP")

class C_PVC:
    def __init__(self, epsilon = 0):
        self.epsilon = epsilon
        self.stop_flag = False

    def calculate_vn_prime(self, graph, weight, k):
        num_vertices = len(graph)
        vertices_weight = [sum(weight[i]) for i in range(num_vertices)]
        vertices_lst = [i for i in range(num_vertices)]
        vertices_lst.sort(key=lambda x: vertices_weight[x], reverse=True)

        n_prime = min(k + int(np.ceil(k / self.epsilon)), num_vertices)
        vn_prime = vertices_lst[:n_prime]
        return vn_prime

    # modify
    def calculat_Q(self, graph, weight, vn_prime):
        num_vertices = len(graph)
        Q = [0] * len(vn_prime)
        vn_prime_set = set(vn_prime)
        for i, v in enumerate(vn_prime):
            Q[i] = sum(1 for j in range(num_vertices) if graph[v][j] == 1 and j not in vn_prime_set)
        return Q
    
    # modify
    def create_weight_prime(self, graph, weight, vn_prime):
        n_prime = len(vn_prime)
        weight_prime = np.zeros((n_prime, n_prime))
        for i in range(n_prime):
            for j in range(i + 1, n_prime):
                if graph[vn_prime[i]][vn_prime[j]] == 1:
                    #weight_prime[i][j] = weight[vn_prime[i]][vn_prime[j]]
                    weight_prime[i][j] = 1
        return weight_prime
    
    def maxk_vc_combine(self, graph, weight_prime, vn_prime, Q, k):
        problem = pulp.LpProblem("COMBINE_PVC", pulp.LpMaximize)
        
        Y = pulp.LpVariable.dicts("Y", range(len(vn_prime)), cat='Binary')
        Z = pulp.LpVariable.dicts("Z", (range(len(vn_prime)), range(len(vn_prime))), cat='Binary')
        
        # Objective: sum of weights of edges in vn_prime and Q values for selected vertices
        
        objective = pulp.lpSum([weight_prime[i][j] * Z[i][j] for i in range(len(vn_prime)) 
                                    for j in range(i + 1, len(vn_prime)) if weight_prime[i][j] > 0]) + \
                    pulp.lpSum([Q[i] * Y[i] for i in range(len(vn_prime))])
        
        problem += objective

        problem += pulp.lpSum([Y[i] for i in range(len(vn_prime))]) <= k

        for i in range(len(vn_prime)):
            for j in range(i + 1, len(vn_prime)):
                if graph[vn_prime[i]][vn_prime[j]] == 1:
                    # the egde can be selected only if at least one endpoint be choosen
                    problem += Z[i][j] <= Y[i] + Y[j]
                    problem += Z[i][j] <= 1
                    #problem += Z[i][j] >= Y[i] + Y[j] - 1
        
        #mute
        problem.solve(pulp.PULP_CBC_CMD(msg=False))

        selected_vertices = [vn_prime[i] for i in range(len(vn_prime)) if pulp.value(Y[i]) == 1]
        max_weight = pulp.value(problem.objective)

        print('selected_vertices: ', selected_vertices)
        print('max_weight: ', max_weight)

        return selected_vertices, max_weight
    
    def stop_program(self):
        self.stop_flag = True
        print("10 minutes, STOP")

class G_PVC:
    def __init__(self, epsilon=0):
        self.epsilon = epsilon
        self.stop_flag = False

    def maxk_vc_greedy(self, graph, weight, k):
        num_vertices = len(graph)
        selected_vertices = []
        max_weight = 0
        graph_copy = copy.deepcopy(graph)

        for _ in range(k):
            '''
            if self.stop_flag:
                print("Algorithm stop")
                break
            '''
            max_edges = 0
            best_vertex = -1

            for vertex in range(num_vertices):
                if vertex in selected_vertices:
                    continue

                edge_count = sum(graph_copy[vertex])
                if edge_count > max_edges:
                    max_edges = edge_count
                    best_vertex = vertex

            if best_vertex == -1:
                break

            selected_vertices.append(best_vertex)
            for i in range(num_vertices):
                graph_copy[best_vertex][i] = 0
                graph_copy[i][best_vertex] = 0

            max_weight += max_edges  # 每条边权重为1，直接加边数

        print('selected_vertices: ', selected_vertices)
        print('max_weight: ', max_weight)

        return selected_vertices, max_weight

    def stop_program(self):
        self.stop_flag = True
        print("10 minutes, STOP")


test_size = [i for i in range(20, 21)]
test_k = [i for i in range(10, 11)]
test_epsilon = 0.4
algorithm1 = PVC(epsilon=test_epsilon)
algorithm2 = LP_PVC(epsilon=test_epsilon)
algorithm3 = C_PVC(epsilon=test_epsilon)
algorithm4 = G_PVC(epsilon=test_epsilon)

tester = PVC_Tester(file_path='D:/Leeds/MSc_Project/code/average2/re1')
graphs = tester.generate(test_size)


#tester.test_algorithm1(graphs, test_k, algorithm1)

tester.test_algorithm2(graphs, test_k, algorithm2)
tester.test_algorithm3(graphs, test_k, algorithm3)
tester.test_algorithm4(graphs, test_k, algorithm4)

tester.process_file()

#tester.k_plot_withPVC()

tester.k_plot_withoutPVC()
tester.solution_csv()


'''
'''


