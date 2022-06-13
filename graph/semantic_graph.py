import os.path
import copy 
from collections import defaultdict
import math
import numpy as np
import pickle
from bidict import bidict
import networkit as nk
from graph.SemanticOperation import SemanticOperation,config_permutations, config_to_unique_str
import random

class SemanticGraph:

    ORACLE_PATH = 'data/'
    ORACLE_NAME = 'oracle_perm_block'

    def __init__(self,configs : bidict,graph :nk.graph,nb_blocks,GANGSTR=True,edges_infos=None,args=None):
        self.configs = configs
        if edges_infos == None:
            self.edges_infos = defaultdict(dict)
        else : 
            self.edges_infos = edges_infos
        self.nk_graph = graph
        self.nb_blocks = nb_blocks
        self.GANGSTR = GANGSTR
        self.args = args
        self.semantic_operation = SemanticOperation(nb_blocks,True)

        self.frontier = set(self.get_frontier_nodes())
        self.graph_transpose = None

##########################################
    # I/O operations : 
##########################################

    def save(self,path,name):
        writer = nk.Format.NetworkitBinary
        graph_filename = f"{path}graph_{name}.nk"
        if os.path.isfile(graph_filename):
            os.remove(graph_filename)
        nk.writeGraph(self.nk_graph,graph_filename, writer)
        with open(f'{path}semantic_network_{name}.pk', 'wb') as f:
            pickle.dump(self,f)

    def load(path:str,name:str):
        reader = nk.Format.NetworkitBinary
        nk_graph = nk.readGraph(f"{path}graph_{name}.nk", reader)
        with open(f'{path}semantic_network_{name}.pk', 'rb') as f:
            semantic_graph = pickle.load(f)
        semantic_graph.nk_graph = nk_graph
        if semantic_graph.nk_graph.isDirected():
            semantic_graph.graph_transpose = nk.graphtools.transpose(semantic_graph.nk_graph)
        return semantic_graph

    def __getstate__(self):
        return {k:v for (k, v) in self.__dict__.items() if not isinstance(v,nk.graph.Graph)}

    def load_oracle(nb_blocks:int):
        return SemanticGraph.load(SemanticGraph.ORACLE_PATH,
                                f'{SemanticGraph.ORACLE_NAME}{nb_blocks}')

##########################################
    # Shortest path operations   : 
##########################################

    def sample_shortest_path(self,source,target,algorithm='dijkstra'):
        ''' sample path among all optimal shortest-path 
            returns path,path_sr,path_length'''
        if algorithm == 'dijkstra':
            algo_class = nk.distance.Dijkstra
        elif algorithm == 'bfs' : 
            algo_class = nk.distance.BFS
        else : 
            raise Exception("unknown shortest path algorithm : ",algorithm)
        source,target = tuple(source),tuple(target)
        try :
            sssp = algo_class(self.nk_graph, self.configs[source], True, False, self.configs[target])
        except KeyError:
            return None,0,float('inf')
        sssp.run()
        return self.sample_shortest_path_with_sssp(source,target,sssp,return_configs=True) 
    
    def sample_shortest_path_with_sssp(self,source,target,sssp,return_configs=False,reversed=False):
        ''' sample path among all optimal shortest-path 
            returns path,path_sr,path_length'''
        source,target = tuple(source),tuple(target)
        try :
            source_node = self.configs[source]
            target_node = self.configs[target]
            config_path,sr,distance = self.sample_shortest_path_with_sssp_from_nodes(source_node,target_node,sssp,return_configs=return_configs,reversed=reversed)
        except KeyError:
            config_path = None
        return config_path,sr,distance

    def sample_shortest_path_with_sssp_from_nodes(self,source_node,target_node,sssp,return_configs=False,reversed=False):
        ''' sample path among all optimal shortest-path with sssp object 
            returns path,path_sr,path_length'''
        if source_node == target_node : 
            path= []
        else : 
            paths = sssp.getPaths(source_node if reversed else target_node)
            if paths : 
                path = random.choice(paths)
                if reversed and path: 
                    path = path[::-1]
            else : 
                path = None
        sr,distance = self.get_score_from_path_node(path)
        if path and return_configs:
            path =  [self.configs.inverse[node] for node in  path]
        return path,sr,distance

    def get_sssp_to_goal(self,goal,use_weight=True):
        '''
        Return a sssp (Single Source Shortest Path) object of shortest path from goal to all other nodes on the tranposed graph.
        '''
        if goal in self.configs:
            if self.graph_transpose == None : 
                self.graph_transpose = nk.graphtools.transpose(self.nk_graph)
            if not self.nk_graph.hasNode(self.configs[goal]):
                raise Exception('missing node on graph')
            if not self.graph_transpose.hasNode(self.configs[goal]):
                raise Exception('missing node on tranpose graph')
            if use_weight :
                sssp_from_goal = nk.distance.Dijkstra(self.graph_transpose,self.configs[goal], True, False)
            else : 
                sssp_from_goal = nk.distance.BFS(self.graph_transpose,self.configs[goal], True, False)
            sssp_from_goal.run()
            return sssp_from_goal
        else : 
            return None
            
    def get_path_from_coplanar(self,goal):
        return self.sample_shortest_path(self.semantic_operation.empty(),goal)
        
##########################################
    # SR path estimation  : 
##########################################


    def get_neighbors_to_goal_sr(self,source,neighbors,goal,reversed_sssp):
        if isinstance(reversed_sssp,nk.distance.BFS):
            source_to_neighbors_sr = np.ones(len(neighbors))
        else:
            source_to_neighbors_sr = np.exp(-np.array([self.getWeight(source,neighbour)
                                            for neighbour in neighbors]))
        _,neighbors_to_goal_sr,neighbors_to_goal_dist = zip(*[self.sample_shortest_path_with_sssp(neighbour,goal,reversed_sssp,reversed=True)
                                        for neighbour in neighbors])

        neighbors_to_goal_sr = np.array(neighbors_to_goal_sr)
        return source_to_neighbors_sr,neighbors_to_goal_sr,neighbors_to_goal_dist

    def get_score_from_path_node(self,path):
        '''
            return a tuple (SR,path_length)
            empty path means unreachable
        '''
        if path == []:
            return (1,0)
        elif path == None : 
            return (0,float('inf'))
        else:
            dist = np.sum([self.getWeight_withNode(n1,n2)
                                for (n1,n2) in zip(path[:-1],path[1:]) 
                            ])
            return (np.exp(-dist),len(path))

    def get_isolated_nodes(self):
        isolated = []
        for c in self.nk_graph.iterNodes():
            if self.nk_graph.isIsolated(c):
                isolated.append(c)
        return isolated
    
    def get_reachables_node_ids(self,source):
        reachables = []
        if source in self.configs:
            source_id = self.configs[source]
            bfs = nk.distance.BFS(self.nk_graph, source_id, True, True)
            bfs.run()
            reachables = bfs.getNodesSortedByDistance()
        return reachables

    def get_frontier_nodes(self):
        if self.empty() not in self.configs:
            return []
        
        dijkstra_from_coplanar = nk.distance.Dijkstra(self.nk_graph,self.configs[self.empty()], True, False)
        dijkstra_from_coplanar.run()
        intermediate_nodes = set()
        for node in self.nk_graph.iterNodes():
            predecessors = dijkstra_from_coplanar.getPredecessors(node)
            if predecessors : 
                intermediate_nodes.update(predecessors)
        isolated = []
        for node in self.nk_graph.iterNodes():
            if node not in intermediate_nodes:
                isolated.append(node)
        return isolated

##########################################
    # Graph construction  : 
#########################################

    def create_node(self,config):
        if config not in self.configs:
            self.configs[config] = self.nk_graph.addNode()

    def edge_config_to_edge_id(self,edge_config):
        c1,c2 = edge_config
        return (self.configs[c1],self.configs[c2])

    def create_edge_stats(self,edge,start_sr):

        # if self.args.one_object_edge and not self.semantic_operation.one_object_edge(edge):
        #     return

        n1,n2 = self.edge_config_to_edge_id(edge)
        if not self.nk_graph.hasEdge(n1,n2):
            self.nk_graph.addEdge(n1,n2)
            self.edges_infos[(n1,n2)] = {'SR':start_sr,'Count':1}
            clamped_sr = max(np.finfo(float).eps, min(start_sr, 1-np.finfo(float).eps))
            self.nk_graph.setWeight(n1,n2,-math.log(clamped_sr))
        else : 
            raise Exception(f'Already existing edge {n1}->{n2}')

    def update_edge_stats(self,edge_configs,success):
        
        # if self.args.one_object_edge and not self.semantic_operation.one_object_edge(edge_configs):
        #     return

        edge_id = self.edge_config_to_edge_id(edge_configs)
        success = int(success)
        
        if not self.edges_infos[edge_id]:
            raise Exception(f"unknown edge {edge_id[0]}->{edge_id[1]}")
        else:
            # update SR  :
            self.edges_infos[edge_id]['Count']+=1
            count = self.edges_infos[edge_id]['Count']
            last_mean_sr = self.edges_infos[edge_id]['SR']
            if self.args.edge_sr == 'moving_average':
                new_mean_sr = last_mean_sr + (1/count)*(success-last_mean_sr)
            elif self.args.edge_sr == 'exp_moving_average':
                new_mean_sr = last_mean_sr + self.args.edge_lr* (success-last_mean_sr)
            else : 
                raise Exception(f"Unknown self.args.edge_sr value : {self.args.edge_sr}")
            self.edges_infos[edge_id]['SR'] = new_mean_sr

    def update_graph_edge_weight(self,edge):
        n1,n2 = edge
        new_mean_sr = self.edges_infos[(n1,n2)]['SR']
        clamped_sr = max(np.finfo(float).eps, min(new_mean_sr, 1-np.finfo(float).eps))
        self.nk_graph.setWeight(n1,n2,-math.log(clamped_sr))

    def update_edge(self,edge,success):
        self.update_edge_stats(edge,success)
        self.update_graph_edge_weight(edge)

    def update(self):
        ''' Synchronize edges stats and edge weigth in nk_graph '''
        for edge in self.edges_infos:
            self.update_graph_edge_weight(edge)
        self.frontier = set(self.get_frontier_nodes())
        self.graph_transpose = nk.graphtools.transpose(self.nk_graph)
    
    def hasNode(self,config):
        if config in self.configs:
            return self.nk_graph.hasNode(self.configs[config])
        return False

    def hasEdge(self,config_start,config_end):
        if config_start in self.configs and config_end in self.configs:
            return self.nk_graph.hasEdge(self.configs[config_start],
                                         self.configs[config_end])
        return False

    def iterNeighbors(self,config):
        '''iter over neighbors of a node, take in a semantic config
            return a generator over semantic configs.'''
        if config in self.configs:
            if not self.nk_graph.hasNode(self.configs[config]):
                raise Exception('Missing node on graph')
            return (self.configs.inverse[node_id] for node_id in self.nk_graph.iterNeighbors(self.configs[config]))
        else : 
            return []

    def getNodeId(self,config):
        return self.configs.get(config,None)

    def getConfig(self,nodeId):
        return self.configs.inverse[nodeId]

    def getWeight(self,c1,c2):
        n1,n2 = self.getNodeId(c1), self.getNodeId(c2)
        if n1 == None or n2 == None:
            raise Exception("Unknown config")
        return self.getWeight_withNode(n1,n2)

    def getWeight_withNode(self,n1,n2):
        if not self.nk_graph.hasNode(n1) or not  self.nk_graph.hasNode(n2):
            raise Exception('Unknown node')
        if not self.nk_graph.hasEdge(n1,n2):
            raise Exception('Unknown edge')
        return self.nk_graph.weight(n1,n2)
    
    def empty(self):
        return self.semantic_operation.empty()
    
    def log(self,logger):
        logger.record_tabular('agent_nodes',self.nk_graph.numberOfNodes())
        logger.record_tabular('agent_edges',self.nk_graph.numberOfEdges())
        
def augment_with_all_permutation(nk_graph,configs,nb_blocks,GANGSTR=True):
    '''
    Takes a nk_graph as entry, configs wich translate configuration into node id. 
    configs is supposed to only contains unique ordered configuration
    Return a new nk_graph and a new config dict with all ordered configurations.
    '''
    new_configs = copy.deepcopy(configs)
    new_nk_graph = copy.deepcopy(nk_graph)
    config_to_perms = dict()
    semantic_operator = SemanticOperation(nb_blocks,GANGSTR)

    # creates new nodes
    for config in configs:
        config_to_perms[config] = config_permutations(config,semantic_operator)
        for config_perm in config_to_perms[config]:
            if config_perm not in new_configs:
                new_config_perm_id = new_nk_graph.addNode()
                new_configs[config_perm] = new_config_perm_id
                
    # creates new edges
    for config,config_perms in config_to_perms.items():
        for perm_id,config_perm in enumerate(config_perms):
            new_perm_id = new_configs[config_perm]
            for neighbour_id in nk_graph.iterNeighbors(configs[config]):
                perm_corresponding_neighbour = config_to_perms[new_configs.inverse[neighbour_id]][perm_id]
                perm_corresponding_id = new_configs[perm_corresponding_neighbour]
                if not new_nk_graph.hasEdge(new_perm_id,perm_corresponding_id):
                    new_nk_graph.addEdge(new_perm_id,perm_corresponding_id)

    return new_nk_graph,