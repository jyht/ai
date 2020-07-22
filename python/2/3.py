import networkx as nx
import matplotlib.pyplot as plt
g = nx.Graph()
g.clear() #将图上元素清空

#https://www.jianshu.com/p/e543dc63454f


a = [2,3]
g.add_nodes_from(a)
g.add_nodes_from("spam") #添加了4个节点，名为s,p,a,m
g.nodes() #可以将以上5个节点打印出来看看
H = nx.path_graph(10)
g.add_nodes_from(H) #将0~9加入了节点
#但请勿使用g.add_node(H)

g.add_edge(1,2)
e = (2,3)
g.add_edge(*e) #直接g.add_edge(e)数据类型不对，*是将元组中的元素取出
g.add_edges_from([(1,2),(1,3)])
g.add_edges_from([("a","spam") , ("a",2)])

n = 10
H = nx.path_graph(n)
g.add_edges_from(H.edges()) #添加了0~1,1~2 ... n-2~n-1这样的n-1条连续的边

g.number_of_nodes() #查看点的数量
g.number_of_edges() #查看边的数量
g.nodes() #返回所有点的信息(list)
g.edges() #返回所有边的信息(list中每个元素是一个tuple)
g.neighbors(1) #所有与1这个点相连的点的信息以列表的形式返回
g[1] #查看所有与1相连的边的属性，格式输出：{0: {}, 2: {}} 表示1和0相连的边没有设置任何属性（也就是{}没有信息），同理1和2相连的边也没有任何属性



g = nx.Graph(day="Monday") 
g.graph # {'day': 'Monday'}

MG=nx.MultiGraph()
MG.add_weighted_edges_from([(1,2,.5), (1,2,.75), (2,3,.5)])
GG=nx.Graph()

g = nx.cubical_graph()
nx.draw(g, pos=nx.spectral_layout(g), nodecolor='r', edge_color='b')
plt.show()
