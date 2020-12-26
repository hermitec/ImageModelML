from matplotlib import pyplot

f = open("graph_raw.txt","r")
x = f.readlines()
graph = []
for i in x:
    print(i)
    graph.append([float(i.split(",")[0]),float(i.split(",")[1].split(";")[0])])
pyplot.plot(graph)
pyplot.show()