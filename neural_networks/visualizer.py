import value

from graphviz import Digraph

def trace(root: value.Value) -> tuple[set, set]:
    nodes, edges = set(), set()
    def build(v: value.Value):
        if v not in nodes:
            nodes.add(v)
            for parent in v._parent:
                edges.add((parent, v))
                build(parent)
    
    build(root)
    return nodes, edges

def draw_directed(root: value.Value):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # left to right
    
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n)) 
        dot.node(name = uid, label = "{label} : data={data:.4f}, grad={grad:.4f}".format(label=n.label, data=n.data, grad=n.grad), shape="record")

        if n._op:
           dot.node(name = uid + n._op, label = n._op)
           dot.edge(uid + n._op, uid)
    
    for e in edges:
        n1, n2 = e
        uid1 = str(id(n1))
        uid2 = str(id(n2))

        dot.edge(uid1, uid2 + n2._op)
    
    dot.render()


        
        