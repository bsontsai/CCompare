import antlr4
from CParser import CParser
from CLexer import CLexer
from CListener import CListener
import graphviz
import os

import math
replace_variables = True
lambda_tree = 0.1

class Node:
    def __init__(self, node_type, value):
        self.node_type = node_type
        self.value = value
        self.children = []

def get_expression(antlr_node):
    if isinstance(antlr_node, antlr4.TerminalNode):
        return antlr_node.getSymbol().text
    str = ""
    for child in antlr_node.getChildren():
        str += get_expression(child)
    return str

def convert_antlr_tree_to_custom_tree(antlr_node):
    # Determine the node type based on the class name of antlr_node
    node_type = antlr_node.__class__.__name__
    
    # Initialize value as None by default
    value = None
    node = Node(node_type, value)

    # Handle terminal nodes by capturing specific token types
    if isinstance(antlr_node, antlr4.TerminalNode):
        token = antlr_node.getSymbol()
        token_type = token.type
        token_text = token.text

        # Customize how you capture values based on token type
        if token_type == CLexer.Identifier:
            if (replace_variables):
                value = f"Identifier: var"
            else:
                value = f"Identifier: {token_text}"
        else:
            value = f"Token: {token_text}"
    # For expression nodes, do in order traversal and make node
    elif("Expression" in node_type):
        node.node_type = "Expression"
        value = get_expression(antlr_node)
    # For non-terminal nodes, use the node type as the value
    else:
        value = node_type
        # Handle non-terminal nodes by recursively converting children
        for child in antlr_node.getChildren():
            child_node = convert_antlr_tree_to_custom_tree(child)
            node.children.append(child_node)
    
    node.value = value
    return node

def visualize_custom_tree(root):
    def add_nodes(graph, node):
        if node:
            graph.node(str(id(node)), label=str(node.value))
            for child in node.children:
                graph.edge(str(id(node)), str(id(child)))
                add_nodes(graph, child)
    
    dot = graphviz.Digraph(comment='Custom Tree Visualization')
    add_nodes(dot, root)
    dot.render('custom_tree', view=True)  # Renders and opens the tree in your default viewer

def tree_to_string(root):
    # put children in lexical order
    sorted_children = sorted(root.children, key=lambda x: x.value)
    str = ""
    for child in sorted_children:
        str += tree_to_string(child)
    str = "[" + root.value + str + "]"
    # print("root.value = " + root.value + ", str = " + str)
    return str

def build_suffix_array(s):
    suffix_array = []
    for i in range(0, len(s)):
        suffix_array.append(s[i:])
    return sorted(suffix_array)

def find_common_substrings(s1, s2):
    sa1 = build_suffix_array(s1)
    sa2 = build_suffix_array(s2)

    common_substrings = []

    i = j = 0
    while i < len(sa1) and j < len(sa2):
        suffix1 = sa1[i]
        suffix2 = sa2[j]
        lcp_length = 0

        while i < len(sa1) and j < len(sa2) and s1[suffix1 + lcp_length] == s2[suffix2 + lcp_length]:
            i += 1
            j += 1
            lcp_length += 1

        if lcp_length > 0:
            common_substrings.append(s1[suffix1:suffix1 + lcp_length])
        
        if i < len(sa1) and (j == len(sa2) or s1[suffix1:suffix1 + lcp_length] < s2[suffix2:suffix2 + lcp_length]):
            i += 1
        else:
            j += 1
    
    return common_substrings

def edit_distance(str1, str2):  
    m, n = len(str1), len(str2)  
    dp = [[0] * (n + 1) for _ in range(m + 1)]  
    for i in range(m + 1):  
        for j in range(n + 1):  
            if i == 0:  
                dp[i][j] = j  
            elif j == 0:  
                dp[i][j] = i  
            elif str1[i - 1] == str2[j - 1]:  
                dp[i][j] = dp[i - 1][j - 1]  
            else:  
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])  
  
    return dp[m][n]  

def tree_height(root):
    if not root.children:
        return 1
    else:
        return 1 + max(tree_height(x) for x in root.children)

def make_node_list(root):
    list = []
    list.append(root)
    for child in root.children:
        child_list = make_node_list(child)
        for element in child_list:
            list.append(element)
    return list

def big_c(s1, s2):
    # Case 1: s1 and s2 both leaf nodes of the "Expression" 
    if (not s1.children and not s2.children and s1.node_type == "Expression" and s2.node_type == "Expression"):
        return edit_distance(s1.value, s2.value) / max(len(s1.value), len(s2.value)) * lambda_tree
    # Case 2: root of s1 is different from root of s2
    elif (s1.node_type != s2.node_type and s1.value != s2.value):
        return 0
    # Case 3: roots fo s1 and s2 are both leaf nodes
    elif (not s1.children and not s2.children):
        # should do something but use 0 for now
        return 0
    else:
        result = 1
        s1_list = make_node_list(s1)
        s2_list = make_node_list(s2)
        s1_list.pop(0)
        s2_list.pop(0)
        for s1_node in s1_list:
            stree_max = 0
            for s2_node in s2_list:
                stree_max = max(stree_max, big_c(s1_node, s2_node))
            result *= 1 + stree_max
        return result * pow(lambda_tree, min(tree_height(s1), tree_height(s2)))

def kernel_function(t1, t2):
    t1_list = make_node_list(t1)
    t2_list = make_node_list(t2)
    kernel_score = 0
    for t1_node in t1_list:
        for t2_node in t2_list:
            kernel_score += big_c(t1_node, t2_node)
    return kernel_score


with open("CCompare/func_target.c", "r") as c_file:
    c_code = c_file.read()

# Define an input stream with your C code
input_stream = antlr4.InputStream(c_code)

# Create a lexer and token stream
lexer = CLexer(input_stream)
stream = antlr4.CommonTokenStream(lexer)

# Create a parser and parse the code
parser = CParser(stream)
tree = parser.compilationUnit()

target_root = convert_antlr_tree_to_custom_tree(tree)

# visualize_custom_tree(root)

# tree to string test
# a = Node("test", "a")
# b = Node("test", "b")
# c = Node("test", "c")
# d = Node("test", "d")
# e = Node("test", "e")
# b.children = [e, d]
# a.children = [c, b]
# test_str = tree_to_string(a)
# print(test_str)

# common substring test
# string1 = "ABABC"
# string2 = "BABC"
# common_substrings = find_common_substrings(string1, string2)
# print(common_substrings)

# edit distance test
# str1 = "kitten"  
# str2 = "sitting"  
# print(edit_distance(str1, str2)) 

# target_str = tree_to_string(root)
# print(target_str)

kt2t2 = kernel_function(target_root, target_root)
# print("kt2t2 = " + str(kt2t2))

for func_file in os.listdir("CCompare/funcs"):
    with open("CCompare/funcs/" + func_file, "r") as c_file:
        c_code = c_file.read()

    # Define an input stream with your C code
    input_stream = antlr4.InputStream(c_code)

    # Create a lexer and token stream
    lexer = CLexer(input_stream)
    stream = antlr4.CommonTokenStream(lexer)

    # Create a parser and parse the code
    parser = CParser(stream)
    tree = parser.compilationUnit()

    root = convert_antlr_tree_to_custom_tree(tree)

    # visualize_custom_tree(root)

    # func_str = tree_to_string(root)
    # print(func_str)

    # Normalize tree kernel
    kt1t1 = kernel_function(root, root)
    # print("kt1t1 = " + str(kt1t1))
    kt1t2 = kernel_function(root, target_root)
    # print("kt1t2 = " + str(kt1t2))
    kprime = kt1t2 / math.sqrt(kt1t1 * kt2t2)
    print(func_file + ": " + str(kprime))
