

from hashlib import new
from logging import root
from re import M, search
from tempfile import tempdir
from turtle import distance
import pandas as pd
import numpy as np
 
class Node(object):
    
    """A node in a Range Tree."""

    def __init__(self, value) -> None:
        self.value = value
        self.left = None
        self.right = None
        self.dim= 0
        self.assoc=None


def BuildRangeTree2d(data, enable=True):
    '''
    Construct a 2 dimensional range tree
    Arguments:
        data         : The data to be stored in range tree
        enable       : to toggle whether to build the xtree or ytree 
    Returns:
        tree
    '''

    if not data:        #No data
        return None
    if len(data) == 1:      #size of data=1--> leaf
        node = Node(data[0])   
       
    else:
        mid_val = len(data)//2 #get mid val to be root
        
        node = Node(data[mid_val])
         
        node.left = BuildRangeTree2d(data[:mid_val], enable)
        node.right = BuildRangeTree2d(data[mid_val+1:], enable)

    if enable==True:
        data1 =data.sort( key=lambda x: x[1]) # sort by y dimension
        node.assoc = BuildRangeTree2d(data1,enable)
        
    return node
    

def getValue (point, dim):
    
    '''
    Reads the desired value from node
    Returns : value of node
    '''
    if dim==0:
            value = point.value[0]
    else:
            value = point.value[1]
    
    return value


def insert(tree,point,enable=True):

    '''
    inserts an node to a tree
    tree:   existing tree
    point:   values of the node to be inserted
    enable  : to toggle whether to insert on a xtree or ytree 

    '''
    
    x = point[0]
    y = point[1]
    
    
    if tree is None:
        tree=Node(point)    #insert value
        return tree
    if enable==False:          #for dim=0
        dim=0                                      
        if x>getValue (tree,dim):        
                tree.right=insert(tree.right,point,enable)  #recursive call for right branch
                
        else: 
            tree.left=insert(tree.left,point,enable)    #recursive call for left branch
                             
    else:  
        dim=1         #for dim=1
        if y>getValue (tree,dim):        
                tree.right=insert(tree.right,point,enable)  #recursive call for right branch
                
        else:  
            tree.left=insert(tree.left,point,enable)    #recursive call for left branch

    return tree

def print_tree(root):


    '''    
    prints x dimension of tree
    useful for confirmation of insertion
    '''

    quene = []
    quene.append(root)
    while len(quene) != 0 :
        node = quene[0]
        if node.left == None:
            ll = '-'
        else:
            ll = node.left.value
        if node.right == None:
            rr = '-'
        else:
            rr = node.right.value
        print('  {n}  \n _|_ \n|   |\n{l}   {r}\n==========='.format(n = node.value, l = ll, r = rr))
        quene.pop(0)
        if node.left != None:
            quene.append(node.left)
        if node.right != None:
            quene.append(node.right)
    print('\n') 
    print('--------------------------------')


def NN(tree,query,best_node,best_distance,dim):

    '''
    Finds nearest neighbor for a specific query on a tree.
    Returns the node that is the closest to the query and the distance from it

    tree:   the existing tree we apply knn on
    query:  values of said query we are using
    best_node:  node closest to our query at the moment 
    best_distance:  distance from best_node
    dim: dimension of tree dim=0-->x,dim=1-->y

    '''

    if not tree:
        return best_node

    
    d=distance(query[dim],getValue(tree,dim))

    if d<best_distance:
        best_node=tree
        best_distance=d

        #deciding side

        if query[dim]<getValue(tree,dim):
            good_side=tree.left
            bad_side=tree.right
        else:
            good_side=tree.right
            bad_side=tree.left

        #good side
        best_node=NN(good_side,query,best_node,best_distance,dim=1)
        #checking for the bad side
        if abs(getValue(tree,dim)-query[dim])<best_distance:
                best_node=NN(bad_side,query,best_node,best_distance,dim=1)
        
    return best_node

def K_NN(tree,query,best_node,best_distance,dim,k):
    
    '''
    returns k nearest neighbors using NN function
    '''
    
    k_node_value=[]
    
    for i in range(k):
        best_node=NN(tree,query,best_node,best_distance,dim)
        k_node_value.append(best_node.value)

        if query[dim]<best_node.value[dim]:
            tree=best_node.left
        else:
            tree=best_node.right
        

    return k_node_value

import time
import matplotlib.pyplot as plt

def main():
    df=pd.read_csv(r'MulDimDataStr/out2d.csv',engine='python')
    
    df=df.drop(columns=['Unnamed: 0'])
    #print(df)
    mylist=df.values.tolist()
    x=[]
    y=[]
    # for sublist in mylist:
    #     x.append(sublist[0])
    #     y.append(sublist[1])
    # plt.scatter(x, y)
    # plt.show()
    #mylist=[[0,1],[1,2],[2,0],[3,1],[4,3],[2,1],[2,1],[0,0],[1,1]] 
    mylist.sort()    #sort by x dimension
  

    tree=BuildRangeTree2d(mylist,enable=True)
    #print_tree(tree)
    point=[1,5]
    
    point1=[9,9]
    
    newtree=insert(tree,point,True)
    #print_tree(newtree)
    point2=[2,1]
    #newtree1=insert(tree,point1,True)
    
    #print_tree(newtree)
    

    query=[0.8,0.2]
    start_time = time.time()
    nearest=K_NN(tree,query,None,np.infty,0,8)
    end_time = time.time()

    print(nearest)
    
    # for sublist in nearest:
    #     x.append(sublist[0])
    #     y.append(sublist[1])
    # x.append(query[0])
    # y.append(query[1])
    # plt.scatter(x, y)
    # plt.show()

    total=end_time-start_time

    print("time duration of k-NN on Range tree:",total)


if __name__ == "__main__":
    main()