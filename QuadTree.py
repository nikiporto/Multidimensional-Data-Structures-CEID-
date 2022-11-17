import pandas as pd

class Point:
    
    def __init__(self, x, y):
        self.x=x 
        self.y=y

    def __str__(self):
        return '('+str(self.x)+','+str(self.y)+ ')'

    def __repr__(self):
        return '('+str(self.x)+','+str(self.y)+ ')'


class Node:

    def __init__(self, kx, ky, w, h):
        self.kx = kx 
        self.ky = ky
        self.w = w
        self.h = h
        self.w_edge = kx - w/2
        self.e_edge = kx + w/2
        self.n_edge = ky - h/2
        self.s_edge = ky + h/2

    def contains(self, point):

        point_x = point.x
        point_y = point.y

        return (point_x >= self.w_edge and point_x <  self.e_edge and point_y >= self.n_edge and point_y < self.s_edge)

    def intersects(self, other):
        
        return not (other.w_edge > self.e_edge or other.e_edge < self.w_edge or other.n_edge > self.s_edge or other.s_edge < self.n_edge)


class QuadTree:

    def __init__(self, bounds, capacity, depth=0):

        self.bounds = bounds
        self.capacity = capacity
        self.points = []
        self.depth = depth
        self.div = False

    def divide(self):

        kx = self.bounds.kx
        ky = self.bounds.ky
        w = self.bounds.w / 2
        h = self.bounds.h / 2

        self.nw = QuadTree(Node(kx - w/2, ky - h/2, w, h), self.capacity, self.depth + 1)
        self.ne = QuadTree(Node(kx + w/2, ky - h/2, w, h), self.capacity, self.depth + 1)
        self.se = QuadTree(Node(kx + w/2, ky + h/2, w, h), self.capacity, self.depth + 1)
        self.sw = QuadTree(Node(kx - w/2, ky + h/2, w, h), self.capacity, self.depth + 1)

        self.div = True

    def insert(self, point):

        if not self.bounds.contains(point):

            return False

        if len(self.points) < self.capacity:

            self.points.append(point)
            return True

        if not self.div:
            self.divide()

        return (self.ne.insert(point) or self.nw.insert(point) or self.se.insert(point) or self.sw.insert(point))

    def search(self, bounds, searchresult):

        if not self.bounds.intersects(bounds):

            return False

        for point in self.points:
            if bounds.contains(point):
                searchresult.append(point)

        if self.div:
            self.nw.search(bounds, searchresult)
            self.ne.search(bounds, searchresult)
            self.se.search(bounds, searchresult)
            self.sw.search(bounds, searchresult)
        return searchresult


width, height = 1.01, 1.01

df=pd.read_csv(r'c:MulDimDataStr/out2d.csv',engine='python')
    
df=df.drop(columns=['Unnamed: 0'])

mylist=df.values.tolist()

df_min_max_scaled = df.copy()

for column in df_min_max_scaled.columns:
    df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min()) 

normlist=df_min_max_scaled.values.tolist()

points = [Point(list[0], list[1]) for list in normlist]

domain = Node(width/2, height/2, width, height)
qtree = QuadTree(domain, 1)
for point in points:
    print(point)
    qtree.insert(point)

import time


searchresult=[]
searchspace=Node(0.5,0.5,0.1,0.1)
start_time = time.time()
q=qtree.search(searchspace, searchresult)
end_time = time.time()

total=end_time-start_time
print(q)

print("time duration of range search on Quad tree:",total)


