# #1991
# #1:14
# #1:33

# N = int(input())
# l = dict()
# for _ in range(N):
#     node, left, right = map(str, input().split())
#     l[node] = (left, right)
    
# # print(l)

# def preorder(node):

#     if node == '.':
#         return

#     print(node, end='')
#     preorder(l[node][0])
#     preorder(l[node][1])

# def inorder(node):

#     if node == '.':
#         return

#     inorder(l[node][0])
#     print(node, end='')
#     inorder(l[node][1])


# def postorder(node):

#     if node == '.':
#         return

#     postorder(l[node][0])
#     postorder(l[node][1])
#     print(node, end='')


# preorder('A')
# print()
# inorder('A')
# print()
# postorder('A')


# #5639
# import sys
# sys.setrecursionlimit(10 ** 9) # 재귀 허용 깊이를 수동으로 늘려주는 코드
# input = sys.stdin.readline

# graph = []

# #입력값이 없을 때까지 입력받는 방법
# #원리 = 개행문자가 입력되면 break되므로 엔터를 한번더 해줘야 입력이 들어간다.
# while True:
#     try:
#         graph.append(int(input()))
#     except:
#         break

# def dfs(start, end):

#     if start > end:
#         return

#     mid = end + 1

#     #서브트리 찾기
#     for i in range(start+1, end+1):
#         if graph[start] < graph[i]:
#             mid = i
#             break

#     dfs(start+1, mid-1)
#     dfs(mid, end)
#     print(graph[start])

# dfs(0, len(graph)-1)


# #1197
# import sys
# input = sys.stdin.readline
 
# V, E = map(int, input().split())
# Vroot = [i for i in range(V+1)]
# Elist = []
# for _ in range(E):
#     Elist.append(list(map(int, input().split())))
 
# Elist.sort(key=lambda x: x[2])
 
 
# def find(x):
#     if x != Vroot[x]:
#         Vroot[x] = find(Vroot[x])
        
#     return Vroot[x]
 
 
# answer = 0
# for s, e, w in Elist:
#     sRoot = find(s)
#     eRoot = find(e)
#     if sRoot != eRoot:
#         if sRoot > eRoot:
#             Vroot[sRoot] = eRoot
#         else:
#             Vroot[eRoot] = sRoot
#         answer += w
 
# print(answer)


# #1260
# from collections import deque


# N, M, V = map(int, input().split())

# adj = [[0]*(N+1) for _ in range(N+1)]
# for _ in range(1, M+1):
#     a, b = map(int, input().split())
#     adj[a][b] = 1
#     adj[b][a] = 1



# visited = [0] * (N+1)

# def dfs(i):
    
#     visited[i] = 1
#     print(i, end=' ')

#     for j in range(1, N+1):
#         if not visited[j] and adj[i][j] == 1:
#             visited[j] = 1
#             dfs(j)



# def bfs(i):
#     visited2 = [0] * (N+1)
#     dq = deque()
#     dq.append(V)
#     visited2[V] = 1

#     while len(dq)>0:

#         v = dq.popleft()
#         print(v, end=' ')
#         for k in range(N+1):
#             if not visited2[k] and adj[v][k] == 1:
#                 visited2[k] = 1
#                 dq.append(k)         

# dfs(V)
# print()
# bfs(V)


# #11724
# #8:26
# #8:51
# import sys
# #리밋 수정 필수
# sys.setrecursionlimit(10 ** 9)
# input = sys.stdin.readline
# N, M = map(int, input().split())
# adj = [[0]*(N+1) for _ in range(N+1)]
# chk = [0]*(N+1)

# for _ in range(M):
#     u, v = map(int, input().split())
#     adj[u][v] = 1
#     adj[v][u] = 1


# def dfs(i):
#     for j in range(1, N+1):
#         if not chk[j] and adj[i][j] == 1:
            
#             chk[j] = 1
#             dfs(j)
#     return 

# ans = 0
# for i in range(1, N+1):
#     if not chk[i]:
#         ans += 1
#         chk[i] = 1
#         dfs(i)    

# print(ans)


# #2606
# #8:52
# #8:57
# import sys
# sys.setrecursionlimit(10 ** 9)
# input = sys.stdin.readline
# N = int(input())
# M = int(input())
# adj = [[0]*(N+1) for _ in range(N+1)]
# chk = [0]*(N+1)

# for _ in range(M):
#     u, v = map(int, input().split())
#     adj[u][v] = 1
#     adj[v][u] = 1


# def dfs(i):
#     global ans
#     for j in range(1, N+1):
#         if not chk[j] and adj[i][j] == 1:
#             ans += 1
#             chk[j] = 1
#             dfs(j)
#     return 

# ans = 0
# chk[1] = 1
# dfs(1)

# print(ans)


# #11725
# import sys
# sys.setrecursionlimit(10**6)
# input = sys.stdin.readline

# N = int(input())
# l = [[] for _ in range(N+1)]

# for _ in range(N-1):
#     a, b = map(int, input().split())
#     l[a].append(b)
#     l[b].append(a)
# # print(l)

# visited = [0] * (N+1)

# def dfs(s):
#     for i in l[s]:
#         if visited[i] == 0:
#             visited[i] = s
#             dfs(i)

# dfs(1)

# for i in range(2, N+1):
#     print(visited[i])


#1707