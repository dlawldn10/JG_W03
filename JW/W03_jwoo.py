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
# #입력부분 수정해보기
# import sys
# sys.setrecursionlimit(10 ** 9) # 재귀 허용 깊이를 수동으로 늘려주는 코드
# # input = sys.stdin.readline
# input = sys.stdin.readlines

# graph = []

# #입력값이 없을 때까지 입력받는 방법
# #원리 = 개행문자가 입력되면 break되므로 엔터를 한번더 해줘야 입력이 들어간다.
# # while True:
# #     try:
# #         graph.append(int(input()))
# #     except:
# #         break

# for i in input:
#     graph.append(int(input()))

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


# #1707
# import sys
# input = sys.stdin.readline
# sys.setrecursionlimit(10**6)

# def dfs(v, color):
#     chk[v] = color
#     for i in coord[v]:
#         if chk[i] == 0 :
#             if not dfs(i, -color):
#                 return False
#         elif chk[i] == chk[v]:
#             return False
    
#     return True


# for _ in range(int(input())):
#     V, E = map(int, input().split())
#     coord = [[] for _ in range(V+1)]
#     chk = [0]*(V+1)
#     for _ in range(E):
#         a, b = map(int, input().split())
#         coord[a].append(b)
#         coord[b].append(a)

#     flag = True
#     for i in range(1, V+1):
#         if chk[i] == 0:
#             flag = dfs(i, 1)
#             if not flag:
#                 break

#     print('YES' if flag == True else 'NO')


# #21606
# #1:00
# #3:00
# #내 코드
# #60점
# import copy, sys
# input = sys.stdin.readline
# sys.setrecursionlimit(10**6)

# N = int(input())
# A = input().rstrip()
# coord = [[] for _ in range(N+1)]
# for _ in range(N-1):
#     a, b = map(int, input().split())
#     coord[a].append(b)
#     coord[b].append(a)

# ans = 0

# def dfs(v, status):
#     global ans
#     visited[v] = 1
#     for i in coord[v]:
#         #방문 안한
#         if visited[i] == 0:
#             #실외
#             if A[i-1] == '0':
#                 visited[i] = 1
#                 dfs(i, 0)
                    
#             #실내
#             elif A[i-1] == '1':
#                 visited[i] = 1
#                 ans += 1
                
#     #갈곳이 더이상 없는 경우
#     #이전 노드(=마지막 노드)가 실내인지 실외인지 판단
#     #실내인 경우 성공
#     if status == 1:
#         return True
#     #실외인 경우 실패
#     elif status == 0:
#         return False


# for i in range(1, N+1):
#     visited = [0 for _ in range(N+1)]
#     #방문 안한 노드이고, 실내일때 출발
#     if visited[i] == 0 and A[i-1] == '1':
#         if not dfs(i, 1):
#             continue

# print(ans)


# #100점 코드
# #실외를 하나의 컴포넌트로 생각하여, 그 주변에 인접한 실내의 개수를 dfs로 count.
# #또한, 실내 <-> 실내 길 사이에 실외가 하나도 없는 경우는 위 방법으로 count되지 않으므로, 실내(a, b)에서 출발해 실내(b, a)에서 끝나는 경우 2가지를 추가해야 한다. 
# #따라서 이때 +2를 더해준다
# import sys
# input = sys.stdin.readline
# sys.setrecursionlimit(10**6)

# N = int(input())
# A = input().rstrip()
# coord = [[] for _ in range(N+1)]

# ans = 0

# for _ in range(N-1):
#     a, b = map(int, input().split())
#     coord[a].append(b)
#     coord[b].append(a)
#     if A[a-1] == '1' and A[b-1] == '1':
#         ans += 2

# def dfs(v, cnt):
    
#     visited[v] = True
#     for i in coord[v]:
#         #방문 안한 실외
#         if A[i-1] == '0' and visited[i] == False:
#             cnt = dfs(i, cnt)
                
#         #실내
#         elif A[i-1] == '1':
#             cnt += 1
                
#     return cnt

# sum = 0
# visited = [False for _ in range(N+1)]
# for i in range(1, N+1):
    
#     #방문 안한 노드이고, 실외일때 출발
#     if visited[i] == False and A[i-1] == '0':
#         x = dfs(i, 0)
#         sum += x*(x-1)

# print(sum + ans)


# #14888
# import sys
# input = sys.stdin.readline
# sys.setrecursionlimit(10**6)

# N = int(input())
# nums = list(map(int, input().split()))
# add, sub, mul, div = map(int, input().split())

# max_value = -1e9
# min_value = 1e9

# def dfs(i, res):
#     global add, sub, mul, div, max_value, min_value
#     if i == N:
#         max_value = max(max_value, res)
#         min_value = min(min_value, res)
#         # print(min_value)

#     else:
#         if add > 0:
#             add -= 1
#             dfs(i+1, res + nums[i])
#             add += 1

#         if sub > 0:
#             sub -= 1
#             dfs(i+1, res - nums[i])
#             sub += 1

#         if mul > 0:
#             mul -= 1
#             dfs(i+1, res*nums[i])
#             mul += 1

#         if div > 0:
#             div -= 1
#             dfs(i+1, int(res/nums[i]))
#             div += 1
    
# dfs(1, nums[0])

# print(max_value)
# print(min_value)



# # 2573
# # 9:04
# # 10:30
# # pypy통과, python3 시간초과
# # 찾아보니 dfs만으로는 python에서는 시간초과가남.
# # 밑에 bfs 쓰는 코드 읽어보기.
# import sys
# input = sys.stdin.readline
# sys.setrecursionlimit(10**4)

# N, M = map(int, input().split())
# adj = []
# for i in range(N):
#     adj.append(list(map(int, input().split())))


# dy = (0, 1, 0, -1)
# dx = (-1, 0, 1, 0)


# def dfs(x, y):
#     visited[x][y] = 1
#     for k in range(4):
#         ny = y + dy[k]
#         nx = x + dx[k]

#         #방문하지 않은 노드이고 노드의 값이 0이 아닐때
#         if visited[nx][ny] == 0 and adj[nx][ny] != 0:
#             visited[nx][ny] = 1
#             dfs(nx, ny)

#         #사방을 확인했을때 0이 있으면 chk에 1더해주기
#         if adj[nx][ny] == 0:
#             chk[x][y] += 1


#     return True

# year = 0
# while True:
#     chk = [[0]*M for _ in range(N)]
#     visited = [[0]*M for _ in range(N)]
#     result = []
#     ans = 0
    
#     for i in range(N):
#         for j in range(M):
#             if visited[i][j] == 0 and adj[i][j] != 0:
#                 result.append(dfs(i, j))
    
#     #빙산 녹이기
#     for i in range(N):
#         for j in range(M):
#             if adj[i][j] != 0:
#                 adj[i][j] = 0 if adj[i][j] - chk[i][j] < 0 else adj[i][j] - chk[i][j]
#     # print(adj)
    
#     if len(result) == 0:
#         print(0)
#         break
#     if len(result) >= 2:
#         print(year)
#         break

#     year += 1



# bfs 코드
# import sys
# from collections import deque
# read = sys.stdin.readline

# def bfs():
#     q = deque()
#     q.append(artic[0])

#     visited = [[False] * M for _ in range(N)]
#     visited[artic[0][0]][artic[0][1]] = True

#     dx = [0, 0, 1, -1]
#     dy = [1, -1, 0, 0]

#     selected_iceberg = 0  # 탐색한 빙산
#     reduce = []

#     # 녹일 빙산 탐색
#     while q:
#         x, y = q.popleft()

#         selected_iceberg += 1
#         cnt = 0  # 인접한 바다 개수

#         for i in range(4):
#             nx = x + dx[i]
#             ny = y + dy[i]

#             if 0 <= nx < N and 0 <= ny < M:
#                 if arr[nx][ny] == 0:
#                     cnt += 1
#                 elif arr[nx][ny] > 0 and not visited[nx][ny]:  # 육지인 경우
#                     visited[nx][ny] = True
#                     q.append((nx, ny))

#         if cnt != 0:
#             reduce.append((x, y, cnt))

#     # 녹이기
#     for x, y, h in reduce:
#         arr[x][y] = arr[x][y] - h if arr[x][y] - h > 0 else 0
#         if arr[x][y] == 0 and (x, y) in artic:
#             artic.remove((x, y))

#     return selected_iceberg

# # 입력
# N, M = map(int, read().split())
# arr = [list(map(int, read().split())) for _ in range(N)]

# # 풀이
# answer = 0
# artic = []  # 빙산

# for x in range(1, N):
#     for y in range(1, M):
#         if arr[x][y] != 0:
#             artic.append((x, y))

# while True:
#     # 덩어리가 2개 이상인 경우
#     if len(artic) != bfs():
#         break

#     answer += 1

#     if sum(map(sum, arr[1:-1])) == 0:  # 빙하가 다 녹을때까지 덩어리가 1개
#         answer = 0
#         break

# # 출력
# print(answer)


# #2671
# #정답 코드
# import sys
# input = sys.stdin.readline
# sys.setrecursionlimit(10**4)

# N, M = map(int, input().split())

# adj = [[] for _ in range(N+1)]
# adj2 = [[] for _ in range(N+1)]

# for _ in range(M):
#     a, b = map(int, input().split())
#     adj[a].append(b)
#     adj2[b].append(a)

# # print(adj)

# mid = (N+1)/2

# def dfs(arr, v):
#     global cnt
#     for i in arr[v]:
#         if chk[i] == 0:
#             chk[i] = 1
#             cnt += 1
#             dfs(arr, i)


# ans = 0
# for i in range(1, N+1):
#     chk = [0]*(N+1)
#     cnt = 0
#     dfs(adj, i)
#     if cnt >= mid:
#         ans += 1
        
#     cnt = 0
#     dfs(adj2, i)
#     if cnt >= mid:
#         ans += 1
        
# print(ans)


# #2178
# from collections import deque
# import sys

# dy = (1, 0, -1, 0)
# dx = (0, -1, 0, 1)

# input = sys.stdin.readline
# N, M = map(int, input().split())

# graph = [input().rstrip() for _ in range(N)]

# chk = [[False]*M for _ in range(N)]
# # print(chk)
# # print(graph)

# def is_valid_coord(x, y):
#     return 0 <= y < M and 0 <= x < N

# def bfs(sx, sy, sd):
#     global ans

#     q = deque()
#     chk[sx][sy] = True
#     q.append((sx, sy, sd))

#     while len(q):
#         x, y, d = q.popleft()

#         if x == N-1 and y == M-1:
#             print(d)
#             break

#         for j in range(4):
#             nx = x + dx[j]
#             ny = y + dy[j]
#             nd = d + 1
#             # print(nx, ny)
#             if is_valid_coord(nx, ny) and not chk[nx][ny] and graph[nx][ny] == '1':
#                 chk[nx][ny] = True
#                 q.append((nx, ny, nd))

# bfs(0, 0, 1)


# #18352
# #풀이2
# #7:26
# #8:48
# from collections import deque
# import sys

# dy = (1, 0, -1, 0)
# dx = (0, -1, 0, 1)

# input = sys.stdin.readline
# N, M, K, X = map(int, input().split())

# graph = [[] for _ in range(N+1)]
# for i in range(1, 1+M):
#     a, b = map(int, input().split())
#     graph[a].append(b)

# chk = [-1]*(N+1)
# #방문 안했을 때 -1
# #방문 했을 때 0
# #다른 곳을 탐색할때 방문했던 곳이 또다른 최단거리 도시였을 수 있음.
# #-> 이미 방문한 지점 다시 방문 시 d값을 비교하여 최단거리로 갱신해줌


# q = deque()
# chk[X] = 0
# q.append((X, 1))

# ans = []
# while len(q):
#     x, d = q.popleft()

#     if d > K:
#         continue
        
#     for j in graph[x]:
#         if chk[j] == -1:
#             chk[j] = d
#             q.append((j, d+1))
#         elif chk[j] > d:
#             chk[j] = d
#             q.append((j, d+1))

# #모두 탐색했는데 도시 아무것도 없을 때
# if K not in chk:
#     print(-1)

# for i in range(1, N+1):
#     if chk[i] == K:
#         print(i)


# #풀이 1
# from collections import deque
# import sys
# input = sys.stdin.readline
# N, M, K, X = map(int, input().split())

# graph = [[] for _ in range(N+1)]
# for i in range(1, 1+M):
#     a, b = map(int, input().split())
#     graph[a].append(b)

# chk = [False]*(N+1)
# # print(graph)
# # print(chk)


# q = deque()
# chk[X] = True
# q.append((X, 0))

# ans = []

# while len(q):
#     x, d = q.popleft()

#     if d == K:
#         ans.append(x)
#         d = 1
#         continue
        
#     for j in graph[x]:
#         if not chk[j]:
#             chk[j] = True
#             q.append((j, d+1))

# #모두 탐색했는데 도시 아무것도 없을 때
# if len(ans) == 0:
#     print(-1)
# else:
#     for i in sorted(ans):
#         print(i)
# # print(chk)


#1916
# #시간초과
# from collections import deque
# import sys
# input = sys.stdin.readline
# N = int(input())
# M = int(input())

# graph = [[] for _ in range(N+1)]
# for i in range(1, 1+M):
#     a, b, c = map(int, input().split())
#     graph[a].append([b,c])

# S, E = map(int, input().split())
# chk = [False]*(N+1)

# q = deque()
# chk[S] = True
# q.append([S, 0])

# ans = []

# while len(q):
#     #현재 도시, 비용
#     x, cost = q.popleft()

#     if x == E:
#         # print(x)
#         print(cost)
#         break
        
#     tmp = 100000
#     idx = 0
#     #비용이 최소인 방향으로 이동해야함.
#     for j in graph[x]:
#         if not chk[j[0]] and j[1] < tmp:
#             tmp = j[1]
#             idx = j[0]
#     chk[idx] = True
#     q.append([idx, cost+tmp])
    

# #정답 코드
# #다익스트라 알고리즘
# import heapq
# from sys import maxsize
# import sys


# input = sys.stdin.readline

# n = int(input())
# m = int(input())

# graph = [[] for _ in range(n + 1)]
# visited = [maxsize] * (n + 1)
# for _ in range(m):
#     a, b, c = map(int, input().split())
#     graph[a].append((c, b))

# start, end = map(int, input().split())


# def dijkstra(x):
#     pq = []
#     heapq.heappush(pq, (0, x))
#     visited[x] = 0

#     while pq:
#         d, x = heapq.heappop(pq)

#         if visited[x] < d:
#             continue

#         for nw, nx in graph[x]:
#             nd = d + nw

#             if visited[nx] > nd:
#                 heapq.heappush(pq, (nd, nx))
#                 visited[nx] = nd


# dijkstra(start)

# print(visited[end])


# #2665
# import sys
# from heapq import heappush, heappop

# input = sys.stdin.readline
# n = int(input())
# room = []
# for _ in range(n):
#     room.append(list(map(int, input().rstrip())))
# visit = [[0] * n for _ in range(n)]


# def is_valid_coord(x, y):
#     return 0 <= y < n and 0 <= x < n

# def dijkstra():
#     dx = [1, -1, 0, 0]
#     dy = [0, 0, -1, 1]
#     heap = []
#     heappush(heap, [0,0,0])
#     visit[0][0] = 1

#     while len(heap):
#         a, x, y = heappop(heap)

#         if x == n-1 and y == n-1:
#             print(a)
#             return

#         for i in range(4):
#             nx = x + dx[i]
#             ny = y + dy[i]
#             if is_valid_coord(nx, ny) and visit[nx][ny] == 0:
#                 visit[nx][ny] = 1
#                 if room[nx][ny] == 0:
#                     heappush(heap, [a+1, nx, ny])
#                 elif room[nx][ny] == 1:
#                     heappush(heap, [a, nx, ny])

# dijkstra()



# #7569
# import sys
# from collections import deque

# input = sys.stdin.readline

# M, N, H = map(int, input().split())

# # box = [[list(map(int, input().split())) for _ in range(N)] for _ in range(H)]
# dq = deque()
# box = []
# for i in range(H):
#     tmp = []
#     for j in range(N):
#         tmp.append(list(map(int,input().split())))
#         for k in range(M):
#             if tmp[j][k]==1:
#                 dq.append((i, j, k))
#     box.append(tmp)

# # 앞, 오, 뒤, 왼, 위, 아래
# dx = (0, 1, 0, -1, 0, 0)
# dy = (1, 0, -1, 0, 0, 0)
# dz = (0, 0, 0, 0, 1, -1)

# def is_valid_coord(z, y, x):
#     return 0 <= x < M and 0 <= y < N and 0 <= z < H



# #익은 토마토를 검사해나가면서
# def bfs():

#     while len(dq):
#         #가로, 세로, 높이, day
#         z, y, x = dq.popleft()

#         for i in range(0, 6):
#             nx = x + dx[i]
#             ny = y + dy[i]
#             nz = z + dz[i]
#             # print(nz, ny, nx)
#             if is_valid_coord(nz, ny, nx):
#                 if box[nz][ny][nx] == 0:
#                     #안익은 곳 발견하면 익히기
#                     # chk[nz][ny][nx] = 1
#                     box[nz][ny][nx] = box[z][y][x] + 1
#                     dq.append((nz, ny, nx))


# bfs()

# day = 0
# #안익은 토마토 있는지 검사하기
# for i in range(H):
#     for j in range(N):
#         for k in range(M):
#             if box[i][j][k] == 0:
#                 print(-1)
#                 exit(0)
#         day = max(day, max(box[i][j]))

# print(day-1)



# #3055
# from collections import deque


# r, c = map(int, input().split())
# graph = [list(input()) for _ in range(r)]
# visited = [[-1] * c for _ in range(r)]
# dx, dy = (-1, 1, 0, 0), (0, 0, -1, 1)


# def bfs():
#     q = deque()

#     for i in range(r):
#         for j in range(c):
#             if graph[i][j] == "*":
#                 q.appendleft((i, j))
#                 visited[i][j] = 0
#             elif graph[i][j] == "S":
#                 q.append((i, j))
#                 visited[i][j] = 0

#     while q:
#         x, y = q.popleft()

#         for i in range(4):
#             nx, ny = x + dx[i], y + dy[i]

#             if not 0 <= nx < r or not 0 <= ny < c:
#                 continue
#             if visited[nx][ny] != -1:
#                 continue
#             if graph[nx][ny] == "*" or graph[nx][ny] == "X":
#                 continue
#             if graph[nx][ny] == "D" and graph[x][y] == "*":
#                 continue
#             if graph[nx][ny] == "D" and graph[x][y] == "S":
#                 return visited[x][y] + 1

#             q.append((nx, ny))
#             visited[nx][ny] = visited[x][y] + 1
#             graph[nx][ny] = graph[x][y]

#     return "KAKTUS"


# print(bfs())


# #2294
# import sys
# from collections import deque
# input = sys.stdin.readline

# n, k = map(int, input().split())
# chk = [0 for _ in range(k+1)]
# coins = [int(input()) for _ in range(n)]

# q = deque()

# for coin in coins:
#     if coin < k:
#         chk[coin] = 1
#         q.append([coin, 1])

# while len(q):
#     cum, cnt = q.popleft()

#     if cum == k:
#         print(cnt)
#         break

#     for coin in coins:
#         new_cum = cum + coin
#         new_cnt = cnt + 1
#         if new_cum > k:
#             continue
#         elif new_cum <= k and chk[new_cum] == 0:
#             q.append([new_cum, new_cnt])
#             chk[new_cum] = 1

# if cum != k:
#     print(-1)



# #2252
# from collections import deque
# import sys

# input = sys.stdin.readline

# #노드들의 갯수
# n = int(input())

# #간선들의 정보
# m = int(input())

# #진입차수를 기록하는 리스트
# inDegree = [0]*(n+1)

# #정렬결과를 저장하는 리스트
# result = [0]*(n+1)

# #간선들을 인접 리스트로 저장
# connect = [[]*(n+1) for _ in range(n+1)]
# for i in range(1, m+1):
#     x, y = map(int, input().split())
#     connect[x].append(y)
#     inDegree[y] += 1

# #탐색을 위한 큐 선언
# q = deque()

# #진입차수가 0인 노드를 큐에 삽입.
# for i in range(1, n+1):
#     if inDegree[i] == 0:
#         q.append(i)

# #정렬이 완전히 수행되려면 정확히 n개의 노드를 방문.
# for i in range(1, n+1):
#     # print(q)
#     x = q.popleft()
#     result[i] = x
#     for y in connect[x]:
#         #새롭게 진입차수가 0이된 정점을 큐에 삽입.
#         inDegree[y] -= 1
#         if inDegree[y] == 0:
#             q.append(y)

# #결과 출력
# for i in range(1, n+1):
#     print(result[i], end=' ')


# #2637
# from collections import deque
# import sys
# input = sys.stdin.readline

# n = int(input())
# m = int(input())
# connect = [[]*(n+1) for _ in range(n+1)]
# degree = [0]*(n+1)
# needs = [[0]*(n+1) for _ in range(n+1)]

# for i in range(m):
#     next, start, cnt = map(int, input().split())
#     connect[start].append((next, cnt))
#     degree[next] += 1

# q = deque()
# for i in range(1, n+1):
#     if degree[i] == 0:
#         q.append(i)


# while q:
    
#     #현재 부품
#     now = q.popleft()

#     for next, cnt in connect[now]:
#         #현재 부품이 기본 부품인지 아닌지 판별
#         #기본 부품일때
#         if needs[now].count(0) == n+1:
#             needs[next][now] += cnt
#         else:
#         #중간 부품일때
#             for i in range(1, n+1):
#                 needs[next][i] += needs[now][i] * cnt

#         degree[next] -= 1
#         if degree[next] == 0:
#             q.append(next)

# # print(connect)
# # print(degree)
# # print(needs)

# for x in enumerate(needs[n]):
#     if x[1] > 0:
#         print(*x)


# #1432
# #다시보기
# import heapq
# import sys
# input = sys.stdin.readline

# n = int(input())
# graph = [[] for _ in range(n+1)]
# degree = [0]*(n+1)
# result = [0]*(n+1)

# for i in range(0, n):
#     l = input().rstrip()
#     for j in range(0, n):
#         if l[j] == '1':
#             graph[j+1].append(i+1)
#             degree[i+1] += 1
# # print(graph)
# # print(degree)

# q = []
# for i in range(1, n+1):
#     if degree[i] == 0:
#         heapq.heappush(q, -i)

# N = n
# while q:
#     x = -heapq.heappop(q)
#     result[x] = N

#     for i in graph[x]:
#         degree[i] -= 1
#         if degree[i] == 0:
#             heapq.heappush(q, -i)

#     N -= 1

# if result.count(0) > 1:
#     print(-1)
# else:
#     print(*result[1:])



# #1948
# from collections import deque
# import sys
# input = sys.stdin.readline

# n = int(input())
# m = int(input())

# time = [0] * (n+1)
# degree = [0]*(n+1)
# graph = [[] for _ in range(n+1)]
# bgraph = [[] for _ in range(n+1)]
# cnt = [[] for _ in range(n+1)]

# for i in range(m):
#     a, b, t = map(int, input().split())
#     graph[a].append((t, b))
#     bgraph[b].append(a)
#     degree[b] += 1

# start, end = map(int, input().split())

# q = deque()
# q.append(start)

# while q:
#     #현재(출발) 도시
#     now = q.popleft()

#     for t, e in graph[now]:
#         degree[e] -= 1
#         if time[e] < time[now] + t:
#             time[e] = time[now] + t
#             cnt[e] = [now]
#         elif time[e] == time[now] + t:
#             cnt[e].append(now)

#         # 선행 도로를 모두 지나갔을 때
#         if degree[e] == 0:
#             q.append(e)

# q.append(end)
# route = set()
# while q:
#     now = q.popleft()
#     for x in cnt[now]:
#         if (now, x) not in route:
#             route.add((now, x))
#             q.append(x)

# print(time[end])
# print(len(route))    
