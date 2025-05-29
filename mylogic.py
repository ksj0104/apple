import sys

rows, cols = 11, 20
v = [[0 for _ in range(cols + 1)] for _ in range(rows + 1)]
seg = [[0] * 1000 for _ in range(rows)]

def mk(row, idx, s, e):
    if s == e:
        seg[row][idx] = v[row][s]
        return
    mid = (s + e) // 2
    lc = idx + idx + 1
    rc = lc + 1
    mk(row, lc, s, mid)
    mk(row, rc, mid + 1, e)
    seg[row][idx] = seg[row][lc] + seg[row][rc]

def query(row, left, right, idx, s, e):
    if left == s and right == e:
        return seg[row][idx]
    lc = idx + idx + 1
    rc = lc + 1
    mid = (s + e) // 2
    if right <= mid:
        return query(row, left, right, lc, s, mid)
    elif left > mid:
        return query(row, left, right, rc, mid + 1, e)
    else:
        return query(row, left, mid, lc, s, mid) + query(row, mid + 1, right, rc, mid + 1, e)

def update(row, col, val, idx, s, e):
    if s == e:
        seg[row][idx] = val
        return
    lc = idx + idx + 1
    rc = lc + 1
    mid = (s + e) // 2
    if col <= mid:
        update(row, col, val, lc, s, mid)
    else:
        update(row, col, val, rc, mid + 1, e)
    seg[row][idx] = seg[row][lc] + seg[row][rc]

def get_sum(x1, y1, x2, y2):
    ret = 0
    for i in range(x1, x2 + 1):
        ret += query(i, y1, y2, 0, 0, cols - 1)
    return ret

# The rest of the code translation will continue below due to size.
def find_rect():
    ret = []
    for i in range(rows):
        for j in range(cols):
            if v[i][j] == 0:
                continue
            for i2 in range(i, rows):
                for j2 in range(j, -1, -1):
                    res = get_sum(i, j2, i2, j)
                    if res == 10:
                        if v[i2][j2] == 0:
                            continue
                        ret.append(((i, j2), (i2, j)))
                    elif res > 10:
                        break
                for j2 in range(j, cols):
                    res = get_sum(i, j, i2, j2)
                    if res == 10:
                        if v[i2][j2] == 0:
                            continue
                        ret.append(((i, j), (i2, j2)))
                    elif res > 10:
                        break
    return ret

def find_only():
    ret = []
    rev = []
    cnt = [[0] * 30 for _ in range(20)]

    for i in range(rows):
        for j in range(cols):
            for i2 in range(i, rows):
                for j2 in range(j, cols):
                    res = get_sum(i, j, i2, j2)
                    if res == 10:
                        if v[i2][j2] == 0:
                            continue
                        rev.append(((i, j), (i2, j2)))
                        for a in range(i, i2 + 1):
                            for b in range(j, j2 + 1):
                                cnt[a][b] += 1
                    elif res > 10:
                        break

    for it in rev:
        x1, y1 = it[0]
        x2, y2 = it[1]
        flag = True
        for i in range(x1, x2 + 1):
            for j in range(y1, y2 + 1):
                if cnt[i][j] != 1:
                    flag = False
                    break
            if not flag:
                break
        if flag:
            ret.append(((x1, y1), (x2, y2)))
    return ret

def is_cross(x1, y1, x2, y2, x3, y3, x4, y4):
    for i in range(x1, x2 + 1):
        for j in range(y1, y2 + 1):
            if x3 <= i <= x4 and y3 <= j <= y4:
                return True
    return False

def dfs(node, vis, ma, graph):
    if vis[node]:
        return False
    vis[node] = 1
    for nxt in graph[node]:
        if ma[nxt + 1000] == -1 or dfs(ma[nxt + 1000], vis, ma, graph):
            ma[nxt + 1000] = node
            return True
    return False

# Final function will be converted next.
ans = []
total_score = 0

def backtracking():
    global total_score
    score = 0
    while True:
        fo = find_only()
        if not fo:
            break
        for it in fo:
            x1, y1 = it[0]
            x2, y2 = it[1]
            ans.append(it)
            for i in range(x1, x2 + 1):
                for j in range(y1, y2 + 1):
                    if v[i][j]:
                        score += 1
                    v[i][j] = 0
                    update(i, j, 0, 0, 0, cols - 1)

    tmp = []
    res = find_rect()
    _size = len(res)
    graph = [[] for _ in range(_size)]
    vis = [0] * _size
    ma = [-1] * 2000

    for i in range(_size):
        for j in range(i + 1, _size):
            if not is_cross(res[i][0][0], res[i][0][1], res[i][1][0], res[i][1][1],
                            res[j][0][0], res[j][0][1], res[j][1][0], res[j][1][1]):
                graph[i].append(j)
                graph[j].append(i)

    for i in range(_size):
        if not vis[i]:
            vis = [0] * _size
            dfs(i, vis, ma, graph)

    for i in range(1000, 1000 + _size):
        if ma[i] != -1:
            f1 = ma[i]
            f2 = i - 1000
            tmp.append(res[f1])
            tmp.append(res[f2])

    for it in tmp:
        x1, y1 = it[0]
        x2, y2 = it[1]
        ans.append(it)
        for i in range(x1, x2 + 1):
            for j in range(y1, y2 + 1):
                if v[i][j]:
                    score += 1
                v[i][j] = 0
                update(i, j, 0, 0, 0, cols - 1)

    total_score += score
    if score:
        backtracking()



def solve(grid, row, col):
    global v, rows, cols
    rows = row
    cols = col

    for i in range(rows):
        for j in range(cols):
            v[i][j] = grid[i][j]

    # 입력은 외부에서 처리된다고 가정 (v 배열은 미리 채워짐)
    for i in range(rows):
        mk(i, 0, 0, cols - 1)

    backtracking()
    print("예상 점수 : ", total_score)
    print(ans)
    return ans

