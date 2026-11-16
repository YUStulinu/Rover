# Path planning on an occupancy grid: obstacles inflated by the rover's
# radius plus a margin, A* with 8 neighbors and the octile distance as the
# heuristic, then shortcutting along lines of sight.

struct Grid
    res::Float64
    nx::Int
    ny::Int
    occ::BitMatrix  # occ[i, j]: cell with center ((i - 0.5) res, (j - 0.5) res) is blocked
end

"""Occupancy grid of a room of size (W, D) with boxes (min, max corners), inflated by `inflate`."""
function Grid(size, boxes; res = 0.1, inflate = 0.5)
    nx, ny = ceil(Int, size[1] / res), ceil(Int, size[2] / res)
    occ = falses(nx, ny)
    for i in 1:nx, j in 1:ny
        x, y = (i - 0.5) * res, (j - 0.5) * res
        if x < inflate || y < inflate || x > size[1] - inflate || y > size[2] - inflate
            occ[i, j] = true
            continue
        end
        for b in boxes
            cx = clamp(x, b[1][1], b[2][1])
            cy = clamp(y, b[1][2], b[2][2])
            if (x - cx)^2 + (y - cy)^2 < inflate^2
                occ[i, j] = true
                break
            end
        end
    end
    Grid(res, nx, ny, occ)
end

cell(g::Grid, x, y) = (clamp(floor(Int, x / g.res) + 1, 1, g.nx), clamp(floor(Int, y / g.res) + 1, 1, g.ny))
center(g::Grid, c) = ((c[1] - 0.5) * g.res, (c[2] - 0.5) * g.res)
free(g::Grid, c) = 1 <= c[1] <= g.nx && 1 <= c[2] <= g.ny && !g.occ[c[1], c[2]]
free(g::Grid, x::Real, y::Real) = free(g, cell(g, x, y))

"""A* from (x0, y0) to (x1, y1); the path as points, or nothing."""
function astar(g::Grid, start, goal)
    s, t = cell(g, start...), cell(g, goal...)
    (free(g, s) && free(g, t)) || return nothing
    h(c) = (dx = abs(c[1] - t[1]); dy = abs(c[2] - t[2]); max(dx, dy) + (sqrt(2) - 1) * min(dx, dy))
    gcost = Dict(s => 0.0)
    parent = Dict{Tuple{Int,Int},Tuple{Int,Int}}()
    # a binary heap of (f, cell)
    heap = [(h(s), s)]
    closed = Set{Tuple{Int,Int}}()
    push_heap!(x) = (push!(heap, x); i = length(heap); while i > 1 && heap[i>>1][1] > heap[i][1]; heap[i], heap[i>>1] = heap[i>>1], heap[i]; i >>= 1; end)
    function pop_heap!()
        top = heap[1]
        last = pop!(heap)
        if !isempty(heap)
            heap[1] = last
            i = 1
            while true
                l, r, m = 2i, 2i + 1, i
                l <= length(heap) && heap[l][1] < heap[m][1] && (m = l)
                r <= length(heap) && heap[r][1] < heap[m][1] && (m = r)
                m == i && break
                heap[i], heap[m] = heap[m], heap[i]
                i = m
            end
        end
        top
    end
    while !isempty(heap)
        _, c = pop_heap!()
        c in closed && continue
        if c == t
            path = [c]
            while haskey(parent, path[end])
                push!(path, parent[path[end]])
            end
            pts = [center(g, p) for p in reverse(path)]
            pts[1] = (start[1], start[2])
            pts[end] = (goal[1], goal[2])
            return pts
        end
        push!(closed, c)
        for dx in -1:1, dy in -1:1
            (dx == 0 && dy == 0) && continue
            n = (c[1] + dx, c[2] + dy)
            free(g, n) || continue
            # no cutting corners
            (dx != 0 && dy != 0) && (!free(g, (c[1] + dx, c[2])) || !free(g, (c[1], c[2] + dy))) && continue
            nc = gcost[c] + (dx != 0 && dy != 0 ? sqrt(2) : 1.0)
            if nc < get(gcost, n, Inf)
                gcost[n] = nc
                parent[n] = c
                push_heap!((nc + h(n), n))
            end
        end
    end
    nothing
end

"""Whether the straight segment between two points stays in free cells."""
function line_free(g::Grid, a, b)
    n = max(2, ceil(Int, hypot(b[1] - a[1], b[2] - a[2]) / (g.res / 3)))
    all(free(g, a[1] + (b[1] - a[1]) * k / n, a[2] + (b[2] - a[2]) * k / n) for k in 0:n)
end

"""Shortcut a path: keep a point only when the next ones cannot be reached in a straight line."""
function shortcut(g::Grid, pts)
    out = [pts[1]]
    i = 1
    while i < length(pts)
        j = length(pts)
        while j > i + 1 && !line_free(g, pts[i], pts[j])
            j -= 1
        end
        push!(out, pts[j])
        i = j
    end
    out
end

"""Points every `step` meters along a polyline."""
function resample(pts, step)
    out = [pts[1]]
    carry = 0.0
    for k in 1:length(pts)-1
        a, b = pts[k], pts[k+1]
        L = hypot(b[1] - a[1], b[2] - a[2])
        s = step - carry
        while s <= L
            push!(out, (a[1] + (b[1] - a[1]) * s / L, a[2] + (b[2] - a[2]) * s / L))
            s += step
        end
        carry = L - (s - step)
    end
    hypot(out[end][1] - pts[end][1], out[end][2] - pts[end][2]) > 1e-9 && push!(out, pts[end])
    out
end

path_length(pts) = sum(hypot(pts[k+1][1] - pts[k][1], pts[k+1][2] - pts[k][2]) for k in 1:length(pts)-1; init = 0.0)

"""Distance from a point to a polyline."""
function distance_to_path(p, pts)
    best = Inf
    for k in 1:length(pts)-1
        a, b = pts[k], pts[k+1]
        vx, vy = b[1] - a[1], b[2] - a[2]
        L2 = vx^2 + vy^2
        u = L2 > 0 ? clamp(((p[1] - a[1]) * vx + (p[2] - a[2]) * vy) / L2, 0, 1) : 0.0
        best = min(best, hypot(p[1] - a[1] - u * vx, p[2] - a[2] - u * vy))
    end
    best
end
