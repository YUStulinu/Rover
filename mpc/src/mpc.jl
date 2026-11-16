# Model predictive control of a unicycle (x, y, heading; commands: speed and
# yaw rate) following a path among obstacles. Each control step solves a
# sequence of quadratic programs: the model is linearized along the
# trajectory predicted from the previous solution, the states are written in
# terms of the commands (condensed form), and the QP keeps the commands and
# their rates within limits and every predicted position on the free side
# of a half-plane through the nearest obstacle point.

Base.@kwdef struct MpcParams
    N::Int = 20            # horizon steps
    dt::Float64 = 0.1      # seconds per step
    v_max::Float64 = 0.8
    v_min::Float64 = -0.1
    w_max::Float64 = 1.2
    dv_max::Float64 = 0.12 # per step (1.2 m/s^2)
    dw_max::Float64 = 0.3
    v_ref::Float64 = 0.6
    q_pos::Float64 = 10.0
    q_head::Float64 = 1.0
    r_v::Float64 = 0.1
    r_w::Float64 = 0.05
    rd_v::Float64 = 1.0
    rd_w::Float64 = 0.5
    clearance::Float64 = 0.45  # rover radius plus a margin
    sqp_iterations::Int = 2
end

mutable struct Mpc
    p::MpcParams
    U::Matrix{Float64}  # 2 x N, the last solution (shifted at each call)
    u_prev::Vector{Float64}
    progress::Int       # index of the path point reached
    solve_ms::Vector{Float64}
    qp_iterations::Vector{Int}
end

Mpc(p::MpcParams = MpcParams()) = Mpc(p, zeros(2, p.N), zeros(2), 1, Float64[], Int[])

wrap(a) = atan(sin(a), cos(a))

function rollout(z0, U, dt)
    N = size(U, 2)
    Z = zeros(3, N + 1)
    Z[:, 1] = z0
    for k in 1:N
        x, y, th = Z[:, k]
        v, w = U[:, k]
        Z[:, k+1] = [x + dt * v * cos(th), y + dt * v * sin(th), th + dt * w]
    end
    Z
end

"""Reference states along the path from the point nearest to z (positions every v dt, headings of the path)."""
function reference(c::Mpc, z, path)
    p = c.p
    # progress: the nearest path point not far behind the last one
    lo, hi = max(1, c.progress - 5), min(length(path), c.progress + 60)
    best, bi = Inf, c.progress
    for i in lo:hi
        d = hypot(path[i][1] - z[1], path[i][2] - z[2])
        if d < best
            best, bi = d, i
        end
    end
    c.progress = bi
    # path points are 0.1 m apart
    R = zeros(3, p.N)
    step = p.v_ref * p.dt / 0.1
    for k in 1:p.N
        i = min(length(path), bi + round(Int, k * step))
        j = min(length(path), i + 1)
        i2 = i == j ? max(1, i - 1) : i
        heading = atan(path[j][2] - path[i2][2], path[j][1] - path[i2][1])
        R[:, k] = [path[i][1], path[i][2], heading]
    end
    R
end

"""Nearest point of the obstacles (boxes and walls) to (x, y), and its distance."""
function nearest_obstacle(x, y, size, boxes)
    best = (Inf, x, y)
    for b in boxes
        cx, cy = clamp(x, b[1][1], b[2][1]), clamp(y, b[1][2], b[2][2])
        d = hypot(x - cx, y - cy)
        d < best[1] && (best = (d, cx, cy))
    end
    for (d, cx, cy) in ((x, 0.0, y), (size[1] - x, size[1], y), (y, x, 0.0), (size[2] - y, x, size[2]))
        d < best[1] && (best = (d, cx, cy))
    end
    best
end

"""One control step from state z = (x, y, heading): the command (v, w) to apply now."""
function control!(c::Mpc, z, path, size, boxes)
    p = c.p
    N, dt = p.N, p.dt
    t0 = time()
    # warm start: the previous solution shifted by one step
    U = hcat(c.U[:, 2:end], c.U[:, end])
    R = reference(c, z, path)
    # slow down near the end of the path
    remaining = hypot(path[end][1] - z[1], path[end][2] - z[2])
    for k in 1:N
        d = hypot(R[1, k] - path[end][1], R[2, k] - path[end][2])
        if d < 1e-6 && remaining < 0.3
            R[:, k] = [path[end][1], path[end][2], R[3, k]]
        end
    end
    Q = Diagonal(repeat([p.q_pos, p.q_pos, p.q_head], N))
    Rw = Diagonal(repeat([p.r_v, p.r_w], N))
    Rd = Diagonal(repeat([p.rd_v, p.rd_w], N))
    # difference operator: (D U)_k = u_k - u_{k-1}, with u_{-1} the last applied command
    D = zeros(2N, 2N)
    for k in 1:N
        D[2k-1:2k, 2k-1:2k] = I(2)
        k > 1 && (D[2k-1:2k, 2k-3:2k-2] = -I(2))
    end
    d0 = zeros(2N)
    d0[1:2] = c.u_prev
    iters = 0
    for it in 1:p.sqp_iterations
        Zb = rollout(z, U, dt)
        # sensitivities S: d z_k / d u_j (3N x 2N)
        S = zeros(3N, 2N)
        for j in 1:N
            th, v = Zb[3, j], U[1, j]
            B = dt * [cos(th) 0; sin(th) 0; 0 1]
            blk = B
            S[3j-2:3j, 2j-1:2j] = blk
            for k in j+1:N
                thk, vk = Zb[3, k], U[1, k]
                A = [1 0 -dt * vk * sin(thk); 0 1 dt * vk * cos(thk); 0 0 1]
                blk = A * blk
                S[3k-2:3k, 2j-1:2j] = blk
            end
        end
        Ub = vec(U)
        E = vec(Zb[:, 2:end]) - vec(R)
        for k in 1:N
            E[3k] = wrap(E[3k])
        end
        P = 2 * (S' * Q * S + Rw + D' * Rd * D)
        q = 2 * (S' * Q * E + Rw * Ub + D' * Rd * (D * Ub - d0))
        # constraints on the change dU
        rows = Vector{Vector{Float64}}()
        lo = Float64[]
        hi = Float64[]
        umin = repeat([p.v_min, -p.w_max], N)
        umax = repeat([p.v_max, p.w_max], N)
        for i in 1:2N
            r = zeros(2N)
            r[i] = 1
            push!(rows, r)
            push!(lo, umin[i] - Ub[i])
            push!(hi, umax[i] - Ub[i])
        end
        rate = repeat([p.dv_max, p.dw_max], N)
        DU = D * Ub - d0
        for i in 1:2N
            push!(rows, D[i, :])
            push!(lo, -rate[i] - DU[i])
            push!(hi, rate[i] - DU[i])
        end
        for k in 1:N
            x, y = Zb[1, k+1], Zb[2, k+1]
            d, ox, oy = nearest_obstacle(x, y, size, boxes)
            d > 1.5 && continue
            n = ((x - ox) / max(d, 1e-6), (y - oy) / max(d, 1e-6))
            push!(rows, n[1] * S[3k-2, :] + n[2] * S[3k-1, :])
            push!(lo, p.clearance - d)
            push!(hi, Inf)
        end
        A = reduce(vcat, (r' for r in rows))
        res = solve_qp(P, q, A, lo, hi; max_iter = 2000)
        iters += res.iterations
        dU = res.x
        # take the step, then respect the limits exactly
        U = reshape(Ub + dU, 2, N)
        for k in 1:N
            U[1, k] = clamp(U[1, k], p.v_min, p.v_max)
            U[2, k] = clamp(U[2, k], -p.w_max, p.w_max)
        end
    end
    c.U = U
    c.u_prev = U[:, 1]
    push!(c.solve_ms, 1000 * (time() - t0))
    push!(c.qp_iterations, iters)
    (U[1, 1], U[2, 1])
end
