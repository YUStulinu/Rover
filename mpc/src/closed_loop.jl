# The closed loop: the simulator (rover-sim serve) and, optionally, the
# odometry (rover-stream) run as processes; at every camera frame the
# controller takes the pose (the true one, or the odometry's), plans when a
# new goal starts, and sends the next command.

function request(p, msg)
    println(p, to_json(msg))
    flush(p)
    parse_json(readline(p))
end

tup(v) = (Float64(v[1]), Float64(v[2]))

"""Goals in free space: the corners of the free lane around the room, then
points in the middle drawn at random among free cells, alternated."""
function mission_goals(grid, size, seed; n = 6)
    rng = MersenneTwister(seed)
    corners = [(size[1] - 1.5, 1.5), (size[1] - 1.5, size[2] - 1.5), (1.5, size[2] - 1.5)]
    middle = Tuple{Float64,Float64}[]
    while length(middle) < 3
        x, y = 3.0 + rand(rng) * (size[1] - 6), 3.0 + rand(rng) * (size[2] - 6)
        free(grid, x, y) && all(hypot(x - m[1], y - m[2]) > 2.0 for m in middle) && push!(middle, (x, y))
    end
    goals = Tuple{Float64,Float64}[]
    for k in 1:3
        push!(goals, middle[k])
        push!(goals, corners[k])
    end
    push!(goals, (1.5, 1.5))
    goals[1:min(n + 1, length(goals))]
end

"""
Drive the rover through a list of goals. With `vio` the controller sees the
odometry's pose (put in the map frame through the known start pose);
otherwise the true pose. Returns the metrics, the logged trajectory, the planned paths and the
world as the simulator described it.
"""
function run_mission(; sim_exe, stream_exe = "", seed = 1, obstacles = 6, out_dir, use_vio = false, max_seconds = 400.0,
                     params = MpcParams(), verbose = false)
    mkpath(out_dir)
    sim = open(`$sim_exe serve`, "r+")
    x0, y0, yaw0 = 1.5, 1.5, 0.0
    world = request(sim, Dict("op" => "reset", "seed" => seed, "obstacles" => obstacles, "out" => out_dir, "x" => x0, "y" => y0, "yaw" => yaw0))
    size = tup(world["size"])
    boxes = [(tup(b["min"]), tup(b["max"])) for b in world["boxes"]]
    radius = Float64(world["radius"])
    grid = Grid(size, boxes; inflate = radius + 0.25)
    goals = mission_goals(grid, size, seed)
    vio = use_vio ? open(`$stream_exe $out_dir`, "r+") : nothing
    est = (x0, y0, yaw0)
    log = Dict{String,Vector{Float64}}(k => Float64[] for k in ("t", "x", "y", "yaw", "ex", "ey", "eyaw", "v", "w", "cross", "clear"))
    function step!(v, w)
        r = request(sim, Dict("op" => "step", "v" => v, "w" => w))
        tr = r["truth"]
        truth = (Float64(tr["x"]), Float64(tr["y"]), Float64(tr["yaw"]))
        if vio !== nothing
            for m in r["imu"]
                println(vio, "imu ", join((@sprintf("%.9f", x) for x in m), " "))
            end
            f = r["frame"]
            if f !== nothing
                println(vio, "frame ", @sprintf("%.6f", f["t"]), " ", f["left"], " ", f["right"])
                flush(vio)
                ans = split(readline(vio))
                if ans[1] == "pose"
                    xv, yv = parse(Float64, ans[3]), parse(Float64, ans[4])
                    qx, qy, qz, qw = (parse(Float64, ans[k]) for k in 6:9)
                    yawv = atan(2 * (qw * qz + qx * qy), 1 - 2 * (qy^2 + qz^2))
                    est = (x0 + cos(yaw0) * xv - sin(yaw0) * yv, y0 + sin(yaw0) * xv + cos(yaw0) * yv, wrap(yaw0 + yawv))
                end
            end
        else
            est = truth
        end
        r, truth
    end
    # stand still first: the odometry finds gravity and the biases
    for _ in 1:40
        step!(0.0, 0.0)
    end
    mpc = Mpc(params)
    reached = 0
    collided = false
    t = 0.0
    planned = 0.0
    paths = Vector{Vector{Tuple{Float64,Float64}}}()
    for g in goals
        raw = astar(grid, (est[1], est[2]), g)
        raw === nothing && (verbose && println("no path to $g"); break)
        path = resample(shortcut(grid, raw), 0.1)
        push!(paths, path)
        planned += path_length(path)
        mpc.progress = 1
        while t < max_seconds
            if hypot(est[1] - g[1], est[2] - g[2]) < 0.25
                reached += 1
                break
            end
            v, w = control!(mpc, [est[1], est[2], est[3]], path, size, boxes)
            r, truth = step!(v, w)
            t = Float64(r["t"])
            push!(log["t"], t)
            push!(log["x"], truth[1])
            push!(log["y"], truth[2])
            push!(log["yaw"], truth[3])
            push!(log["ex"], est[1])
            push!(log["ey"], est[2])
            push!(log["eyaw"], est[3])
            push!(log["v"], v)
            push!(log["w"], w)
            push!(log["cross"], distance_to_path((truth[1], truth[2]), path))
            push!(log["clear"], nearest_obstacle(truth[1], truth[2], size, boxes)[1] - radius)
            if r["collided"] == true
                collided = true
                break
            end
        end
        collided && break
        verbose && @printf("goal %s: reached %d, t %.1f s\n", g, reached, t)
    end
    request(sim, Dict("op" => "close"))
    close(sim)
    if vio !== nothing
        println(vio, "quit")
        close(vio)
    end
    n = length(log["t"])
    est_err = [hypot(log["ex"][k] - log["x"][k], log["ey"][k] - log["y"][k]) for k in 1:n]
    travelled = sum(hypot(log["x"][k+1] - log["x"][k], log["y"][k+1] - log["y"][k]) for k in 1:n-1; init = 0.0)
    metrics = Dict(
        "seed" => seed, "localization" => use_vio ? "odometry" : "ground truth", "goals" => length(goals), "reached" => reached,
        "collided" => collided, "seconds" => t, "planned_m" => planned, "travelled_m" => travelled,
        "cross_track_rms_m" => sqrt(sum(abs2, log["cross"]) / max(n, 1)), "cross_track_max_m" => maximum(log["cross"]; init = 0.0),
        "min_clearance_m" => minimum(log["clear"]; init = Inf),
        "localization_rms_m" => sqrt(sum(abs2, est_err) / max(n, 1)), "localization_final_m" => n > 0 ? est_err[end] : 0.0,
        "mpc_ms_mean" => sum(mpc.solve_ms) / max(length(mpc.solve_ms), 1), "mpc_ms_max" => maximum(mpc.solve_ms; init = 0.0),
        "qp_iterations_mean" => sum(mpc.qp_iterations) / max(length(mpc.qp_iterations), 1),
    )
    metrics, log, paths, world
end
