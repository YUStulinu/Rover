# Missions in several random worlds, each driven twice: with the true pose and
# with the visual-inertial odometry's pose. One JSON per run in results/missions
# (metrics, the world, the planned paths and the trajectories at 5 Hz).
#
#   julia --project=mpc mpc/run_missions.jl [SEEDS...]
using RoverControl

const ROOT = normpath(joinpath(@__DIR__, ".."))
exe(p) = Sys.iswindows() ? p * ".exe" : p
const SIM = joinpath(ROOT, "sim", "target", "release", exe("rover-sim"))
const STREAM = joinpath(ROOT, "vio", "build", exe("rover-stream"))

function main(seeds)
    outdir = joinpath(ROOT, "results", "missions")
    mkpath(outdir)
    for seed in seeds, use_vio in (false, true)
        tag = "seed$(seed)_$(use_vio ? "vio" : "gt")"
        scratch = joinpath(ROOT, "run", "missions", tag)
        metrics, log, paths, world = run_mission(; sim_exe = SIM, stream_exe = STREAM, seed, out_dir = scratch, use_vio)
        keep = 1:4:length(log["t"])  # 20 Hz -> 5 Hz
        doc = Dict(
            "metrics" => metrics,
            "world" => Dict("size" => world["size"], "boxes" => world["boxes"]),
            "paths" => [[collect(p) for p in path[1:2:end]] for path in paths],
            "trajectory" => Dict(k => [round(log[k][i]; digits = 4) for i in keep] for k in ("t", "x", "y", "yaw", "ex", "ey")),
        )
        open(joinpath(outdir, tag * ".json"), "w") do f
            write(f, to_json(doc))
        end
        rm(scratch; recursive = true, force = true)  # the recorded images are not kept
        println(tag, " ", to_json(metrics))
        flush(stdout)
    end
end

main(isempty(ARGS) ? (1:5) : parse.(Int, ARGS))
