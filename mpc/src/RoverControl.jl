"""
RoverControl: planning and model predictive control for the simulated rover,
and the closed loop with the simulator (Rust) and the visual-inertial
odometry (C++) as separate processes.
"""
module RoverControl

using LinearAlgebra
using Printf
using Random

include("json.jl")
using .MiniJSON

include("planner.jl")
include("qp.jl")
include("mpc.jl")
include("closed_loop.jl")

export Grid, astar, shortcut, resample, path_length, distance_to_path, line_free, free
export solve_qp, QPResult, Mpc, MpcParams, control!, rollout, nearest_obstacle
export parse_json, to_json, run_mission

end
