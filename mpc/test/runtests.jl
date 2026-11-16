using RoverControl
using LinearAlgebra
using Random
using Test

@testset "RoverControl" begin
    @testset "json" begin
        s = """{"a": [1, 2.5, -3e2], "b": {"c": "x\\"y", "d": true, "e": null}}"""
        v = parse_json(s)
        @test v["a"] == [1, 2.5, -300.0]
        @test v["b"]["c"] == "x\"y" && v["b"]["d"] === true && v["b"]["e"] === nothing
        @test parse_json(to_json(v)) == v
    end

    @testset "qp" begin
        # against a known solution: minimize (x1-1)^2 + (x2-2)^2 with x1 + x2 <= 1, x >= 0
        P = 2.0 * Matrix(I, 2, 2)
        q = [-2.0, -4.0]
        A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
        r = solve_qp(P, q, A, [-Inf, 0.0, 0.0], [1.0, Inf, Inf])
        @test r.converged
        @test isapprox(r.x, [0.0, 1.0]; atol = 1e-3)
        # random box-constrained problems against projected gradient run to convergence
        rng = MersenneTwister(1)
        for _ in 1:20
            n = 8
            M = randn(rng, n, n)
            P = M' * M + I
            q = randn(rng, n)
            A = Matrix(1.0I, n, n)
            r = solve_qp(P, q, A, fill(-0.5, n), fill(0.5, n); eps_abs = 1e-7, eps_rel = 1e-7, max_iter = 20000)
            x = zeros(n)
            step = 1 / opnorm(P)
            for _ in 1:20000
                x = clamp.(x - step * (P * x + q), -0.5, 0.5)
            end
            @test norm(r.x - x, Inf) < 1e-4
        end
    end

    @testset "planner" begin
        boxes = [((4.0, 3.0), (6.0, 7.0))]
        g = Grid((12.0, 10.0), boxes; inflate = 0.5)
        raw = astar(g, (2.0, 5.0), (9.0, 5.0))
        @test raw !== nothing
        path = resample(shortcut(g, raw), 0.1)
        # the path goes around the box and never enters inflated space
        @test all(free(g, p...) for p in path)
        @test all(line_free(g, path[k], path[k+1]) for k in 1:length(path)-1)
        L = path_length(path)
        @test 7.0 < L < 7.0 + 2 * (2.5 + 0.5)
        # a goal inside the obstacle has no path
        @test astar(g, (2.0, 5.0), (5.0, 5.0)) === nothing
    end

    @testset "mpc follows a path in the model" begin
        boxes = [((4.0, 3.0), (6.0, 7.0))]
        g = Grid((12.0, 10.0), boxes; inflate = 0.55)
        path = resample(shortcut(g, astar(g, (2.0, 5.0), (9.0, 5.0))), 0.1)
        c = Mpc()
        z = [2.0, 5.0, 0.0]
        u = [0.0, 0.0]
        worst = 0.0
        clearance = Inf
        for k in 1:400
            v, w = control!(c, z, path, (12.0, 10.0), boxes)
            # the model with a little lag, simulated at the controller rate (50 ms)
            u = u + 0.5 * ([v, w] - u)
            z = rollout(z, reshape(u, 2, 1), 0.05)[:, end]
            worst = max(worst, distance_to_path((z[1], z[2]), path))
            clearance = min(clearance, nearest_obstacle(z[1], z[2], (12.0, 10.0), boxes)[1])
            hypot(z[1] - 9, z[2] - 5) < 0.2 && break
        end
        @test hypot(z[1] - 9, z[2] - 5) < 0.25
        @test worst < 0.3
        @test clearance > 0.3
    end
end
