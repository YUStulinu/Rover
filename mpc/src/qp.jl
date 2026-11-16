# A quadratic-program solver: minimize 1/2 x'Px + q'x subject to l <= Ax <= u,
# by the alternating direction method of multipliers as in OSQP (Stellato et
# al., 2020): one Cholesky factorization of P + sigma I + rho A'A, then
# cheap iterations with over-relaxation, stopping on primal and dual
# residuals.

struct QPResult
    x::Vector{Float64}
    y::Vector{Float64}
    iterations::Int
    primal::Float64
    dual::Float64
    converged::Bool
end

function solve_qp(P::AbstractMatrix, q::AbstractVector, A::AbstractMatrix, l::AbstractVector, u::AbstractVector;
                  rho = 0.1, sigma = 1e-6, alpha = 1.6, max_iter = 4000, eps_abs = 1e-5, eps_rel = 1e-5,
                  x0 = zeros(length(q)))
    n, m = length(q), size(A, 1)
    F = cholesky(Symmetric(Matrix(P) + sigma * I + rho * (A' * A)))
    x = copy(x0)
    z = clamp.(A * x, l, u)
    y = zeros(m)
    primal = dual = Inf
    for k in 1:max_iter
        xt = F \ (sigma * x - q + A' * (rho * z - y))
        zt = A * xt
        x = alpha * xt + (1 - alpha) * x
        zr = alpha * zt + (1 - alpha) * z
        znew = clamp.(zr + y / rho, l, u)
        y = y + rho * (zr - znew)
        z = znew
        if k % 10 == 0
            Ax = A * x
            primal = norm(Ax - z, Inf)
            dual = norm(P * x + q + A' * y, Inf)
            ep = eps_abs + eps_rel * max(norm(Ax, Inf), norm(z, Inf))
            ed = eps_abs + eps_rel * max(norm(P * x, Inf), norm(A' * y, Inf), norm(q, Inf))
            if primal < ep && dual < ed
                return QPResult(x, y, k, primal, dual, true)
            end
        end
    end
    QPResult(x, y, max_iter, primal, dual, false)
end
