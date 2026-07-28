# Gauss-Legendre quadrature via the Golub-Welsch algorithm (Jacobi tridiagonal
# eigendecomposition), matching the working pattern already used in the km-viscous-drop
# sister repo. Nodes/weights are precomputed Float64 constants (they never depend on the
# Newton unknown X), so this file has no AD-genericity requirement — only the quadrature
# SUM applied to an X-dependent integrand (design doc eq:gauss-quad) needs to stay generic,
# which is handled by gauss_quad's own type signature.

"""
    gauss_legendre_nodes(n) -> (nodes::Vector{Float64}, weights::Vector{Float64})

`n`-point Gauss-Legendre nodes/weights on `[-1,1]`, via the Golub-Welsch algorithm:
eigenvalues of the symmetric tridiagonal Jacobi matrix (zero diagonal, off-diagonals
`b_k = k/√(4k²-1)`) are the nodes; weights are `2 * (first eigenvector component)²`.
"""
function gauss_legendre_nodes(n::Integer)
    n < 1 && throw(ArgumentError("n must be ≥ 1"))
    if n == 1
        return ([0.0], [2.0])
    end
    offdiag = [k / sqrt(4k^2 - 1) for k in 1:(n-1)]
    J = SymTridiagonal(zeros(n), offdiag)
    F = eigen(J)
    nodes = F.values
    weights = 2 .* (F.vectors[1, :] .^ 2)
    return (nodes, weights)
end

"""
    gauss_quad(g, xc, nodes, weights) -> value

`∫_{xc}^{1} g(x) dx` by mapping the `n`-point Gauss-Legendre rule on `[-1,1]` to
`[xc,1]` via `x = xc + (1+s)(1-xc)/2` (design doc eq:gauss-quad). `xc` may carry a
`ForwardDiff.Dual` type; the affine node map and the returned sum are generic in
`eltype(xc)` so differentiation w.r.t. `xc` flows through correctly.
"""
function gauss_quad(g, xc, nodes::Vector{Float64}, weights::Vector{Float64})
    half = (1 - xc) / 2
    total = sum(weights[i] * g(xc + (1 + nodes[i]) * half) for i in eachindex(nodes))
    return half * total
end
