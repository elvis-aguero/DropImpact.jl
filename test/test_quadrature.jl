@testset "quadrature.jl" begin
    @testset "Gauss-Legendre nodes/weights on [-1,1]" begin
        nodes, weights = gauss_legendre_nodes(8)
        @test sum(weights) ≈ 2.0
        @test sum(weights .* nodes) ≈ 0.0 atol = 1e-14        # odd moment vanishes
        # n-point rule is exact for polynomials up to degree 2n-1
        for k in 0:15
            exact = isodd(k) ? 0.0 : 2 / (k + 1)
            @test sum(weights .* nodes .^ k) ≈ exact atol = 1e-10
        end
    end

    @testset "gauss_quad on [xc,1]" begin
        nodes, weights = gauss_legendre_nodes(20)
        for xc in (-0.9, 0.0, 0.5, 0.95)
            for k in 0:5
                exact = (1 - xc^(k + 1)) / (k + 1)
                @test gauss_quad(x -> x^k, xc, nodes, weights) ≈ exact atol = 1e-10
            end
            @test gauss_quad(x -> besselj0(2.3 * x), xc, nodes, weights) isa Float64
        end
    end

    @testset "gauss_quad generic in eltype(xc) (ForwardDiff through xc)" begin
        nodes, weights = gauss_legendre_nodes(20)
        g(xc) = gauss_quad(x -> x^3, xc, nodes, weights)
        xc0 = 0.4
        # d/dxc ∫_{xc}^1 x^3 dx = -xc^3
        @test ForwardDiff.derivative(g, xc0) ≈ -xc0^3 atol = 1e-8
    end
end
