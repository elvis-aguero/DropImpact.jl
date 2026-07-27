@testset "legendre.jl" begin
    @testset "explicit low-degree values" begin
        for x in (-0.7, 0.0, 0.3, 0.9999)
            @test legendre_P(0, x) ≈ 1.0
            @test legendre_P(1, x) ≈ x
            @test legendre_P(2, x) ≈ (3x^2 - 1) / 2
            @test legendre_P(3, x) ≈ (5x^3 - 3x) / 2
        end
    end

    @testset "derivative vs finite difference" begin
        h = 1e-6
        for l in 0:6, x in (-0.5, 0.2, 0.8)
            fd = (legendre_P(l, x + h) - legendre_P(l, x - h)) / 2h
            @test legendre_dP(l, x) ≈ fd atol = 1e-6
        end
    end

    @testset "high-degree stability (MATLAB's known failure mode, l~100+)" begin
        for l in (50, 100, 200)
            P = legendre_P_table(l, 0.999)
            @test all(isfinite, P)
            @test all(abs.(P) .<= 1.0 + 1e-9)  # |P_l(x)| <= 1 on [-1,1]
        end
    end

    @testset "bonnet_H antiderivative vs direct quadrature" begin
        # H_l^{(k)}(1) - H_l^{(k)}(xc) should equal ∫_{xc}^1 x^k P_l(x) dx.
        nodes, weights = gauss_legendre_nodes(60)
        for lmax in (0, 2, 5), kmax in (0, 1, 3), xc in (-0.6, 0.1, 0.7)
            H1 = bonnet_H(lmax, kmax, 1.0)
            Hxc = bonnet_H(lmax, kmax, xc)
            for l in 0:lmax, k in 0:kmax
                direct = gauss_quad(x -> x^k * legendre_P(l, x), xc, nodes, weights)
                @test H1[l+1, k+1] - Hxc[l+1, k+1] ≈ direct atol = 1e-9
            end
        end
    end
end
