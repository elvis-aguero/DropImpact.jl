@testset "conditioning.jl (Jacobian conditioning regression checks)" begin
    @testset "piston mode (m=0, k_0=0) has zero pressure response" begin
        for M in (1, 5, 10)
            p = Params(We=1.0, Bo=0.2, Oh=0.05, M=M, L=5, N=1, b=6.0, h0=2.0, nq=16)
            bath = BathModeState(M)
            kappa, _ = bath_affine(bath, bath, p, 0.01, 0.01)
            @test kappa[1] == 0.0
        end
    end

    @testset "inner-system conditioning is flat in dt (theta_c held out of the system)" begin
        # The central conditioning claim of the nested closure: with theta_c a parameter
        # rather than an unknown, every column of the inner Galerkin matrix carries the
        # same O(dt^2) affine slope, and a uniform scalar factor cannot change a condition
        # number. The joint system this replaced grew by ~6 orders of magnitude over the
        # same dt range (design doc §subsec:corrections).
        p = Params(We=1.0, Bo=0.1, Oh=0.05, M=10, L=10, N=1, b=6.0, h0=3.0, nq=20)
        lvl0 = initial_level(p)
        hist = SimHistory(lvl0, lvl0)
        xc = cos(0.3)
        q = contact_quad(xc, p)
        conds = Float64[]
        for dt in (1e-2, 1e-3, 1e-4, 1e-5)
            kappa, alpha, lambda, gam, kappa_cm, mu = step_affine(hist, dt, p)
            R = chat -> residual(chat, xc, q, kappa, alpha, lambda, gam, kappa_cm, mu, p)
            push!(conds, cond(ForwardDiff.jacobian(R, zeros(p.N + 1))))
        end
        @test maximum(conds) / minimum(conds) < 10.0
        @test maximum(conds) < 1e4        # order unity, not 1e18
    end
end
