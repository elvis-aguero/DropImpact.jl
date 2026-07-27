@testset "residual.jl" begin
    @testset "R(X) evaluates to finite values (DAG smoke test)" begin
        p = Params(We=1.0, Bo=0.2, Oh=0.05, M=6, L=6, N=2, b=6.0, h0=2.0, nq=16)
        lvl0 = initial_level(p)
        hist = SimHistory(lvl0, lvl0)
        R, _ = build_residual(hist, 1e-3, p)
        X0 = warm_start(nothing, p.N)
        Rval = R(X0)
        @test length(Rval) == p.N + 2
        @test all(isfinite, Rval)
    end

    @testset "ForwardDiff.jacobian vs central finite differences" begin
        p = Params(We=1.0, Bo=0.2, Oh=0.05, M=6, L=6, N=2, b=6.0, h0=2.0, nq=16)
        lvl0 = initial_level(p)
        hist = SimHistory(lvl0, lvl0)
        R, _ = build_residual(hist, 1e-3, p)
        X0 = warm_start(nothing, p.N)

        Jad = ForwardDiff.jacobian(R, X0)
        n = length(X0)
        Jfd = zeros(n, n)
        h = 1e-6
        for j in 1:n
            Xp = copy(X0); Xp[j] += h
            Xm = copy(X0); Xm[j] -= h
            Jfd[:, j] = (R(Xp) .- R(Xm)) ./ (2h)
        end
        @test Jad ≈ Jfd rtol = 1e-3 atol = 1e-5
    end

    @testset "tangency row unchanged by accel closure: matches direct eq:tangency-explicit" begin
        p = Params(We=1.0, Bo=0.2, Oh=0.05, M=6, L=6, N=2, b=6.0, h0=2.0, nq=16)
        lvl0 = initial_level(p)
        hist = SimHistory(lvl0, lvl0)
        R, (kappa, alpha, lambda, gam, kappa_cm, mu) = build_residual(hist, 1e-3, p)
        X0 = warm_start(nothing, p.N)
        am, beta, zcm = unpack_state(X0, kappa, alpha, lambda, gam, kappa_cm, mu, p)
        theta_c = X0[end]
        rc = forward_map_r(beta, theta_c, p.L)
        rth = SpectralKM.r_theta(beta, theta_c, p.L)
        zdth = SpectralKM.zd_theta(beta, theta_c, p.L)
        bessel_sum = sum(am[m+1] * p.k[m+1] * besselj1(p.k[m+1] * rc) for m in 0:p.M)
        expected = -bessel_sum * rth - zdth
        @test R(X0)[end] ≈ expected rtol = 1e-10
    end
end
