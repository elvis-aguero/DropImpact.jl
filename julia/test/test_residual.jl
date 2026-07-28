@testset "residual.jl" begin
    # A fixed small problem, with the contact angle held FIXED throughout: theta_c is a
    # parameter of the outer selection, never an unknown of this system.
    mkp() = Params(We=1.0, Bo=0.2, Oh=0.05, M=6, L=6, N=2, b=6.0, h0=2.0, nq=16)
    function setup(theta_c=0.2; dt=1e-3)
        p = mkp()
        lvl0 = initial_level(p)
        hist = SimHistory(lvl0, lvl0)
        kappa, alpha, lambda, gam, kappa_cm, mu = step_affine(hist, dt, p)
        xc = cos(theta_c)
        q = contact_quad(xc, p)
        R = chat -> residual(chat, xc, q, kappa, alpha, lambda, gam, kappa_cm, mu, p)
        return p, hist, xc, q, R, (kappa, alpha, lambda, gam, kappa_cm, mu)
    end

    @testset "inner residual has N+1 rows and is finite" begin
        p, _, _, _, R, _ = setup()
        Rval = R([1e-3, 0.0, 0.0])
        @test length(Rval) == p.N + 1      # NOT N+2: there is no theta_c row
        @test all(isfinite, Rval)
    end

    @testset "ForwardDiff.jacobian vs central finite differences" begin
        p, _, _, _, R, _ = setup()
        chat0 = [1e-3, 0.0, 0.0]
        Jad = ForwardDiff.jacobian(R, chat0)
        n = length(chat0)
        Jfd = zeros(p.N + 1, n)
        h = 1e-7
        for j in 1:n
            Xp = copy(chat0); Xp[j] += h
            Xm = copy(chat0); Xm[j] -= h
            Jfd[:, j] = (R(Xp) .- R(Xm)) ./ (2h)
        end
        @test Jad ≈ Jfd rtol = 1e-4 atol = 1e-12
    end

    @testset "Galerkin matrix is symmetric (self-adjoint pairing, same-basis w-measure test)" begin
        # Design doc §subsubsec:compliance: with b_l carrying BOTH the vertical factor x
        # and the area weight w, and with the test functions equal to the trial basis in
        # the w measure, the frozen-geometry Galerkin matrix is symmetric. That property is
        # what gives the per-step problem an energy functional, and it is destroyed by
        # changing either ingredient -- so it is worth guarding.
        p, _, _, _, R, _ = setup(0.3)
        J = ForwardDiff.jacobian(R, zeros(p.N + 1))
        @test norm(J - J') / norm(J + J') < 1e-8
    end

    @testset "zero pressure gives the free-flight state" begin
        p, _, xc, q, _, (kappa, alpha, lambda, gam, kappa_cm, mu) = setup(0.25)
        am, beta, zcm, f = unpack_state(zeros(p.N + 1), xc, q, kappa, alpha,
                                        lambda, gam, kappa_cm, mu, p)
        @test f ≈ 0 atol = 1e-14                  # no pressure, no force
        @test am ≈ alpha atol = 1e-14             # bath advances on history alone
        @test beta ≈ gam atol = 1e-14
        @test zcm ≈ mu atol = 1e-14
    end

    @testset "C is even in theta, so tangency degenerates at the poles" begin
        # The parity identity the design doc's tangency discussion rests on
        # (eq:tangency-degenerate): C is even in theta at EVERY theta, hence dC/dtheta
        # vanishes identically at theta = 0 and pi for any state whatsoever.
        p = mkp()
        am = [0.01 * (-1)^m / (m + 1) for m in 0:p.M]
        beta = [l < 2 ? 0.0 : 0.02 / l for l in 0:p.L]
        zcm = 0.97
        for th in (0.1, 0.4, 1.0, 2.5)
            @test C_at_theta(am, beta, zcm, th, p) ≈ C_at_theta(am, beta, zcm, -th, p) rtol = 1e-12
        end
        @test abs(tangency_residual(am, beta, zcm, 1e-8, p)) < 1e-6
    end

    @testset "feasibility filters agree with direct evaluation" begin
        p = mkp()
        beta = zeros(p.L + 1)
        am = zeros(p.M + 1)
        # undeformed droplet: dr/dx < 0 across the patch, so w > 0
        @test check_monotone_r(beta, cos(0.3), p.L)
        # droplet far above a flat bath cannot intersect it; one buried in it must
        @test check_nonintersect(am, beta, 3.0, 0.3, p)
        @test !check_nonintersect(am, beta, -1.0, 0.3, p)
    end
end
