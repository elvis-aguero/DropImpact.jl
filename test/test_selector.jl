@testset "contact-edge selector" begin
    # WRITTEN BEFORE THE FIX, AND EXPECTED TO FAIL against theta_c = inf{feasible}. These pin the
    # degeneracy documented in derivations/tangency-selector.tex and verified symbolically in
    # derivations/cas_tangency_residual.py: the infimum rule uses only the SIGN of dC/dtheta at
    # the edge, never its magnitude, so once non-intersection stops binding it returns theta_c ~ 0
    # -- not because the patch physically vanishes but because the rule has nothing left to
    # determine it with.
    #
    # Measured failure being pinned (derivations/debug_feasibility_collapse.jl): at t = 0.009 the
    # non-intersection predicate is false below theta = 0.0492 and inf{feasible} = 0.0492; one
    # step later it holds even at theta = 1e-4, inf{feasible} collapses 200x, and the patch never
    # recovers -- which is what makes the contact time ~2x too long.

    @testset "the patch must not collapse when non-intersection goes inactive" begin
        p = Params(We=0.0231, Bo=0.02, Oh=0.03, b=6.0, h0=3.0, M=60, L=120, N=3, nq=200)
        dt = 1e-3
        hist = SimHistory(initial_level(p), initial_level(p))
        seed = nothing
        for _ in 1:20_000
            th = onset_theta_c(hist, dt, p)
            th !== nothing && (seed = th; break)
            trial = free_flight_step(hist, dt, p)
            hist.prev = hist.curr; hist.curr = trial
        end
        @test seed !== nothing

        theta_prev = seed
        chat_guess = zeros(p.N + 1); chat_guess[1] = 1e-3
        thetas = Float64[]
        for _ in 1:14
            lvl, info = contact_step(hist, dt, theta_prev, chat_guess, p)
            lvl === nothing && break
            push!(thetas, info.theta_c)
            hist.prev = hist.curr; hist.curr = lvl
            theta_prev = info.theta_c
            chat_guess = lvl.X[1:p.N+1]
        end
        @test length(thetas) >= 12
        # A physical patch cannot lose 99% of its extent in one 1e-3 step while the force is
        # still positive. Measured worst ratio under the current rule is ~0.005.
        worst = minimum(thetas[i] / thetas[i-1] for i in 2:length(thetas))
        @test worst > 0.5
        # and it must not simply sit at the floor thereafter
        @test maximum(thetas[max(1, end-3):end]) > 0.01
    end

    @testset "tangency residual is analytic, not a finite difference" begin
        # A quantity that is root-found must not be an h=1e-6 centred difference of a
        # Fourier-Bessel sum. Compared against a Richardson-extrapolated difference at h=1e-3:
        # an analytic T agrees to ~1e-10; an h=1e-6 difference only to ~1e-6.
        p = Params(We=1.0, Bo=0.02, Oh=0.01, b=6.0, h0=3.0, M=30, L=20, N=3, nq=100)
        am = zeros(p.M + 1); am[2] = 3e-3; am[5] = -1.5e-3
        beta = zeros(p.L + 1); beta[3] = 2e-2; beta[5] = -8e-3
        zcm = 0.92
        for th in (0.25, 0.6, 1.1)
            h = 1e-3
            c(x) = C_at_theta(am, beta, zcm, x, p)
            ref = (8 * (c(th + h) - c(th - h)) - (c(th + 2h) - c(th - 2h))) / (12h)
            @test isapprox(tangency_residual(am, beta, zcm, th, p), ref; rtol=1e-8, atol=1e-12)
        end
    end

    @testset "tangency residual vanishes at both poles, for any state" begin
        # eq:tangency-degenerate. Holds because of axisymmetry (J1(0)=0), not the chain rule.
        p = Params(We=1.0, Bo=0.02, Oh=0.01, b=6.0, h0=3.0, M=30, L=20, N=3, nq=100)
        for trial in 1:3
            am = zeros(p.M + 1); am[2] = 1e-3 * trial; am[4] = -2e-3 / trial
            beta = zeros(p.L + 1); beta[3] = 1e-2 * trial; beta[6] = -5e-3
            for th in (1e-7, pi - 1e-7)
                @test abs(tangency_residual(am, beta, 0.9, th, p)) < 1e-5
            end
        end
    end
end
