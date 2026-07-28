@testset "timestepper.jl" begin
    @testset "initial_level starts at the moment of first touch" begin
        p = Params(We=2.0, Bo=0.2, Oh=0.05, M=4, L=4, N=1, b=6.0, h0=2.0, nq=16)
        lvl0 = initial_level(p)
        @test lvl0.com.z == 1.0
        @test lvl0.com.v ≈ -sqrt(p.We)
        @test abs(gap_at_pole(lvl0.bath, lvl0.drop, lvl0.com, p.L)) < 1e-12
    end

    @testset "free_flight_step matches the exact ballistic solution z=z0+v0 t - Bo t^2/2" begin
        # BDF2 is exact for a quadratic solution of a linear constant-coefficient ODE
        # GIVEN an exact history. Here the very first step has no `dt_prev` yet (Level's
        # dt=0 sentinel) and uses the BDF1 limit instead (bdf2_coeffs), which is only
        # first-order accurate; that O(Bo*dt^2) offset then propagates additively (not
        # growing, since the ODE is linear) through the later, truly-exact BDF2 steps —
        # so all steps are checked to the SAME loose, first-step-dominated tolerance,
        # not to full precision.
        p = Params(We=1.0, Bo=0.3, Oh=0.05, M=3, L=3, N=1, b=6.0, h0=2.0, nq=16)
        lvl0 = initial_level(p)
        hist = SimHistory(lvl0, lvl0)
        z0, v0 = lvl0.com.z, lvl0.com.v
        t = 0.0
        dt = 0.01
        tol = 5 * 0.5 * p.Bo * dt^2
        for _ in 1:5
            trial = free_flight_step(hist, dt, p)
            hist.prev = hist.curr
            hist.curr = trial
            t += dt
            z_exact = z0 + v0 * t - 0.5 * p.Bo * t^2
            @test trial.com.z ≈ z_exact atol = tol
        end
    end

    @testset "onset_theta_c fires exactly when the pole penetrates" begin
        p = Params(We=1.0, Bo=0.2, Oh=0.05, M=6, L=6, N=1, b=6.0, h0=2.0, nq=16)
        lvl0 = initial_level(p)
        hist = SimHistory(lvl0, lvl0)
        # z_cm(0) = 1 puts the pole exactly at the bath surface, so one step of free
        # fall already penetrates and onset must be detected immediately.
        th = onset_theta_c(hist, 1e-3, p)
        @test th !== nothing
        @test 0 < th < π / 2
        # Lifted well clear of the bath, there is no crossing to find.
        high = Level(lvl0.bath, lvl0.drop, COMState(3.0, -0.1), 0.0, 0.0, nothing)
        @test onset_theta_c(SimHistory(high, high), 1e-3, p) === nothing
    end

    @testset "the feasible set of contact angles is a band, not a half-line" begin
        # The property that forced the outer search to be a bracketed edge search rather
        # than a bisection on a monotone predicate (design doc §subsubsec:contact-angle):
        # candidates are infeasible at small theta_c (interpenetration) and infeasible
        # again at large theta_c (eq:check-monotone-r fails), with a feasible band between.
        # Guarding the UPPER edge is enough to catch a regression to monotonicity.
        p = Params(We=1.0, Bo=0.2, Oh=0.05, M=8, L=8, N=1, b=6.0, h0=2.0, nq=20)
        lvl0 = initial_level(p)
        hist = SimHistory(lvl0, lvl0)
        okbig, _ = feasible_at(hist, 1e-3, 1.45, [1e-3, 0.0], p)
        @test !okbig      # a patch reaching the droplet's widest point is inadmissible
    end

    @testset "select_theta_c converges as dt -> 0 to the previous angle" begin
        # A consistency check on the whole nested step: with the state frozen, shrinking
        # the step must return the contact angle it started from, since the pressure's
        # authority over the geometry scales as dt^2.
        p = Params(We=1.0, Bo=0.2, Oh=0.05, M=10, L=10, N=1, b=6.0, h0=2.0, nq=20)
        lvl0 = initial_level(p)
        hist = SimHistory(lvl0, lvl0)
        th0 = onset_theta_c(hist, 1e-3, p)
        lvl, info = contact_step(hist, 1e-3, th0, [1e-3, 0.0], p)
        @test lvl !== nothing
        hist.prev = hist.curr; hist.curr = lvl
        prev = info.theta_c
        got = Float64[]
        for dt in (1e-4, 1e-5, 1e-6)
            r = select_theta_c(hist, dt, prev, lvl.X[1:p.N+1], p; dt_ref=1e-3)
            @test r !== nothing
            push!(got, r.theta_c)
        end
        @test abs(got[end] - prev) < abs(got[1] - prev) + 1e-12   # monotone approach
        @test abs(got[end] - prev) < 5e-3
    end

    @testset "contact_step returns a usable Level, with a decelerating force" begin
        p = Params(We=1.0, Bo=0.2, Oh=0.05, M=10, L=10, N=1, b=6.0, h0=2.0, nq=20)
        lvl0 = initial_level(p)
        hist = SimHistory(lvl0, lvl0)
        th0 = onset_theta_c(hist, 1e-3, p)
        lvl, info = contact_step(hist, 1e-3, th0, [1e-3, 0.0], p)
        @test lvl !== nothing
        @test all(isfinite, lvl.bath.a)
        @test all(isfinite, lvl.drop.beta)
        @test isfinite(lvl.com.z)
        @test length(lvl.X) == p.N + 2            # (chat..., theta_c)
        @test 0 < info.theta_c < π / 2
        # The selected angle sits on the feasibility boundary, where the net contact
        # force is positive: the pressure must DECELERATE the droplet. A sign error in
        # the COM force integral (a bug this project has actually had) flips this.
        @test info.f > 0
    end
end
