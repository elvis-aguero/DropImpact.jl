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

    @testset "warm_start_extrapolated falls back sensibly with no prior contact solution" begin
        X = warm_start_extrapolated(0.01, nothing, 0.0, nothing, 0.0, 2)
        @test length(X) == 4  # N+2 = 2+2
        @test X[end] > 0
    end

    # DISABLED 2026-07-27: "kinematic_contact_step escapes the theta_c=0 degenerate root"
    # tested a now-superseded interim mechanism (predates the design doc's
    # §subsubsec:contact-angle geometric-crossing theory, eq:theta-c-crossing — that
    # theory is not yet implemented in Julia at all, see STATUS.md). Its exact numeric
    # behavior stopped being a correctness target once a genuine, confirmed sign bug
    # (z_d entering the contact condition with the wrong sign — see accel_closure.jl's
    # outer_bracket_of_x) was fixed underneath it. Re-enable or replace with a test of
    # eq:theta-c-crossing itself once that's implemented; this Julia `Test` version
    # doesn't support `@testset ... skip=true`, so the body is removed rather than kept
    # dead and silently un-run.

    @testset "contact_step returns a usable Level for a small, well-posed case" begin
        p = Params(We=1.0, Bo=0.2, Oh=0.05, M=6, L=6, N=1, b=6.0, h0=2.0, nq=16)
        lvl0 = initial_level(p)
        hist = SimHistory(lvl0, lvl0)
        X_guess = warm_start(nothing, p.N)
        lvl, result, admis = contact_step(hist, 1e-3, X_guess, p)
        @test lvl !== nothing
        @test result.status in (Converged, Stalled)
        @test all(isfinite, lvl.bath.a)
        @test all(isfinite, lvl.drop.beta)
        @test isfinite(lvl.com.z)
    end
end
