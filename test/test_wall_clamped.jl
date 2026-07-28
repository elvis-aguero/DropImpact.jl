@testset "wall = :clamped (design doc eq:route-b-multiplier)" begin
    @testset "invalid wall symbol still rejected" begin
        @test_throws ArgumentError Params(We=1.0, Bo=0.2, Oh=0.05, M=3, L=3, N=1,
            b=6.0, h0=2.0, nq=16, wall=:bogus)
    end

    @testset ":clamped reuses :free's Neumann basis and weight exactly" begin
        p_free = Params(We=1.0, Bo=0.2, Oh=0.05, M=10, L=3, N=1, b=6.0, h0=2.0, nq=16, wall=:free)
        p_clamped = Params(We=1.0, Bo=0.2, Oh=0.05, M=10, L=3, N=1, b=6.0, h0=2.0, nq=16, wall=:clamped)
        @test p_clamped.k == p_free.k
        @test p_clamped.bath_norm == p_free.bath_norm
        @test p_clamped.j0kb ≈ [besselj0(k * p_clamped.b) for k in p_clamped.k]
        @test all(!=(0.0), p_clamped.j0kb)  # design doc §subsubsec:wall: never vanishes on the Neumann set
    end

    @testset "apply_clamp enforces the pinning constraint for an arbitrary pressure moment" begin
        p = Params(We=1.0, Bo=0.2, Oh=0.05, M=12, L=3, N=1, b=6.0, h0=2.0, nq=16, wall=:clamped)
        kappa, alpha = bath_affine(BathModeState(p.M), BathModeState(p.M), p, 1e-3, -1.0)
        cm = [0.3 * (-1.0)^m / (m + 1) for m in 0:p.M]  # arbitrary nonzero, not a physical solve
        am_free = alpha .+ kappa .* cm
        am = SpectralKM.apply_clamp(am_free, kappa, p)
        @test sum(am .* p.j0kb) ≈ 0.0 atol=1e-12
        @test am != am_free  # the correction is not a no-op for a generic am_free
    end

    @testset "free_flight_step keeps pinning exact with zero pressure forcing" begin
        p = Params(We=1.0, Bo=0.2, Oh=0.05, M=8, L=3, N=1, b=6.0, h0=2.0, nq=16, wall=:clamped)
        a_prev = [0.02 * cos(m) for m in 0:p.M]
        a_curr = [0.03 * sin(m + 0.5) for m in 0:p.M]
        adot = zeros(p.M + 1)
        prev = Level(BathModeState(a_prev, adot), DropModeState(p.L), COMState(1.0, -1.0), 0.0, 1e-3, nothing)
        curr = Level(BathModeState(a_curr, adot), DropModeState(p.L), COMState(1.0, -1.0), 1e-3, 1e-3, nothing)
        hist = SimHistory(prev, curr)
        trial = free_flight_step(hist, 1e-3, p)
        @test sum(trial.bath.a .* p.j0kb) ≈ 0.0 atol=1e-12
    end

    @testset "a full impact under :clamped conserves volume AND holds the pin, together" begin
        p = Params(We=1.0958, Bo=0.017, Oh=0.006, M=20, L=20, N=3, b=6.0, h0=3.0, nq=24, wall=:clamped)
        levels, diag, phases = run_simulation(p; t_end=4.0, dt_init=1e-3, dt_min=1e-9, dt_max=0.05)
        @test length(diag) > 0  # contact actually happened, this isn't a vacuous pass

        pin_resid = maximum(sum(lv.bath.a .* p.j0kb) for lv in levels)
        @test abs(pin_resid) < 1e-9

        vol = lv -> 2 * sum(lv.bath.a[m+1] * (p.b / p.k[m+1]) * besselj1(p.k[m+1] * p.b)
                             for m in 1:p.M)
        max_vol = maximum(abs(vol(lv)) for lv in levels)
        @test max_vol < 1e-9  # structural: kappa_0 = 0 regardless of the clamp correction
    end
end
