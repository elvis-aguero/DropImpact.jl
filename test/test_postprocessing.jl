@testset "contact-time metrics" begin
    # Synthetic phase/time series standing in for the structure a real run produces: one long
    # physical contact, then detachment chatter -- a one-step free-flight gap followed by a
    # few-step re-entry. Built by hand so the three candidate definitions have known values
    # and the test does not need to run an impact.
    #
    #  idx    1     2     3     4     5     6     7     8     9
    #  t     0.0   1.0   2.0   3.0   3.001 3.01  3.011 3.02  4.0
    #  ph     F     C     C     C     F     C     C     F     F
    #
    # contact runs: idx 2..4 (opens at t[1]=0.0, closes t[4]=3.0, duration 3.0)
    #               idx 6..7 (opens at t[5]=3.001, closes t[7]=3.011, duration 0.010)
    times = [0.0, 1.0, 2.0, 3.0, 3.001, 3.01, 3.011, 3.02, 4.0]
    phases = [FreeFlight, InContact, InContact, InContact, FreeFlight,
              InContact, InContact, FreeFlight, FreeFlight]

    ivs = contact_intervals(times, phases)
    @test length(ivs) == 2
    @test ivs[1].duration ≈ 3.0
    @test ivs[1].nsteps == 3
    @test ivs[2].duration ≈ 0.010 atol = 1e-12
    @test ivs[2].nsteps == 2

    # The three definitions must differ, and each must be the value it claims to be.
    @test primary_contact_time(times, phases) ≈ 3.0           # longest interval
    # contact_time sums per-step increments: 1.0+1.0+1.0 over the first run, then
    # (3.01-3.001)+(3.011-3.01) = 0.009+0.001 over the second.
    @test contact_time(times, phases) ≈ 3.010
    span = ivs[end].t_end - ivs[1].t_start
    @test span ≈ 3.011
    @test primary_contact_time(times, phases) < contact_time(times, phases) < span

    @testset "primary is the LONGEST interval, not the first" begin
        # This is the distinction that matters when a run chatters at onset: an
        # under-resolved case opens with a handful of tiny intervals before the real
        # contact, and taking the first would report a contact time orders of magnitude
        # too small. Measured directly at M=L=20, nq=16, where the first interval is
        # 0.0081 long against a physical 3.56.
        t = [0.0, 0.001, 0.002, 0.003, 1.0, 2.0, 3.0]
        ph = [FreeFlight, InContact, FreeFlight, FreeFlight, InContact, InContact, InContact]
        iv = contact_intervals(t, ph)
        @test length(iv) == 2
        @test iv[1].duration < 0.01            # the chatter comes first...
        @test iv[2].duration ≈ 3.0 - 0.003     # ...and the physical contact second
        @test primary_contact_time(t, ph) ≈ iv[2].duration
    end

    @testset "no contact at all" begin
        t = [0.0, 1.0, 2.0]
        ph = [FreeFlight, FreeFlight, FreeFlight]
        @test isempty(contact_intervals(t, ph))
        @test primary_contact_time(t, ph) == 0.0
        @test contact_time(t, ph) == 0.0
    end

    @testset "still in contact at t_end" begin
        # The interval must close at the last recorded time rather than being dropped.
        t = [0.0, 1.0, 2.0]
        ph = [FreeFlight, InContact, InContact]
        iv = contact_intervals(t, ph)
        @test length(iv) == 1
        @test iv[1].t_end ≈ 2.0
        @test primary_contact_time(t, ph) ≈ 2.0
    end
end

@testset "run_simulation returns index-aligned phases" begin
    # `phases` must be one label per accepted level. `diag` deliberately is NOT: it carries a
    # row per contact step only, so it is shorter and indexes differently. Conflating the two
    # is what made contact_time unreachable from a real run at one point.
    p = Params(We=1.0958, Bo=0.017, Oh=0.006, M=8, L=8, N=1, b=6.0, h0=3.0, nq=12)
    levels, diag, phases = run_simulation(p; t_end=0.6, dt_init=1e-3)
    @test length(phases) == length(levels)
    @test length(diag) <= length(levels)
    @test count(==(InContact), phases) == length(diag)
    @test phases[1] == FreeFlight              # the initial level is always free flight
end

@testset "threshold_contact_time (AlventosaEtAl2023's definition)" begin
    # Their t_c is the time the centre of mass spends BELOW z=R, with crossings interpolated --
    # a trajectory metric that deliberately includes post-detachment free flight. It is NOT the
    # InContact duration that `contact_time` reports, and conflating the two produced a spurious
    # 40-70% discrepancy against their published data.
    mk(z) = Level(BathModeState(2), DropModeState(2), COMState(z, 0.0), 0.0, 0.0, nothing)

    @testset "exact on a synthetic parabolic dip" begin
        # z(t) = 1 + (t-1)^2 - 0.25 dips below 1 at t = 0.5 and returns at t = 1.5.
        ts = collect(0.0:0.005:3.0)
        zs = @. 1 + (ts - 1)^2 - 0.25
        lv = [mk(z) for z in zs]
        tc = threshold_contact_time(ts, lv)
        @test tc !== nothing
        @test isapprox(tc, 1.0; atol=2e-3)      # interpolation, so tight but not exact
    end

    @testset "starting exactly at the threshold (as initial_level does)" begin
        # initial_level puts z_cm at exactly 1.0, so the descending crossing must be t=0
        # rather than being missed or interpolated from a nonexistent earlier sample.
        ts = collect(0.0:0.01:2.0)
        zs = @. 1 - sin(pi * ts / 1.0) * (ts <= 1.0)
        lv = [mk(z) for z in zs]
        tc = threshold_contact_time(ts, lv)
        @test tc !== nothing
        @test isapprox(tc, 1.0; atol=0.02)
    end

    @testset "returns nothing when the drop never comes back up" begin
        ts = collect(0.0:0.01:1.0)
        lv = [mk(1.0 - t) for t in ts]
        @test threshold_contact_time(ts, lv) === nothing
    end

    @testset "returns nothing when the drop never descends" begin
        ts = collect(0.0:0.01:1.0)
        lv = [mk(1.0 + t) for t in ts]
        @test threshold_contact_time(ts, lv) === nothing
    end

    @testset "threshold is configurable and monotone in it" begin
        ts = collect(0.0:0.002:3.0)
        zs = @. 1 + (ts - 1)^2 - 0.25
        lv = [mk(z) for z in zs]
        # a LOWER threshold is crossed later and exited earlier => shorter interval
        @test threshold_contact_time(ts, lv; z_threshold=0.9) <
              threshold_contact_time(ts, lv; z_threshold=1.0)
    end

    @testset "mismatched inputs are rejected rather than silently misread" begin
        ts = collect(0.0:0.1:1.0)
        lv = [mk(1.0) for _ in 1:3]
        @test_throws ArgumentError threshold_contact_time(ts, lv)
    end
end
