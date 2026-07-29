# PHYSICALLY GROUNDED TESTS.
#
# WHY THIS FILE EXISTS. Before it, the suite had ~700 assertions and not one of them checked
# that the model produces correct PHYSICS. `test_reductions.jl`, despite its name, tests only
# internal algebraic identities -- that the code implements the equations as written. Nothing
# compared any observable against a known physical value, a closed-form limit, a conservation
# law, or the literature. That is how a 2x overprediction of the contact time survived the
# entire suite unnoticed.
#
# Every test below asserts something that must hold on physical grounds, independent of how the
# equations are discretised, and each names the physical law it comes from. They are cheap: the
# free-flight and free-oscillation checks need no impact at all, and the impact-based ones use
# small truncations and short horizons.

@testset "physics" begin

    # =========================================================================================
    # 1. FREE FLIGHT IS BALLISTIC.  dv/dtau = -Bo with no contact force, so the centre of mass
    #    must follow v(t) = v0 - Bo*t exactly. Catches sign errors and spurious forces acting
    #    while the drop is airborne -- a real class of bug here: provenance.tex records the
    #    wall constraint once being left unenforced during free flight.
    # =========================================================================================
    @testset "free flight obeys dv/dt = -Bo (ballistic, no spurious forces)" begin
        for Bo in (0.0, 0.05, 0.5)
            p = Params(We=0.25, Bo=Bo, Oh=0.01, M=8, L=8, N=3, b=6.0, h0=3.0, nq=60)
            hist = SimHistory(initial_level(p), initial_level(p))
            dt = 1e-3
            v0, z0, t = hist.curr.com.v, hist.curr.com.z, 0.0
            for _ in 1:200
                lvl = free_flight_step(hist, dt, p)
                hist.prev = hist.curr; hist.curr = lvl; t += dt
            end
            @test isapprox(hist.curr.com.v, v0 - Bo * t; rtol=1e-6, atol=1e-9)
            @test isapprox(hist.curr.com.z, z0 + v0 * t - Bo * t^2 / 2; rtol=1e-5, atol=1e-8)
        end
    end

    # =========================================================================================
    # 2. AN INVISCID DROP MODE MUST NOT DECAY.  At Oh = 0 there is no dissipation mechanism, so
    #    a perturbed surface mode must ring forever at constant amplitude. Any decay is
    #    numerical dissipation masquerading as physics.
    # =========================================================================================
    @testset "inviscid drop mode does not decay (Oh = 0)" begin
        p = Params(We=0.25, Bo=0.0, Oh=0.0, b=6.0, h0=3.0, M=6, L=6, N=3, nq=60,
                   viscous=:lamb)          # Reid roots are undefined at Oh = 0
        l, dt = 2, 5e-4
        bprev = zeros(p.L + 1); bcurr = zeros(p.L + 1)
        dprev = zeros(p.L + 1); dcurr = zeros(p.L + 1)
        bcurr[l+1] = 1e-3
        amp0 = abs(bcurr[l+1])
        peak = 0.0
        for _ in 1:20_000                     # ~5 oscillation periods
            _, gam = drop_affine(DropModeState(copy(bcurr), copy(dcurr)),
                                 DropModeState(copy(bprev), copy(dprev)), p, dt, dt)
            bn = gam[l+1]
            dn = bdf_derivative(bn, bcurr[l+1], bprev[l+1], dt, dt)
            bprev, dprev = copy(bcurr), copy(dcurr)
            bcurr = zeros(p.L + 1); dcurr = zeros(p.L + 1)
            bcurr[l+1], dcurr[l+1] = bn, dn
            peak = max(peak, abs(bn))
        end
        # BDF2 is dissipative at O(dt^2), so allow a small budget but no more.
        @test abs(peak - amp0) / amp0 < 0.05
        @test abs(bcurr[l+1]) < 1.05 * amp0        # and it certainly must not GROW
    end

    # =========================================================================================
    # 3. THE DROP RINGS AT THE RAYLEIGH-LAMB FREQUENCY.  omega_l^2 = l(l-1)(l+2) sigma/(rho R^3),
    #    i.e. omega_l = sqrt(l(l-1)(l+2)) in these units. This is the single most basic physical
    #    fact about a free drop and was previously untested as such.
    # =========================================================================================
    @testset "drop oscillates at the Rayleigh-Lamb frequency" begin
        p = Params(We=0.25, Bo=0.0, Oh=0.0, b=6.0, h0=3.0, M=6, L=10, N=3, nq=60,
                   viscous=:lamb)
        dt = 2e-4
        for l in (2, 3, 4)
            bprev = zeros(p.L + 1); bcurr = zeros(p.L + 1)
            dprev = zeros(p.L + 1); dcurr = zeros(p.L + 1)
            bcurr[l+1] = 1e-3
            zeros_t = Float64[]
            tprev, vprev = 0.0, bcurr[l+1]
            for k in 1:40_000
                _, gam = drop_affine(DropModeState(copy(bcurr), copy(dcurr)),
                                     DropModeState(copy(bprev), copy(dprev)), p, dt, dt)
                bn = gam[l+1]
                dn = bdf_derivative(bn, bcurr[l+1], bprev[l+1], dt, dt)
                bprev, dprev = copy(bcurr), copy(dcurr)
                bcurr = zeros(p.L + 1); dcurr = zeros(p.L + 1)
                bcurr[l+1], dcurr[l+1] = bn, dn
                t = k * dt
                if sign(bn) != sign(vprev) && vprev != 0
                    push!(zeros_t, tprev + (t - tprev) * abs(vprev) / (abs(vprev) + abs(bn)))
                end
                tprev, vprev = t, bn
                length(zeros_t) >= 7 && break
            end
            @test length(zeros_t) >= 3
            halfper = sum(diff(zeros_t)) / (length(zeros_t) - 1)
            omega = pi / halfper
            @test isapprox(omega, sqrt(float(l) * (l - 1) * (l + 2)); rtol=0.01)
        end
    end

    # =========================================================================================
    # 4. THE BATH RINGS AT THE CAPILLARY-GRAVITY DISPERSION FREQUENCY.  For mode m,
    #    omega^2 = (k^3 + Bo*k) tanh(k h0) in these units. Tests the bath operator against the
    #    dispersion relation rather than against itself.
    # =========================================================================================
    @testset "bath modes obey the capillary-gravity dispersion relation" begin
        Bo, h0 = 0.05, 3.0
        p = Params(We=0.25, Bo=Bo, Oh=0.0, b=6.0, h0=h0, M=20, L=6, N=3, nq=60, viscous=:lamb)
        dt = 2e-4
        for m in (3, 6, 10)
            k = p.k[m+1]
            aprev = zeros(p.M + 1); acurr = zeros(p.M + 1)
            adprev = zeros(p.M + 1); adcurr = zeros(p.M + 1)
            acurr[m+1] = 1e-6
            zeros_t = Float64[]
            tprev, vprev = 0.0, acurr[m+1]
            nmax = ceil(Int, 6 * 2pi / sqrt((k^3 + Bo * k) * tanh(k * h0)) / dt)
            for kk in 1:nmax
                _, gam = bath_affine(BathModeState(copy(acurr), copy(adcurr)),
                                     BathModeState(copy(aprev), copy(adprev)), p, dt, dt)
                an = gam[m+1]
                adn = bdf_derivative(an, acurr[m+1], aprev[m+1], dt, dt)
                aprev, adprev = copy(acurr), copy(adcurr)
                acurr = zeros(p.M + 1); adcurr = zeros(p.M + 1)
                acurr[m+1], adcurr[m+1] = an, adn
                t = kk * dt
                if sign(an) != sign(vprev) && vprev != 0
                    push!(zeros_t, tprev + (t - tprev) * abs(vprev) / (abs(vprev) + abs(an)))
                end
                tprev, vprev = t, an
                length(zeros_t) >= 7 && break
            end
            @test length(zeros_t) >= 3
            halfper = sum(diff(zeros_t)) / (length(zeros_t) - 1)
            @test isapprox(pi / halfper, sqrt((k^3 + Bo * k) * tanh(k * h0)); rtol=0.02)
        end
    end

    # =========================================================================================
    # 5. VISCOSITY DAMPS, IT DOES NOT DRIVE.  Raising Oh at fixed We must not increase the
    #    energy the drop leaves with. A CoR that rises with viscosity would mean the viscous
    #    terms are injecting energy.
    # =========================================================================================
    @testset "CoR is non-increasing in Oh (viscosity cannot add energy)" begin
        cors = Float64[]
        for Oh in (0.005, 0.02, 0.08)
            p = Params(We=1.0, Bo=0.02, Oh=Oh, b=6.0, h0=3.0, M=30, L=40, N=3, nq=100)
            levels, diag, phases = run_simulation(p; t_end=8.0)
            c = coefficient_of_restitution([l.t for l in levels], levels, phases)
            c === nothing && continue
            push!(cors, c)
        end
        @test length(cors) >= 2
        for i in 2:length(cors)
            @test cors[i] <= cors[i-1] + 0.02        # tolerance for step-size noise
        end
    end

    # =========================================================================================
    # 6. NO ENERGY CREATION.  The drop cannot rebound faster than it arrived: 0 < CoR <= 1.
    # =========================================================================================
    @testset "0 < CoR <= 1 (no energy created on rebound)" begin
        for (We, Bo, Oh) in ((0.5, 0.02, 0.01), (1.0958, 0.017, 0.006), (3.0, 0.05, 0.03))
            p = Params(We=We, Bo=Bo, Oh=Oh, b=6.0, h0=3.0, M=30, L=40, N=3, nq=100)
            levels, diag, phases = run_simulation(p; t_end=8.0)
            c = coefficient_of_restitution([l.t for l in levels], levels, phases)
            c === nothing && continue
            @test 0 < c <= 1
        end
    end

    # =========================================================================================
    # 7. THE CONTACT FORCE IS A DECELERATION, AND ITS IMPULSE MATCHES THE MOMENTUM CHANGE.
    #    Newton: dv/dtau = -Bo - f, so integrating over contact must reproduce the observed
    #    velocity change. Independent of any metric definition.
    # =========================================================================================
    @testset "impulse of the contact force equals the momentum change" begin
        p = Params(We=1.0958, Bo=0.017, Oh=0.006, b=6.0, h0=3.0, M=30, L=40, N=3, nq=100)
        levels, diag, phases = run_simulation(p; t_end=8.0)
        rows = [d for d in diag if haskey(d, :theta_c)]
        @test !isempty(rows)
        # trapezoid over the reported per-step force, against dv from the trajectory
        ic = findall(==(InContact), phases)
        @test !isempty(ic)
        i0, i1 = first(ic) - 1, last(ic) + 1
        if i0 >= 1 && i1 <= length(levels)
            dv = levels[i1].com.v - levels[i0].com.v
            ts = [d.t for d in rows]; fs = [d.f for d in rows]
            imp = 0.0
            for i in 2:length(ts)
                imp += 0.5 * (fs[i] + fs[i-1]) * (ts[i] - ts[i-1])
            end
            grav = -p.Bo * (levels[i1].t - levels[i0].t)
            @test isapprox(dv, grav - imp; rtol=0.15, atol=0.05)
            @test imp > 0                    # the force decelerates: positive upward impulse
        end
    end

    # =========================================================================================
    # 8. HARDER IMPACTS PENETRATE DEEPER.  AlventosaEtAl2023 find the maximum penetration depth
    #    increases monotonically with We, for water and for oil, in experiment, DNS and model.
    # =========================================================================================
    @testset "penetration depth increases with We (Alventosa et al., all three methods)" begin
        pens = Float64[]
        for We in (0.3, 1.0, 3.0)
            p = Params(We=We, Bo=0.02, Oh=0.01, b=6.0, h0=3.0, M=30, L=40, N=3, nq=100)
            levels, diag, phases = run_simulation(p; t_end=8.0)
            push!(pens, max_penetration_depth(levels, p.L))
        end
        @test issorted(pens)
        @test pens[end] > 1.5 * pens[1]
    end

    # =========================================================================================
    # 9. CONTACT TIME IS OF ORDER THE INERTIO-CAPILLARY TIME.  Every experimental and numerical
    #    source in this literature puts tc/t_sigma in roughly 2-7 (AlventosaEtAl2023 quote
    #    "4-6 t_sigma" for low-Oh literature values). A model outside that band is wrong by a
    #    factor, which is exactly the failure this file was written after.
    # =========================================================================================
    @testset "contact time is O(t_sigma), within the literature band" begin
        p = Params(We=1.0958, Bo=0.017, Oh=0.006, b=6.0, h0=3.0, M=30, L=40, N=3, nq=100)
        levels, diag, phases = run_simulation(p; t_end=10.0)
        ts = [l.t for l in levels]
        tci = contact_time(ts, phases)
        @test 1.0 < tci < 10.0
        tct = threshold_contact_time(ts, levels)
        if tct !== nothing
            @test 1.0 < tct < 12.0
            @test tct >= tci - 1e-9      # the threshold metric includes post-detachment flight
        end
    end

    # =========================================================================================
    # 10. THE ANSWER MUST NOT DEPEND ON THE DOMAIN SIZE.  b is a numerical parameter, not
    #     physics: once the bath is large enough that reflected capillary waves do not return
    #     during contact, observables must stop moving. AlventosaEtAl2023 use b = 25R for
    #     exactly this reason. THIS IS THE TEST THAT WOULD HAVE CAUGHT THE CURRENT BUG -- at
    #     b = 6 (this package's long-standing default) the wall is close enough that a round
    #     trip at k ~ 4 takes ~4 t_sigma, inside the contact.
    #
    #     Deliberately asserted between two LARGE radii, so it tests the asymptotic regime
    #     rather than baking in whatever b = 6 happens to give. Kept cheap by scaling M with b
    #     to hold k_max fixed and using a short horizon.
    # =========================================================================================
    @testset "observables are insensitive to bath radius once b is large" begin
        res = NamedTuple[]
        for (b, M) in ((20.0, 100), (30.0, 150))
            p = Params(We=1.0, Bo=0.02, Oh=0.01, b=b, h0=3.0, M=M, L=40, N=3, nq=100)
            levels, diag, phases = run_simulation(p; t_end=9.0)
            ts = [l.t for l in levels]
            c = coefficient_of_restitution(ts, levels, phases)
            push!(res, (b=b, tc=contact_time(ts, phases), cor=c))
        end
        @test length(res) == 2
        @test isapprox(res[1].tc, res[2].tc; rtol=0.10)
        if res[1].cor !== nothing && res[2].cor !== nothing
            @test isapprox(res[1].cor, res[2].cor; rtol=0.10)
        end
    end
end
