# Reid (1960) arbitrary-Oh drop damping: does it solve the right equation, does it reduce to
# Lamb as Oh -> 0, and does selecting it leave the published model untouched?

@testset "reid.jl (arbitrary-Oh drop viscous coefficients)" begin

    @testset "roots satisfy Reid's characteristic equation" begin
        # The only real correctness check available without an independent implementation:
        # the returned eigenvalue must annihilate the characteristic equation.
        for Oh in (0.001, 0.006, 0.05, 0.1, 0.3)
            for l in (2, 4, 8, 16)
                lam, om, resid = reid_root(l, Oh)
                @test isfinite(lam) && isfinite(om)
                @test resid < 1e-8
                @test lam > 0            # sigma is a DECAY rate
            end
        end
    end

    @testset "reduces to Lamb as Oh -> 0" begin
        # Lamb is the small-viscosity asymptotics of exactly this equation, so the relative
        # discrepancy must vanish with Oh. Measured: 1.6% at Oh=1e-3, 0.17% at Oh=1e-4.
        errs = Float64[]
        for Oh in (1e-2, 1e-3, 1e-4)
            lam, _, _ = reid_root(2, Oh)
            lam_lamb = Oh * (2 - 1) * (2 * 2 + 1)
            push!(errs, abs(lam_lamb - lam) / lam)
        end
        @test issorted(errs; rev=true)          # error shrinks with Oh
        @test errs[end] < 1e-2                  # and is small in the asymptotic regime
    end

    @testset "Lamb's eigenvalue error vs Reid, and it FLIPS SIGN" begin
        # The fair comparison is between EIGENVALUES. Lamb's `c = Oh(l-1)(2l+1)` is an ODE
        # coefficient; it equals the eigenvalue decay rate only while c < omega_0. Past that
        # the oscillator's least-damped root is c - sqrt(c^2 - omega_0^2), which COLLAPSES.
        # Consequence, measured: Lamb over-damps in the underdamped regime and UNDER-damps
        # once its own coefficient pushes it into (spurious) overdamping.
        underdamped(l, Oh) = Oh * (l - 1) * (2l + 1) < sqrt(float(l) * (l - 1) * (l + 2))

        @testset "underdamped: Lamb over-damps, worse with Oh and with l" begin
            for Oh in (0.006, 0.1, 0.3), l in (2, 4, 8, 16)
                underdamped(l, Oh) || continue
                le, _ = lamb_eigenvalue(l, Oh)
                lr, _, _ = reid_root_tracked(l, Oh)
                @test le > lr
            end
            rel(Oh, l) = ((lamb_eigenvalue(l, Oh)[1] - reid_root_tracked(l, Oh)[1])
                          / reid_root_tracked(l, Oh)[1])
            @test rel(0.3, 2) > rel(0.1, 2) > rel(0.006, 2)      # 36.7 / 22.9 / 4.1 %
            @test rel(0.006, 16) > rel(0.006, 8) > rel(0.006, 2) # 11.9 / 9.2 / 4.1 %
            @test rel(0.006, 16) > 0.05        # not negligible at OUR production Oh
        end

        @testset "spuriously overdamped: Lamb UNDER-damps" begin
            # l = 8 and 16 at Oh = 0.3 are past Lamb's threshold; measured -39% and -69%.
            for l in (8, 16)
                @test !underdamped(l, 0.3)
                le, _ = lamb_eigenvalue(l, 0.3)
                lr, _, _ = reid_root_tracked(l, 0.3)
                @test le < lr                  # sign of the error has reversed
            end
        end

        @testset "Lamb's overdamping onset is itself spurious" begin
            # At l = 16, Oh = 0.3 Lamb predicts a non-oscillatory mode (omega = 0) while the
            # tracked Reid eigenvalue still has a nonzero imaginary part (omega ~ 10.5).
            @test lamb_eigenvalue(16, 0.3)[2] == 0.0
            _, om_reid, _ = reid_root_tracked(16, 0.3)
            @test om_reid > 1.0
        end
    end

    @testset "continuation tracks the branch a direct solve loses" begin
        # Direct, Lamb-seeded Newton at l=16, Oh=0.3 lands on a purely real root with
        # lambda = 288.6; continuation in Oh follows the true branch to lambda = 49.2,
        # omega = 10.5. This is the case that motivated reid_root_tracked.
        d_lam, d_om, _ = reid_root(16, 0.3)
        t_lam, t_om, t_res = reid_root_tracked(16, 0.3)
        @test t_res < 1e-8
        @test t_lam < d_lam / 2            # the tracked branch is far less damped
        @test t_om > 1.0 && d_om < 1e-6    # and still oscillatory, where direct was not
        # Continuation must NOT perturb the easy cases: identical to roundoff.
        for (l, Oh) in ((2, 0.006), (16, 0.006), (2, 0.1), (8, 0.1))
            a, _, _ = reid_root(l, Oh)
            b, _, _ = reid_root_tracked(l, Oh)
            @test isapprox(a, b; rtol=1e-10)
        end
    end

    @testset "frequency shift is second-order small at small Oh" begin
        # Damping is where Lamb errs; the frequency is barely affected until Oh is O(0.1).
        # Measured at Oh=0.006: <= 0.5% up to l=16. At Oh=0.5, l=2: 37%.
        for l in (2, 8, 16)
            _, om, _ = reid_root(l, 0.006)
            om_lamb = sqrt(float(l) * (l - 1) * (l + 2))
            @test abs(om_lamb - om) / om < 0.01
        end
        _, om_hi, _ = reid_root(2, 0.5)
        @test abs(sqrt(2.0 * 1 * 4) - om_hi) / om_hi > 0.1
    end

    @testset "drop_viscous_coeffs storage convention and dispatch" begin
        L = 12
        lam_l, om2_l = drop_viscous_coeffs(L, 0.006, :lamb)
        lam_r, om2_r = drop_viscous_coeffs(L, 0.006, :reid)
        @test length(lam_l) == L + 1 && length(om2_l) == L + 1
        # l = 0,1 are never evolved and must stay exactly zero.
        @test lam_l[1] == 0.0 && lam_l[2] == 0.0
        @test lam_r[1] == 0.0 && lam_r[2] == 0.0
        # :lamb must reproduce the closed form exactly.
        for l in 2:L
            @test lam_l[l+1] == 0.006 * (l - 1) * (2l + 1)
            @test om2_l[l+1] == float(l) * (l - 1) * (l + 2)
            @test lam_r[l+1] < lam_l[l+1]
        end
        @test_throws ArgumentError drop_viscous_coeffs(L, 0.006, :bogus)
    end

    @testset "omega2 reproduces Reid's eigenvalue, not just its imaginary part" begin
        # The oscillator's oscillation frequency is sqrt(omega2 - lambda^2), so omega2 must
        # be lambda^2 + omega^2 for the ODE's roots to equal Reid's eigenvalue.
        L = 8
        lam_r, om2_r = drop_viscous_coeffs(L, 0.1, :reid)
        for l in 2:L
            lam, om, _ = reid_root(l, 0.1)
            lam >= 0.1 * (l - 1) * (2l + 1) && continue      # fell back to Lamb
            @test isapprox(om2_r[l+1], lam^2 + om^2; rtol=1e-10)
            # and the ODE's own oscillation frequency then recovers Reid's omega
            @test isapprox(sqrt(om2_r[l+1] - lam_r[l+1]^2), om; rtol=1e-8)
        end
    end

    @testset "faithful two-pole mapping (Vieta) per reid-viscous-closure.tex" begin
        # Derivation: derivations/reid-viscous-closure.tex, verified symbolically in
        # derivations/cas_reid_two_pole.py. lambda = -(s1+s2)/2, omega2 = s1*s2.

        @testset "underdamped: Vieta on the conjugate pair" begin
            for (l, Oh) in ((2, 0.006), (2, 0.3), (8, 0.1), (16, 0.1))
                lam, om2, info = reid_pole_pair(l, Oh)
                @test info === :underdamped
                lr, omr, _ = reid_root_tracked(l, Oh)
                @test isapprox(lam, lr; rtol=1e-10)
                @test isapprox(om2, lr^2 + omr^2; rtol=1e-10)
                # the ODE's own oscillation frequency must return Reid's omega
                @test isapprox(sqrt(om2 - lam^2), omr; rtol=1e-8)
            end
        end

        @testset "overdamped: Vieta on the two SLOWEST real roots, both reproduced" begin
            for (l, Oh) in ((2, 0.8), (2, 1.0), (2, 3.0), (16, 0.5))
                lam, om2, info = reid_pole_pair(l, Oh)
                @test info === :overdamped
                rs = reid_real_roots(l, Oh)
                @test length(rs) >= 2
                g1, g2 = rs[1], rs[2]
                @test isapprox(lam, (g1 + g2) / 2; rtol=1e-8)
                @test isapprox(om2, g1 * g2; rtol=1e-8)
                # BOTH rates come back out of the ODE, which is the whole point:
                disc = sqrt(lam^2 - om2)
                @test isapprox(lam - disc, g1; rtol=1e-6)
                @test isapprox(lam + disc, g2; rtol=1e-6)
            end
        end

        @testset "continuation does NOT return the slowest root once merged" begin
            # reid_real_roots returns ONLY the dominant pair (bounded by the first
            # singularity), so exactly 2 once merged. The failure this documents: at
            # l=2, Oh=3 the dominant pair is (0.357, 20.97) and continuation returns the
            # SECOND, so it must not be used alone.
            rs = reid_real_roots(2, 3.0)
            @test length(rs) == 2
            @test isapprox(rs[1], 0.3568; atol=1e-3)
            @test isapprox(rs[2], 20.9668; atol=1e-3)
            trk, om, _ = reid_root_tracked(2, 3.0)
            @test om <= 1e-8 * max(trk, 1.0)           # merged
            @test isapprox(trk, rs[2]; rtol=1e-6)       # the second, not the first
            @test rs[1] < trk
        end

        @testset "the slowest rate falls with Oh (viscous drops relax more slowly)" begin
            slow = [reid_real_roots(2, Oh)[1] for Oh in (0.8, 1.0, 3.0)]
            @test issorted(slow; rev=true)             # 2.04, 1.28, 0.357
        end

        @testset "omega2 stays near the inviscid stiffness, and is CONTINUOUS at the merge" begin
            # Emergent, not imposed: viscosity damps but does not change what restores.
            om0sq = 2.0 * 1 * 4
            ratios = Float64[]
            for Oh in (0.006, 0.3, 0.5, 0.8, 1.0, 3.0)
                _, om2, _ = reid_pole_pair(2, Oh)
                push!(ratios, om2 / om0sq)
                @test 0.85 < om2 / om0sq < 1.02
            end
            # No jump across the underdamped -> overdamped transition (0.5 -> 0.8).
            @test abs(ratios[4] - ratios[3]) < 0.02
            @test issorted(ratios; rev=true)           # monotone, smooth
        end

        @testset "the superseded critically-damped choice was badly wrong" begin
            # It set omega2 = lambda^2 on the tracked (second) root. At l=2, Oh=3 that is
            # 113.7 against a true stiffness of 7.48 -- an order of magnitude, not a nuance.
            lam, om2, _ = reid_pole_pair(2, 3.0)
            @test lam^2 / om2 > 10
            @test isapprox(om2, 7.48; atol=0.05)
        end
    end

    @testset "STRICTNESS: numerics that must not silently degrade" begin
        # Motivation: Bessel evaluation misbehaves at the arguments this problem needs, and a
        # silent misbehaviour here yields a plausible-looking but wrong damping rate. Every
        # guard below exists because something actually failed during development.

        @testset "the ratio recurrence matches scaled Bessel where that works" begin
            # Two independent evaluations of Q_l = j_{l+1}/j_l must agree tightly.
            for (l, q) in ((2, 5.0 + 0.0im), (2, 37.6 + 37.6im), (8, 12.0 + 0.0im),
                           (16, 21.5 + 0.0im), (120, 131.0 + 0.0im), (60, 3.0 + 80.0im))
                a = sph_bessel_ratio(l, q)
                b = besseljx(l + 3 / 2, q) / besseljx(l + 1 / 2, q)
                @test isapprox(a, b; rtol=1e-11)
            end
        end

        @testset "the recurrence survives where plain Bessel does NOT" begin
            # l=120, q=0.27 underflows besselj to zero and silently produced a spurious
            # q* = 0.271 during development. The recurrence returns the small-argument limit.
            v = sph_bessel_ratio(120, 0.27 + 0.0im)
            @test isfinite(abs(v)) && abs(v) > 0
            @test isapprox(real(v), 0.27 / (2 * 120 + 3); rtol=1e-2)   # Q_n ~ q/(2n+3)
        end

        @testset "q*(l) is a true singularity and lies below the first Bessel zero" begin
            for l in (2, 4, 8, 16, 32, 60, 120)
                qs = reid_first_singularity(l)
                jz1 = first_bessel_zero_half(l)
                @test isfinite(qs) && qs > 0
                @test qs < jz1                       # must be the FIRST singularity
                g = qs - 2 * real(sph_bessel_ratio(l, complex(qs, 0.0)))
                @test abs(g) < 1e-8                  # genuinely a zero of q - 2Q
            end
            # Oh-independence: memoised on l alone.
            @test reid_first_singularity(16) === reid_first_singularity(16)
        end

        @testset "STRUCTURAL INVARIANT: 0 or exactly 2 roots in the bracket, never 1 or 3" begin
            # This is the guard against garbage: if the bracket ever admits an odd number of
            # roots, the dominant pair has been mis-identified and every downstream damping
            # rate is wrong. Verified over a grid, not a single point.
            for l in (2, 4, 8, 16, 32), Oh in (0.05, 0.2, 0.5, 0.8, 1.5, 3.0)
                rs = reid_real_roots(l, Oh)
                @test length(rs) in (0, 2)
                # and consistency with the oscillatory/merged classification
                _, om, _ = reid_root_tracked(l, Oh)
                merged = om <= 1e-8 * max(reid_root_tracked(l, Oh)[1], 1.0)
                @test (length(rs) == 2) == merged
            end
        end

        @testset "every returned root actually annihilates the characteristic equation" begin
            for l in (2, 8, 16), Oh in (0.8, 1.5, 3.0)
                for r in reid_real_roots(l, Oh)
                    @test abs(reid_char_residual(complex(r, 0.0), l, Oh)) < 1e-6
                end
                lam, om, resid = reid_root_tracked(l, Oh)
                @test resid < 1e-8
            end
        end

        @testset "no NaN or non-physical coefficient ever reaches Params" begin
            # A NaN damping would propagate into the affine map and poison a whole run.
            for Oh in (0.006, 0.1, 0.5, 1.0, 2.0), L in (8, 24)
                lam, om2 = drop_viscous_coeffs(L, Oh, :reid)
                @test all(isfinite, lam) && all(isfinite, om2)
                for l in 2:L
                    @test lam[l+1] > 0                      # must damp, not grow
                    @test om2[l+1] > 0                      # must restore
                end
            end
        end
    end

    @testset "selecting :reid does not disturb the published model" begin
        pl = Params(We=1.0958, Bo=0.017, Oh=0.006, b=6.0, h0=3.0, L=12)
        pr = Params(We=1.0958, Bo=0.017, Oh=0.006, b=6.0, h0=3.0, L=12, viscous=:reid)
        @test pl.viscous === :lamb              # default is the published model
        @test pr.viscous === :reid
        # :lamb path must be bit-identical to the hardcoded formulas it replaced.
        for l in 2:12
            @test pl.drop_lambda[l+1] == 0.006 * (l - 1) * (2l + 1)
            @test pl.drop_omega2[l+1] == float(l) * (l - 1) * (l + 2)
        end
        @test_throws ArgumentError Params(We=1.0, Bo=0.01, Oh=0.01, b=6.0, h0=3.0,
                                          viscous=:bogus)
    end

    @testset "the two models give different dynamics, and :reid damps less" begin
        # End-to-end: the coefficients reach drop_affine and change the affine map.
        dt = 1e-3
        for viscous in (:lamb, :reid)
            p = Params(We=1.0958, Bo=0.017, Oh=0.1, b=6.0, h0=3.0, L=8, viscous=viscous)
            d0 = DropModeState(p.L)
            lam, gam = drop_affine(d0, d0, p, dt, dt)
            @test all(isfinite, lam) && all(isfinite, gam)
        end
        pl = Params(We=1.0958, Bo=0.017, Oh=0.1, b=6.0, h0=3.0, L=8)
        pr = Params(We=1.0958, Bo=0.017, Oh=0.1, b=6.0, h0=3.0, L=8, viscous=:reid)
        d0 = DropModeState(pl.L)
        lam_l, _ = drop_affine(d0, d0, pl, dt, dt)
        lam_r, _ = drop_affine(d0, d0, pr, dt, dt)
        @test lam_l[3] != lam_r[3]
        # Less damping => a larger pressure->mode response magnitude at fixed dt.
        @test abs(lam_r[3]) > abs(lam_l[3])
    end
end
