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

    @testset "Lamb OVERPREDICTS damping, increasingly with Oh and with l" begin
        # This is the substantive physical claim and the motivation for the whole module.
        # It is asserted only in the UNDERDAMPED regime: once Lamb predicts overdamping
        # (lambda_Lamb > omega_Lamb) the least-damped Reid branch is no longer the complex
        # pair, the Lamb-seeded Newton stops following it, and `drop_viscous_coeffs` falls
        # back to Lamb by design. Measured example of the failure this guards:
        # Oh=0.3, l=16 returns a purely real root with lambda = 288.6 against Lamb's 148.5.
        for Oh in (0.006, 0.1, 0.3), l in (2, 4, 8, 16)
            lam_lamb = Oh * (l - 1) * (2l + 1)
            om_lamb = sqrt(float(l) * (l - 1) * (l + 2))
            lam_lamb < om_lamb || continue          # underdamped only
            lam, _, _ = reid_root(l, Oh)
            @test lam_lamb > lam
        end
        # Monotone in Oh at fixed l (measured at l=2: 4.1%, 22.9%, 36.7%).
        rel(Oh, l) = (Oh * (l - 1) * (2l + 1) - reid_root(l, Oh)[1]) / reid_root(l, Oh)[1]
        @test rel(0.3, 2) > rel(0.1, 2) > rel(0.006, 2)
        # Monotone in l at fixed Oh (measured at Oh=0.006: 4.1% -> 11.9% over l=2..16).
        @test rel(0.006, 16) > rel(0.006, 8) > rel(0.006, 2)
        # And the magnitude at our own production Oh is not negligible at high l.
        @test rel(0.006, 16) > 0.05
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

    @testset "overdamped modes fall back to Lamb rather than returning a wrong branch" begin
        # Oh = 0.3 puts l = 16 past Lamb's overdamping threshold (Oh_crit ~ 0.13 there).
        lam_r, om2_r = (@test_logs (:warn,) match_mode=:any drop_viscous_coeffs(16, 0.3, :reid))
        @test lam_r[17] == 0.3 * (16 - 1) * (2 * 16 + 1)      # exactly Lamb
        @test om2_r[17] == float(16) * 15 * 18
        # while the low, still-underdamped modes keep their Reid values
        @test lam_r[3] < 0.3 * (2 - 1) * (2 * 2 + 1)
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
