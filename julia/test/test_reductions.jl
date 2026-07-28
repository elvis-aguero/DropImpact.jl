@testset "structural reductions" begin
    @testset "drop_affine: l=0,1 entries are structurally zero (never evolved)" begin
        p = Params(We=1.0, Bo=0.2, Oh=0.05, M=3, L=5, N=2, b=6.0, h0=2.0, nq=16)
        drop = DropModeState(p.L)
        lambda, gam = drop_affine(drop, drop, p, 0.01, 0.01)
        @test lambda[1] == 0.0 && lambda[2] == 0.0
        @test gam[1] == 0.0 && gam[2] == 0.0
    end

    @testset "at beta = 0 the area weight w reduces exactly to x" begin
        # r^2 = 1 - x^2 for an undeformed droplet, so w = -d[r^2]/dx / 2 = x. This identity
        # is why measurements taken only at the undeformed state cannot distinguish the two
        # candidate mechanisms for the droplet block's asymmetry (design doc
        # §subsubsec:compliance) -- worth pinning so the reduction is not silently lost.
        p = Params(We=1.0, Bo=0.2, Oh=0.05, M=3, L=6, N=2, b=6.0, h0=2.0, nq=16)
        beta0 = zeros(p.L + 1)
        for x in (0.1, 0.5, 0.9, 0.999)
            @test w_of_x(beta0, x, p.L) ≈ x rtol = 1e-12
        end
    end

    @testset "b_l reduces to the plain Legendre moment at pole contact" begin
        # eq:b_l-selfadjoint carries an extra x*w relative to AlventosaEtAl2023's
        # b_l = int p P_l dx. Since x -> 1 and w -> 1 as the patch shrinks to the pole, the
        # two must agree there: the departure is O(theta_c^2), not a change of leading-order
        # physics. Checked by comparing against a direct plain-measure quadrature.
        p = Params(We=1.0, Bo=0.2, Oh=0.05, M=3, L=4, N=1, b=6.0, h0=2.0, nq=24)
        beta0 = zeros(p.L + 1)
        chat = [1.0, 0.0]
        prev = Inf
        for theta_c in (0.4, 0.2, 0.1, 0.05)
            xc = cos(theta_c)
            x, wq = mapped_nodes(xc, p.gauss_nodes, p.gauss_weights)
            P, dP = legendre_tables(x, p.L)
            _, _, w = geom_at_nodes(beta0, x, P, dP, p.L)
            bl = b_l_all(chat, xc, x, wq, P, w, p.L)
            # plain-measure reference: int p P_l dx, no x and no w
            plain = [sum(wq[i] * pressure_poly_raw(chat, xc, x[i]) * P[i][l+1]
                         for i in eachindex(x)) for l in 0:p.L]
            rel = maximum(abs.(bl .- plain)) / max(maximum(abs.(plain)), eps())
            @test rel < prev            # gap closes monotonically as the patch shrinks
            prev = rel
        end
        @test prev < 0.02                # and is small by theta_c = 0.05
    end

    @testset "com force is positive for positive pressure, and exact for p == 1" begin
        # f = 2 int p w dx with w = x at beta = 0, so p == 1 gives f = 1 - xc^2 = r_c^2,
        # matching the direct radial integral 2 int_0^{r_c} r dr. The SIGN matters: getting
        # it backwards makes the pressure accelerate the droplet into the bath.
        p = Params(We=1.0, Bo=0.2, Oh=0.05, M=3, L=4, N=0, b=6.0, h0=2.0, nq=24)
        beta0 = zeros(p.L + 1)
        for theta_c in (0.1, 0.3, 0.7)
            xc = cos(theta_c)
            x, wq = mapped_nodes(xc, p.gauss_nodes, p.gauss_weights)
            P, dP = legendre_tables(x, p.L)
            _, _, w = geom_at_nodes(beta0, x, P, dP, p.L)
            f = com_force_closed([1.0], xc, x, wq, w)
            @test f > 0
            @test f ≈ sin(theta_c)^2 rtol = 1e-10
        end
    end

    @testset "c_m at zero pressure is zero, and the m=0 weight is regular" begin
        # Under the corrected normalization 2/(b J0(k_m b))^2 (design doc eq:bessel-norm)
        # the m=0 piston mode is no longer singular: J0(0)=1 gives the finite weight 2/b^2.
        p = Params(We=1.0, Bo=0.2, Oh=0.05, M=5, L=4, N=1, b=6.0, h0=2.0, nq=20)
        beta0 = zeros(p.L + 1)
        xc = cos(0.3)
        x, wq = mapped_nodes(xc, p.gauss_nodes, p.gauss_weights)
        P, dP = legendre_tables(x, p.L)
        _, r, w = geom_at_nodes(beta0, x, P, dP, p.L)
        cm = c_m_all(zeros(p.N + 1), xc, x, wq, r, w, p.k, p.bath_norm)
        @test all(iszero, cm)
        cm1 = c_m_all([1.0, 0.0], xc, x, wq, r, w, p.k, p.bath_norm)
        @test all(isfinite, cm1)          # including m = 0
        @test cm1[1] != 0                 # the piston mode's projection is nonzero...
        kappa, _ = bath_affine(BathModeState(p.M), BathModeState(p.M), p, 0.01, 0.01)
        @test kappa[1] == 0.0             # ...but its pressure response is exactly zero
    end
end
