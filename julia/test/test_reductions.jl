@testset "structural reductions" begin
    @testset "drop_affine: l=0,1 entries are structurally zero (never evolved)" begin
        # b_l_all's own l=0,1 entries are NOT numerically zero (its docstring: "computed
        # here anyway for a uniform 1-indexed vector"), but drop_affine's lambda/gam ARE
        # exactly zero there since those modes are simply never advanced — the real
        # structural invariant.
        p = Params(We=1.0, Bo=0.2, Oh=0.05, M=3, L=5, N=2, b=6.0, h0=2.0, nq=16)
        drop = DropModeState(p.L)
        lambda, gam = drop_affine(drop, drop, p, 0.01, 0.01)
        @test lambda[1] == 0.0 && lambda[2] == 0.0
        @test gam[1] == 0.0 && gam[2] == 0.0
    end

    @testset "K_of_x at the undisturbed (zero) frozen state reduces to bare Bo" begin
        # a=adot=beta=betadot=0 everywhere: every bath/drop term in K vanishes by
        # inspection (each is linear or higher in the frozen state), leaving only the
        # constant +Bo from z_cm''=1.5f-Bo's gravity term.
        p = Params(We=1.0, Bo=0.37, Oh=0.05, M=5, L=5, N=1, b=6.0, h0=2.0, nq=16)
        a0 = zeros(p.M + 1); beta0 = zeros(p.L + 1)
        for x in (-0.5, 0.0, 0.6, 0.95)
            @test K_of_x(x, a0, a0, beta0, beta0, p) ≈ p.Bo
        end
    end

    @testset "Pi_of_x at zero pressure moments is zero" begin
        p = Params(We=1.0, Bo=0.2, Oh=0.05, M=5, L=5, N=1, b=6.0, h0=2.0, nq=16)
        a0 = zeros(p.M + 1); beta0 = zeros(p.L + 1)
        cm0 = zeros(p.M + 1); bl0 = zeros(p.L + 1)
        for x in (-0.5, 0.0, 0.6, 0.95)
            @test Pi_of_x(x, cm0, bl0, 0.0, a0, beta0, p) ≈ 0.0 atol = 1e-14
        end
    end

    @testset "outer_bracket_of_x reduces to -cosθ=-x when a_frozen is zero" begin
        # SIGN CORRECTED 2026-07-27: the bracket is -cosθ+sinθΣa_mk_mJ1(...), not
        # +cosθ+... — z_d enters the contact condition C with the opposite sign from
        # an earlier version of both the design doc and this code (see
        # accel_closure.jl's outer_bracket_of_x docstring).
        p = Params(We=1.0, Bo=0.2, Oh=0.05, M=5, L=5, N=1, b=6.0, h0=2.0, nq=16)
        a0 = zeros(p.M + 1); beta0 = zeros(p.L + 1)
        for x in (-0.5, 0.0, 0.6, 0.95)
            @test outer_bracket_of_x(x, a0, beta0, p) ≈ -x
        end
    end

    @testset "xi_tau_of_x vanishes with L=2 (no evolving drop modes contribute)" begin
        p = Params(We=1.0, Bo=0.2, Oh=0.05, M=3, L=2, N=1, b=6.0, h0=2.0, nq=16)
        betadot = zeros(p.L + 1)
        @test xi_tau_of_x(betadot, 0.3, p.L) == 0.0
    end

    @testset "min_nq_for_exact_com matches its documented degree formula" begin
        for N in 0:3, L in 2:6
            @test min_nq_for_exact_com(N, L) == cld(N + 2L + 2, 2)
        end
    end
end
