# Regression tests for the RESOLVABLE-RANK (bandwidth) law of the compliance operator.
#
# WHY THIS FILE EXISTS. The resolvable-rank law was established in
# `derivations/audit_compliance_operator.jl` (AUDIT 4) with decoupled M/L and n_q refined
# to independence, but lived only in a hand-run script. In its absence from CI, a
# DIFFERENT and unverified explanation -- "the compliance operator is compact, hence
# pointwise pressure is not resolvable at any finite M,N" -- was asserted in prose
# (`src/residual.jl`, `paper-formulation.tex`) and drove design conclusions for some time.
# That claim was not merely unverified but unfalsifiable as stated: the ASSEMBLED operator
# is a sum of at most M+(L-1)+1 separable terms, hence finite rank, so "singular values
# accumulating at zero" is a statement about a continuum object the testbed never forms.
#
# The three tests below are exactly the checks that would have caught it. They pin the
# mechanism to the joint truncation, where it is measurable, and they are cheap enough to
# run every CI pass.
#
# All numbers are measured, not predicted; thresholds are deliberately loose (order-of-
# magnitude separations, not fitted constants) so that these test the MECHANISM and do not
# become a fossil of one machine's roundoff.

using SpectralKM
using SpectralKM: legendre_P_table
using LinearAlgebra
using Test

include(joinpath(@__DIR__, "..", "derivations", "audit_compliance_operator_core.jl"))

"""
Numerical rank of the symmetrized compliance operator at a frozen geometry: the number of
pressure directions the truncated bath+droplet respond to above a relative threshold.
This is `n_*` in `derivations/DIAGNOSTICS-NOTATION.md`.
"""
function resolvable_rank(; M, L, theta_c, nq, delta=1e-3, thr=1e-8)
    p = Params(We=1.0958, Bo=0.017, Oh=0.006, M=M, L=L, N=4, b=6.0, h0=3.0, nq=nq, check_budget=false)
    A, _, _, _, w, x, om = compliance(p, theta_c, zeros(p.L + 1), delta)
    n = length(x)
    S = [w[i] * (-A[i, j]) / om[j] for i in 1:n, j in 1:n]
    sv = svdvals(Symmetric((S + S') / 2))
    sv ./= sv[1]
    return count(>(thr), sv)
end

"""Largest singular value of the same operator -- used to prove a delta-invariance test
is not vacuous (i.e. that delta genuinely enters the assembly)."""
function top_singular_value(; M, L, theta_c, nq, delta)
    p = Params(We=1.0958, Bo=0.017, Oh=0.006, M=M, L=L, N=4, b=6.0, h0=3.0, nq=nq, check_budget=false)
    A, _, _, _, w, x, om = compliance(p, theta_c, zeros(p.L + 1), delta)
    n = length(x)
    S = [w[i] * (-A[i, j]) / om[j] for i in 1:n, j in 1:n]
    return maximum(svdvals(Symmetric((S + S') / 2)))
end

"""cond of the inner Galerkin matrix in the production shifted-Legendre pressure basis."""
function galerkin_cond(; M, L, N, theta_c, nq, delta=1e-3)
    p = Params(We=1.0958, Bo=0.017, Oh=0.006, M=M, L=L, N=N, b=6.0, h0=3.0, nq=nq, check_budget=false)
    A, _, _, _, w, x, om = compliance(p, theta_c, zeros(p.L + 1), delta)
    n = length(x)
    xc = cos(theta_c)
    B = [[legendre_P_table(N, 2 * (x[i] - xc) / (1 - xc) - 1)[k+1] for k in 0:N] for i in 1:n]
    Ap = [[sum(A[i, j] * B[j][b] for j in 1:n) for i in 1:n] for b in 1:N+1]
    J = [sum(B[i][a] * w[i] * om[i] * Ap[b][i] for i in 1:n) for a in 1:N+1, b in 1:N+1]
    return cond(J)
end

@testset "rank_law.jl (resolvable-pressure-rank / bandwidth law)" begin

    # ------------------------------------------------------------------
    # (1) The rank is set by the DROPLET bandwidth L*theta_c, not by the
    #     bath's k_M*r_c. This is AUDIT 4's conclusion, promoted to CI.
    #
    #     The decoupling matters: an earlier study held M = L, in which
    #     k_M*r_c and L*theta_c are collinear, and so had no power to
    #     attribute the rank to either block. It wrongly credited the bath.
    # ------------------------------------------------------------------
    @testset "rank tracks the droplet block, not the bath block" begin
        tc, nq = 0.4, 160

        # Starve the DROPLET, keep the bath generous -> rank must collapse.
        r_bath_rich_drop_poor = resolvable_rank(M=160, L=10, theta_c=tc, nq=nq)
        # Starve the BATH, keep the droplet generous -> rank must survive.
        r_bath_poor_drop_rich = resolvable_rank(M=10, L=160, theta_c=tc, nq=nq)

        @test r_bath_poor_drop_rich > r_bath_rich_drop_poor
        # Measured at theta_c=0.4, nq=160, thr=1e-8: 25 (droplet-rich, 16x FEWER bath
        # modes) against 15 (bath-rich). The bath block does contribute -- the claim is
        # that the droplet block DOMINATES, not that the bath is irrelevant -- so the
        # margin asserted here is the measured 1.67x, floored well below it.
        @test r_bath_poor_drop_rich > 1.4 * r_bath_rich_drop_poor
    end

    @testset "rank grows with L at fixed bath and theta_c" begin
        tc, nq = 0.4, 160
        ranks = [resolvable_rank(M=160, L=Lv, theta_c=tc, nq=nq) for Lv in (10, 40, 160)]
        @test issorted(ranks)
        @test ranks[end] > ranks[1]
    end

    @testset "rank grows with the contact angle at fixed truncations" begin
        # theta_c enters the law through the SAME product L*theta_c, so widening the patch
        # buys resolvable directions exactly as refining the droplet does. This is why the
        # budget is tightest at contact onset.
        ranks = [resolvable_rank(M=80, L=80, theta_c=tc, nq=160) for tc in (0.1, 0.3, 0.6)]
        @test issorted(ranks)
        @test ranks[end] > ranks[1]
    end

    # ------------------------------------------------------------------
    # (2) THE TEST THAT WOULD HAVE FALSIFIED THE COMPACTNESS CLAIM.
    #
    #     If the rank were set by a continuum smoothing property, it would move with the
    #     time-step parameter delta, which sets where the compliance symbol turns over
    #     from growing to decaying (k_* = (a/delta)^{2/3}). It essentially does not: across
    #     FOUR DECADES of delta the rank moves by at most one index, while sigma_1 itself
    #     moves nearly five orders of magnitude.
    #
    #     Stated carefully, because the naive version of this claim is overfitted: the rank
    #     is *exactly* constant only at thr=1e-10, which is an artifact of that threshold.
    #     Measured at M=L=80, theta_c=0.3, nq=160, delta = 1e-4 ... 1e0:
    #       thr=1e-8   -> 12, 12, 11, 11, 11
    #       thr=1e-10  -> 12, 12, 12, 12, 12
    #       thr=1e-12  -> 14, 14, 13, 13, 13
    #     The threshold-robust invariant is therefore "spread <= 1 index", and that is what
    #     is asserted, at all three thresholds. A compactness-driven rank would move with
    #     delta far more than one index over four decades.
    # ------------------------------------------------------------------
    @testset "rank is near-invariant in delta while the operator scale is not" begin
        M, L, tc, nq = 80, 80, 0.3, 160
        deltas = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)

        for thr in (1e-8, 1e-10, 1e-12)
            ranks = [resolvable_rank(M=M, L=L, theta_c=tc, nq=nq, delta=d, thr=thr)
                     for d in deltas]
            @test maximum(ranks) - minimum(ranks) <= 1
        end

        # Guard against a vacuous invariance: delta must genuinely enter the assembly.
        # Measured spread is ~8e4; asserted loosely.
        s1 = [top_singular_value(M=M, L=L, theta_c=tc, nq=nq, delta=d) for d in deltas]
        @test maximum(s1) / minimum(s1) > 1e3
    end

    @testset "rank is independent of the quadrature once n_q is generous" begin
        # The rank is a property of the truncations, not of the node count. (rank <= n_q
        # caps it only when n_q is starved -- a confound in an earlier revision of AUDIT 4.)
        # Measured: 14 at every nq in 80, 160, 240, 320.
        ranks = [resolvable_rank(M=80, L=80, theta_c=0.4, nq=nqv) for nqv in (80, 160, 240)]
        @test all(==(ranks[1]), ranks)
    end

    # ------------------------------------------------------------------
    # (3) BUDGET GUARD: cond(J_N) blowing up under N-refinement is a JOINT-truncation
    #     artifact, removable by refining M,L -- not an intrinsic limit on the pressure.
    #
    #     Measured at N=12, theta_c=0.3, production basis, nothing else changed:
    #       M=L=80  -> cond ~ 5.0e10
    #       M=L=320 -> cond ~ 7.2e1
    # ------------------------------------------------------------------
    @testset "joint refinement removes the N-refinement conditioning blow-up" begin
        c_coarse = galerkin_cond(M=80,  L=80,  N=12, theta_c=0.3, nq=200)
        c_fine   = galerkin_cond(M=320, L=320, N=12, theta_c=0.3, nq=200)

        @test c_coarse > 1e8          # the blow-up is real at the coarse truncation
        @test c_fine < 1e5            # and it is GONE after joint refinement
        @test c_coarse / c_fine > 1e5 # by many orders, same basis, same N
    end

    @testset "cond stays moderate while N is inside the budget" begin
        # At M=L=80, theta_c=0.3 the measured rank is ~13-14, so N+1 <= ~8 is well inside
        # the budget and must not be ill-conditioned. This is the invariant production
        # relies on: provision N against the budget rather than upward for safety margin.
        for N in (1, 3, 6)
            @test galerkin_cond(M=80, L=80, N=N, theta_c=0.3, nq=200) < 1e4
        end
    end

end

@testset "numerical defaults (physics-only Params construction)" begin
    # The point of the defaults: a caller supplies physics and nothing else. The values are
    # justified beside DEFAULT_M in src/types.jl from the measured cost asymmetry (L nearly
    # free, M and nq linear, N cheap only inside the budget).
    p = Params(We=1.0958, Bo=0.017, Oh=0.006, b=6.0, h0=3.0)
    @test p.viscous === SpectralKM.DEFAULT_VISCOUS
    @test p.M == SpectralKM.DEFAULT_M
    @test p.L == SpectralKM.DEFAULT_L
    @test p.N == SpectralKM.DEFAULT_N
    @test p.nq == SpectralKM.DEFAULT_NQ

    @testset "the defaults are themselves inside the budget" begin
        # This is the invariant that matters: shipping a default that violates the budget
        # would hand every user the ill-conditioned regime. Checked at a tight post-onset
        # angle, not the onset limit (where n_* -> its floor and no N would pass).
        for tc in (0.15, 0.3, 0.6)
            budget = 0.6 * resolvable_rank_estimate(M=p.M, L=p.L, b=p.b, theta_c=tc)
            @test p.N + 1 <= budget
        end
    end

    @testset "defaults integrate the COM force exactly" begin
        # nq must exceed the exactness requirement for the COM integrand at the default N,L.
        @test p.nq >= min_nq_for_exact_com(p.N, p.L)
    end

    @testset "explicit truncations still override" begin
        q = Params(We=1.0, Bo=0.01, Oh=0.01, M=80, L=80, N=3, b=6.0, h0=3.0, nq=200)
        @test (q.M, q.L, q.N, q.nq) == (80, 80, 3, 200)
    end

    @testset "budget guard warns on an over-budget N and is suppressible" begin
        # L=40 with N=12 is far outside the budget: the regime measured at 90% of contact
        # steps carrying negative pressure and ~6x the walltime.
        @test_logs (:warn,) match_mode=:any Params(We=1.0958, Bo=0.017, Oh=0.006,
                                                   b=6.0, h0=3.0, L=40, N=12)
        @test_logs Params(We=1.0958, Bo=0.017, Oh=0.006, b=6.0, h0=3.0,
                          L=40, N=12, check_budget=false)
    end

    @testset "resolvable_rank_estimate grows with L and theta_c, mildly with M" begin
        est(; M=60, L=120, tc=0.3) = resolvable_rank_estimate(M=M, L=L, b=6.0, theta_c=tc)
        @test est(L=240) > est(L=120) > est(L=60)
        @test est(tc=0.6) > est(tc=0.3) > est(tc=0.15)
        # The bath enters, but weakly compared with the droplet: doubling M must move the
        # estimate less than doubling L does.
        @test (est(M=120) - est()) < (est(L=240) - est())
    end
end
