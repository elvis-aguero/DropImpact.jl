@testset "conditioning.jl (Jacobian conditioning regression checks)" begin
    @testset "piston mode (m=0, k_0=0) never contributes to the bath Jacobian slope" begin
        for M in (1, 5, 10)
            p = Params(We=1.0, Bo=0.2, Oh=0.05, M=M, L=5, N=1, b=6.0, h0=2.0, nq=16)
            bath = BathModeState(M)
            kappa, _ = bath_affine(bath, bath, p, 0.01, 0.01)
            @test kappa[1] == 0.0
        end
    end

    @testset "accel-level closure: cond(J) is flat in dt (regression guard against the confirmed O(δ²) position-level ill-conditioning this closure replaces)" begin
        p = Params(We=1.0, Bo=0.1, Oh=0.05, M=10, L=10, N=1, b=6.0, h0=3.0, nq=20)
        lvl0 = initial_level(p)
        hist = SimHistory(lvl0, lvl0)
        X0 = warm_start(nothing, p.N)
        conds = Float64[]
        for dt in (1e-2, 1e-3, 1e-4, 1e-5)
            R, _ = build_residual(hist, dt, p)
            J = ForwardDiff.jacobian(R, X0)
            push!(conds, cond(J))
        end
        # All within a small factor of each other, NOT growing by orders of magnitude —
        # the old position-level closure's per-singular-value dt^2 scaling would fail
        # this badly (e.g. >1e6x variation across this dt range).
        @test maximum(conds) / minimum(conds) < 10.0
    end
end
