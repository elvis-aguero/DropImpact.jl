@testset "newton.jl" begin
    @testset "method of manufactured solutions — linear system" begin
        Atrue = [2.0 0.3; -0.1 1.5]
        btrue = [0.4, -0.2]
        R(X) = Atrue * X - btrue
        result = newton_solve(R, [0.0, 0.0])
        @test result.status == Converged
        @test result.X ≈ Atrue \ btrue atol = 1e-6
    end

    @testset "method of manufactured solutions — mildly nonlinear" begin
        Xtarget = [0.3, -0.5, 0.1]
        R(X) = [X[1]^2 - Xtarget[1]^2, sin(X[2]) - sin(Xtarget[2]), X[3]^3 - Xtarget[3]^3]
        result = newton_solve(R, Xtarget .+ [0.05, -0.03, 0.02])
        @test result.status == Converged
        @test result.X ≈ Xtarget atol = 1e-5
    end

    @testset "near-singular Jacobian: LM damping still returns a finite, usable result" begin
        # A deliberately rank-deficient-near X0 system (mirrors the confirmed severe
        # ill-conditioning risk this solver was built to survive).
        R(X) = [X[1] + X[2] - 1.0, 1e-10 * (X[1] - X[2])]
        result = newton_solve(R, [0.0, 0.0])
        @test all(isfinite, result.X)
        @test result.status in (Converged, Stalled, MaxIterExceeded)
    end

    @testset "never throws on a pathological/non-finite-producing residual" begin
        R(X) = [1.0 / X[1], X[2]]  # blows up at X[1]=0
        result = newton_solve(R, [0.0, 1.0])
        @test result.status == Stalled
    end
end
