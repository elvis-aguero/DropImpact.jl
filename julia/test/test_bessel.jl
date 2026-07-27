@testset "bessel.jl" begin
    @testset "zeros vs SpecialFunctions.besselj1" begin
        z = bessel_zeros_J1(10)
        @test z[1] == 0.0
        for m in 1:10
            @test abs(besselj0(0) - 1.0) < 1e-15  # sanity: J0(0)=1, unrelated but cheap
            @test abs(SpecialFunctions.besselj1(z[m+1])) < 1e-12
        end
        @test issorted(z)
        # tabulated first three positive zeros of J1 (Abramowitz & Stegun)
        tabulated = [3.8317059702, 7.0155866698, 10.1734681351]
        for m in 1:3
            @test z[m+1] ≈ tabulated[m] atol = 1e-9
        end
    end

    @testset "ForwardDiff through SpecialFunctions Bessel — the single highest-leverage risk" begin
        # Exact call pattern used throughout bessel_moments.jl/accel_closure.jl:
        # a Float64 constant (k_m) times a Dual (from x -> r(x) depending on the
        # differentiated unknown), fed into besselj0/besselj1.
        km = 3.8317059702
        f0(x) = besselj0(km * x)
        f1(x) = besselj1(km * x)
        for x0 in (0.3, 1.7)
            d0 = ForwardDiff.derivative(f0, x0)
            d1 = ForwardDiff.derivative(f1, x0)
            @test d0 ≈ -km * besselj1(km * x0) atol = 1e-10          # J0' = -J1
            @test d1 ≈ km * (besselj0(km * x0) - besselj1(km * x0) / (km * x0)) atol = 1e-10  # J1' = J0-J1/x
        end
    end
end
