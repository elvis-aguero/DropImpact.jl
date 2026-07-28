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
        # Exact call pattern used throughout bessel_moments.jl:
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

    @testset "pinned wall: J_0 zeros, pinning exact, wall slope free" begin
        # Route (A) of derivations/feasibility_pinned_contact_line.jl. The two bases must
        # each do exactly what their label claims at the wall, and neither can do both.
        for M in (4, 8)
            z = bessel_zeros_J0(M)
            @test length(z) == M + 1
            @test all(zi -> abs(SpecialFunctions.besselj0(zi)) < 1e-12, z)
            @test issorted(z) && first(z) > 2.0      # no k=0 mode: 2.4048 is the first
        end
        b = 6.0
        pf = Params(We=1.0, Bo=0.2, Oh=0.05, M=6, L=6, N=1, b=b, h0=2.0, nq=16, wall=:free)
        pp = Params(We=1.0, Bo=0.2, Oh=0.05, M=6, L=6, N=1, b=b, h0=2.0, nq=16, wall=:pinned)
        # free: every mode has zero SLOPE at the wall, nonzero value
        @test all(k -> abs(k * SpecialFunctions.besselj1(k * b)) < 1e-11, pf.k)
        @test maximum(abs(SpecialFunctions.besselj0(k * b)) for k in pf.k) > 0.1
        # pinned: every mode VANISHES at the wall, slope free
        @test all(k -> abs(SpecialFunctions.besselj0(k * b)) < 1e-11, pp.k)
        @test maximum(abs(k * SpecialFunctions.besselj1(k * b)) for k in pp.k) > 0.1
    end

    @testset "Fourier-Bessel weight equals 1/‖J_0(k_m ·)‖² for both wall conditions" begin
        # The coefficient is <p,J_0>/‖J_0‖², so the weight is 1/I with
        # I = ∫_0^b J_0(k r)² r dr -- which reduces to 2/(b J_0(k b))² for :free and
        # 2/(b J_1(k b))² for :pinned, the two expressions Params installs.
        simpson(f, a, c, n) = begin
            h = (c - a) / n; s = f(a) + f(c)
            for i in 1:n-1; s += (isodd(i) ? 4 : 2) * f(a + i * h); end
            s * h / 3
        end
        for wall in (:free, :pinned)
            p = Params(We=1.0, Bo=0.2, Oh=0.05, M=5, L=5, N=1, b=6.0, h0=2.0, nq=16, wall=wall)
            for m in 0:p.M
                km = p.k[m+1]
                I = simpson(r -> SpecialFunctions.besselj0(km * r)^2 * r, 0.0, p.b, 20_000)
                @test p.bath_norm[m+1] ≈ 1 / I rtol = 1e-8
            end
        end
    end
end
