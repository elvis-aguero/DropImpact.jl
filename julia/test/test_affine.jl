@testset "affine.jl" begin
    @testset "bdf2_coeffs reduces to BDF1 at dt_prev<=0" begin
        a, b, c = bdf2_coeffs(0.01, 0.0)
        @test (a, b, c) == (1.0, -1.0, 0.0)
    end

    @testset "bdf2_coeffs matches the standard variable-step formula" begin
        dt, dtprev = 0.02, 0.015
        s = dt / dtprev
        a, b, c = bdf2_coeffs(dt, dtprev)
        @test a ≈ (1 + 2s) / (1 + s)
        @test b ≈ -(1 + s)
        @test c ≈ s^2 / (1 + s)
    end

    @testset "bath_affine vs a direct from-scratch BDF2 linear solve" begin
        # Independent re-derivation (not reusing affine.jl's own algebra): BDF2 applied
        # to a''=-2γa'-ω²a+Fc directly gives a 2-unknown (a^{k+1}, its derivative
        # eliminated) linear equation in a^{k+1}; solve that raw system here and check
        # it matches bath_affine's alpha+kappa*c prediction — this is exactly the kind
        # of independent check that caught a sign error in this formula earlier.
        p = Params(We=1.0, Bo=0.3, Oh=0.07, M=3, L=3, N=1, b=6.0, h0=2.0, nq=10)
        dt, dtprev = 0.013, 0.021
        a_bdf, b_bdf, c_bdf = bdf2_coeffs(dt, dtprev)
        bath_curr = BathModeState([0.1, -0.05, 0.02, 0.01], [0.3, 0.1, -0.2, 0.05])
        bath_prev = BathModeState([0.08, -0.04, 0.015, 0.005], [0.25, 0.05, -0.1, 0.02])
        kappa, alpha = bath_affine(bath_curr, bath_prev, p, dt, dtprev)

        for m in 0:p.M
            km = p.k[m+1]
            γ = 2 * p.Oh * km^2
            ω2 = (km^2 + p.Bo) * km * tanh(km * p.h0)
            F = -2 * km * tanh(km * p.h0)
            xk, xkm1 = bath_curr.a[m+1], bath_prev.a[m+1]
            yk, ykm1 = bath_curr.adot[m+1], bath_prev.adot[m+1]

            # Raw BDF2: (a_bdf*x_new + b_bdf*xk + c_bdf*xkm1)/dt = y_new (defines y_new),
            # and analogously for y_new from (a''=y'): (a_bdf*y_new+b_bdf*yk+c_bdf*ykm1)/dt
            # = -2γ y_new - ω2 x_new + F*cm. Two linear equations in (x_new,y_new); solve
            # directly with a 2x2 solve, not reusing bath_affine's own closed form.
            cm_test = 0.37
            # eliminate y_new = (a_bdf*x_new+b_bdf*xk+c_bdd*xkm1)/dt into the 2nd equation:
            # a_bdf*(a_bdf*x_new+b_bdf*xk+c_bdf*xkm1)/dt/dt + b_bdf*yk/... this is exactly
            # what affine.jl derives; instead solve the 2x2 system numerically for a fresh check.
            AA = [a_bdf/dt 0.0; -1.0 a_bdf/dt]
            # row1: y_new - (a_bdf x_new)/dt = (b_bdf xk + c_bdf xkm1)/dt
            # row2 (a''=y'): (a_bdf y_new)/dt + 2γ y_new + ω2 x_new = -(b_bdf yk+c_bdf ykm1)/dt + F cm
            AA = [-a_bdf/dt 1.0; ω2 (a_bdf/dt+2γ)]
            rhs = [(b_bdf*xk + c_bdf*xkm1)/dt, -(b_bdf*yk + c_bdf*ykm1)/dt + F*cm_test]
            sol = AA \ rhs
            x_new_direct = sol[1]
            x_new_affine = alpha[m+1] + kappa[m+1] * cm_test
            @test x_new_direct ≈ x_new_affine rtol = 1e-9
        end
    end

    @testset "piston mode (m=0, k_0=0) degeneracy" begin
        p = Params(We=1.0, Bo=0.3, Oh=0.07, M=3, L=3, N=1, b=6.0, h0=2.0, nq=10)
        bath = BathModeState(p.M)
        kappa, alpha = bath_affine(bath, bath, p, 0.01, 0.01)
        @test kappa[1] == 0.0
    end

    @testset "com_affine matches direct BDF2 for a pure-gravity (f=0) ballistic step" begin
        p = Params(We=1.0, Bo=0.4, Oh=0.07, M=2, L=2, N=1, b=6.0, h0=2.0, nq=10)
        dt, dtprev = 0.01, 0.01
        com_curr = COMState(1.0, -0.9)
        com_prev = COMState(1.009, -0.896)  # roughly consistent with z''=-Bo history
        kappa_cm, mu = com_affine(com_curr, com_prev, p, dt, dtprev)
        z_new = mu  # f=0
        a_bdf, b_bdf, c_bdf = bdf2_coeffs(dt, dtprev)
        # direct: (a_bdf z_new + b_bdf zk + c_bdf zkm1)/dt = v_new, and BDF2 on v'=-Bo:
        # (a_bdf v_new + b_bdf vk + c_bdf vkm1)/dt = -Bo
        AA = [-a_bdf/dt 1.0; 0.0 a_bdf/dt]
        rhs = [(b_bdf*com_curr.z + c_bdf*com_prev.z)/dt, -(b_bdf*com_curr.v + c_bdf*com_prev.v)/dt - p.Bo]
        sol = AA \ rhs
        @test sol[1] ≈ z_new rtol = 1e-9
    end
end
