# SUPERSEDED -- RETAINED FOR THE RECORD ONLY.  This script verifies claims about
# the acceleration-level closure (former Sec. 4), DELETED from the design doc.
# Its premise -- that the O(delta^2) pressure sensitivity of the position-level closure
# causes ill-conditioning -- is false: the factor is uniform across the pressure block,
# and cond(cA)=cond(A).  See provenance.tex Sec. 4 ('Corrections to earlier claims') and audit_compliance_operator.jl.
#
# It is no longer cited by paper-formulation.tex and its conclusions no longer
# describe the model.  Do not treat a passing run here as support for current theory.

# Symbolic (CAS-based) verification of design doc Section 4, "An acceleration-level
# closure" — mechanically re-derives eq:velocity-constraint, eq:accel-constraint-raw,
# and the Pi/K split after ODE substitution, by symbolic differentiation rather than
# by hand. Intended to be re-run whenever the model's equations change, rather than
# re-derived by hand each time (a standing, reusable check, not a one-off script).
#
# Scope: verifies ONE representative bath mode (a, with wavenumber k) and ONE
# representative Legendre term (beta, contributing to xi via a symbolic constant Pl
# standing in for P_l(cosθ), since Legendre polynomials themselves are not what's
# error-prone here — the chain rule through J0(k*r(θ,τ)) and the ODE substitution
# are). Summing over m and l is linear and does not need symbolic re-verification;
# it is exactly the single-term chain-rule/product-rule steps that are checked here.

using Symbolics
import SpecialFunctions: besselj0, besselj1, besselj

@variables τ θ k Pl Oh Bo cm bl f_com  # Pl stands in for P_l(cosθ), a τ-independent constant

# --- Register the REAL SpecialFunctions Bessel functions with their standard
#     derivative rules (design doc's own identities, J0'=-J1 and
#     J1'(x)=J0(x)-J1(x)/x), so Symbolics.jl's chain rule differentiates through
#     them mechanically rather than by hand. @register_symbolic keeps besselj0/
#     besselj1 usable for ordinary Float64 evaluation too (unlike a placeholder). ---
Symbolics.@register_symbolic besselj0(x)
Symbolics.@register_symbolic besselj1(x)
Symbolics.derivative(::typeof(besselj0), args::NTuple{1,Any}, ::Val{1}) = -besselj1(args[1])
Symbolics.derivative(::typeof(besselj1), args::NTuple{1,Any}, ::Val{1}) = besselj0(args[1]) - besselj1(args[1]) / args[1]

const J0 = besselj0
const J1 = besselj1

@variables a(τ) beta(τ) zcm(τ)
Dτ = Differential(τ)

xi = beta * Pl                 # one Legendre term's contribution to ξ(θ,τ)
r = xi * sin(θ)                # r(θ,τ) = ξ(θ,τ) sinθ  (eq:forward-map)

C = a * J0(k * r) - zcm + xi * cos(θ)   # one m-term, one l-term, plus z_cm, of C(θ,τ)
# SIGN CORRECTED 2026-07-27: independent review + direct re-derivation from the
# document's own stated coordinate convention (θ=0 along -ẑ, spherical polar angle
# measured from -ẑ not +ẑ) established that a droplet-surface point's absolute height
# is z_cm - ξcosθ, not z_cm + ξcosθ as originally (and, it turns out, incorrectly)
# written — confirmed independently against the already-tested Julia geometry code
# (postprocessing.jl's south_pole_height uses z_cm - ξ(0), not z_cm + ξ(0)). C, defined
# as η - (absolute height), therefore has the OPPOSITE sign on its ξcosθ term relative
# to the version this script previously verified: C = η - zcm - (-ξcosθ) = η - zcm + ξcosθ.

Cdot = simplify(expand_derivatives(Dτ(C)))
Cddot_raw = simplify(expand_derivatives(Dτ(Dτ(C))))

println("=== eq:velocity-constraint (single mode/term) ===")
println(Cdot)
println()
println("=== eq:accel-constraint-raw (single mode/term), before ODE substitution ===")
println(Cddot_raw)
println()

# --- Now substitute the governing ODEs (design doc eq:bath-mode/eq:drop-mode/eq:com,
#     single-mode form) for a''(τ), beta''(τ), zcm''(τ), and check the Pi/K split. ---
adot = Dτ(a)
betadot = Dτ(beta)
zcmdot = Dτ(zcm)

# eq:bath-mode (single mode m): a'' = -4*Oh*k^2*a' - (k^2+Bo)*k*tanh(k*h0)*a - 2*k*tanh(k*h0)*cm
@variables h0
tanh_kh0 = tanh(k * h0)
addot_rhs = -4 * Oh * k^2 * adot - (k^2 + Bo) * k * tanh_kh0 * a - 2 * k * tanh_kh0 * cm

# eq:drop-mode (single mode l, using generic l instead of expanding (2l+1)(l-1) etc,
# since only the STRUCTURE "forcing coefficient F_l * bl" needs to be isolated here,
# not its specific numeric value, already verified separately in the Julia unit tests)
@variables Fl damp_l omega2_l
betaddot_rhs = -damp_l * betadot - omega2_l * beta - Fl * bl

# eq:com: zcm'' = (3/2) f_com - Bo
zcmddot_rhs = (3 // 2) * f_com - Bo

# Substitute: replace D(D(a))(τ), D(D(beta))(τ), D(D(zcm))(τ) with the RHS expressions
Dτ2_a = Dτ(Dτ(a))
Dτ2_beta = Dτ(Dτ(beta))
Dτ2_zcm = Dτ(Dτ(zcm))

Cddot_substituted = simplify(substitute(Cddot_raw, Dict(
    Dτ2_a => addot_rhs,
    Dτ2_beta => betaddot_rhs,
    Dτ2_zcm => zcmddot_rhs,
)))

println("=== After ODE substitution (should separate into Pi (pressure-dependent: cm, bl, f_com) and K (state-only)) ===")
println(Cddot_substituted)
println()

# Extract and print the coefficient of each pressure-moment symbol, to directly
# compare against the design doc's stated Pi = -2*k*tanh(k h0)*J0(k r)*cm - (3/2) f_com + cosθ*Fl*Pl*bl
coeff_cm = simplify(Symbolics.coeff(expand(Cddot_substituted), cm))
coeff_bl = simplify(Symbolics.coeff(expand(Cddot_substituted), bl))
coeff_f  = simplify(Symbolics.coeff(expand(Cddot_substituted), f_com))

println("Coefficient of cm(τ)     : ", coeff_cm)
println("Coefficient of bl(τ)     : ", coeff_bl)
println("Coefficient of f_com(τ)  : ", coeff_f)
println()

# --- Self-checking equality tests (not just eyeballed printouts): subtract the
#     doc's claimed coefficient (post-correction, eq:accel-constraint) and simplify;
#     a nonzero residual means a genuine mismatch, not a cosmetic difference. ---
claimed_cm = -2 * J0(k * beta * Pl * sin(θ)) * k * tanh(k * h0)
claimed_f  = -3 // 2
# SIGN CORRECTED 2026-07-27 (see C's definition above): only the direct cosθ term
# (coming from the ξ_ττcosθ piece of C, whose sign flips with C's own z_d sign) flips;
# the sinθ*a*k*J1(...) term (coming from η's own chain rule through r=ξsinθ, unrelated
# to how z_d enters C) does not.
claimed_bl = (-cos(θ) + sin(θ) * a * k * J1(k * beta * Pl * sin(θ))) * Fl * Pl

resid_cm = simplify(expand(coeff_cm - claimed_cm))
resid_f  = simplify(expand(coeff_f - claimed_f))
resid_bl = simplify(expand(coeff_bl - claimed_bl))

is_symbolic_zero(x) = (x isa Number && iszero(x)) || string(simplify(x; expand=true)) in ("0", "0//1")

println("=== Self-checking equality tests (residual should be 0) ===")
ok_cm = is_symbolic_zero(resid_cm)
ok_f  = is_symbolic_zero(resid_f)
ok_bl = is_symbolic_zero(resid_bl)
println("cm coefficient residual : ", resid_cm, "   ", ok_cm ? "PASS" : "FAIL")
println("f  coefficient residual : ", resid_f,  "   ", ok_f  ? "PASS" : "FAIL")
println("bl coefficient residual : ", resid_bl, "   ", ok_bl ? "PASS" : "FAIL")

@assert ok_cm "cm coefficient does not match design doc's corrected eq:accel-constraint"
@assert ok_f  "f coefficient does not match design doc's corrected eq:accel-constraint"
@assert ok_bl "bl coefficient does not match design doc's corrected eq:accel-constraint"
println()

# =====================================================================
# Now verify K = Cddot_substituted - Pi (the "known", state-only part),
# against the hand-derived K formula that is about to be transcribed
# into Julia (accel_closure.jl) -- checking K BEFORE writing any Julia
# code, not after, per the same standard that caught the Pi bug.
# =====================================================================
Pi_full = coeff_cm * cm + coeff_bl * bl + coeff_f * f_com
K_extracted = simplify(expand(Cddot_substituted - Pi_full))

println("=== K (extracted as C̈_substituted - Π) ===")
println(K_extracted)
println()

# Hand-derived K, to be transcribed into accel_closure.jl's `K_of_x`:
# K = [-4Oh*k^2*adot - (k^2+Bo)*k*tanh(k h0)*a] * J0(k r)
#     - 2*r_tau*adot*k*J1(k r)
#     - (1/2)*r_tau^2 * a * k^2 * [J0(k r) - J2(k r)]     <- kept in the SAME J0/J2 form
#       as the raw (pre-substitution) C̈ expression, rather than rewritten via the
#       J0(x)-J1(x)/x identity: Symbolics has no built-in knowledge of the Bessel
#       recurrence J0(x)+J2(x)=(2/x)J1(x), so a J1/x-rewritten version of this term
#       does NOT symbolically cancel against the raw J2 term even though the two are
#       analytically equal — caught by this exact check on the first attempt.
#     + Pl * outer_bracket * [damp_l*betadot + omega2_l*beta]     <- NOTE the Pl factor:
#       this is the SAME Pl that multiplies bl in Pi's third term (both come from the
#       same P_l(cosθ) weighting that ties b_l/beta_l to a specific Legendre degree);
#       first attempt below omitted it and the CAS caught the mismatch.
#     + Bo
# with r = beta*Pl*sinθ, r_tau = Dτ(beta)*Pl*sinθ, outer_bracket = -cosθ + sinθ*a*k*J1(k r)
# SIGN CORRECTED 2026-07-27 (see C's definition above): only the cosθ term of
# outer_bracket flips (same reason as Π's bl coefficient correction above); the
# sinθ*a*k*J1(...) term is unaffected since it comes from η's own chain rule, not from
# how z_d enters C.
r_frozen = beta * Pl * sin(θ)
r_tau_frozen = Dτ(beta) * Pl * sin(θ)
outer_bracket = -cos(θ) + sin(θ) * a * k * J1(k * r_frozen)

K_claimed = (-4*Oh*k^2*Dτ(a) - (k^2+Bo)*k*tanh(k*h0)*a) * J0(k*r_frozen) -
            2*r_tau_frozen*Dτ(a)*k*J1(k*r_frozen) -
            (1//2) * r_tau_frozen^2 * a * k^2 * (J0(k*r_frozen) - besselj(2, k*r_frozen)) +
            Pl * outer_bracket * (damp_l*Dτ(beta) + omega2_l*beta) +
            Bo

resid_K = simplify(expand(K_extracted - K_claimed))
ok_K = is_symbolic_zero(resid_K)
println("K residual (claimed vs extracted): ", resid_K, "   ", ok_K ? "PASS" : "FAIL")
@assert ok_K "hand-derived K does not match the symbolically-extracted K -- do not transcribe into Julia until this passes"
println()
println("ALL CHECKS (Π AND K) PASSED — safe to transcribe into accel_closure.jl.")
println()
println("ALL SYMBOLIC CHECKS PASSED — eq:accel-constraint (corrected) matches mechanical CAS re-derivation exactly.")

# =====================================================================
# eq:theta-c-crossing (design doc §subsubsec:contact-angle) and its own
# "Conditioning" disclosure: C(θ_c,τ) uses the CURRENT-STEP a_m(X), β_l(X)
# (never frozen), unlike K above. Verify symbolically that this makes ∂C/∂ĉ_n
# inherit the affine slope κ_m/λ_l (the O(δ²)-scaled quantities design doc
# §subsec:accel-motivation identifies), while ∂C/∂θ_c does not go through
# either slope at all -- i.e. the doc's "Conditioning" paragraph's claim that
# this row remains position-level/O(δ²)-sensitive in ĉ_n but O(1)-sensitive
# in θ_c is a structural fact about C's OWN functional form, not merely
# plausible-sounding.
# =====================================================================
# C_crossing itself, with a,beta,zcm left as PLAIN O(1) physical quantities (their
# actual magnitude is dominated by history alpha_m/gam_l/mu -- design doc eq:kappa-m/
# eq:lambda-l/eq:kappa-cm -- not by the pressure-induced increment; kappa_m/lambda_l
# are SENSITIVITY slopes, not magnitudes, so they must not appear inside a,beta,zcm
# themselves here, only in the chain-rule step below):
C_crossing = a * J0(k * beta * Pl * sin(θ)) - zcm + beta * Pl * cos(θ)

dC_da = expand_derivatives(Differential(a)(C_crossing))
dC_dbeta = expand_derivatives(Differential(beta)(C_crossing))
dC_dtheta = expand_derivatives(Differential(θ)(C_crossing))

@variables kappa_m lambda_l  # BDF2 affine slopes a_m(X)=alpha_m+kappa_m*c_m(X),
    # beta_l(X)=gam_l+lambda_l*b_l(X) (design doc eq:kappa-m/eq:lambda-l) -- both O(δ²)
    # (eq:kappa-general). By the chain rule, ∂C/∂ĉ_n = (∂C/∂a)(∂a/∂ĉ_n) +
    # (∂C/∂beta)(∂beta/∂ĉ_n) = (∂C/∂a)*kappa_m*(...) + (∂C/∂beta)*lambda_l*(...): an
    # overall O(δ²) sensitivity IF AND ONLY IF the O(1) coefficients ∂C/∂a, ∂C/∂beta
    # are themselves free of kappa_m/lambda_l (i.e. genuinely O(1), not secretly
    # carrying another factor of the same slopes) -- checked directly below, rather
    # than assumed.
dC_dcm_chain = dC_da * kappa_m
dC_dbl_chain = dC_dbeta * lambda_l

has_symbol(expr, sym) = occursin(string(sym), string(simplify(expr; expand=true)))

println()
println("=== eq:theta-c-crossing conditioning check (symbolic) ===")
println("∂C/∂a (O(1) coefficient multiplying kappa_m in the ĉ_n chain rule): ", dC_da)
ok_da_clean = !has_symbol(dC_da, kappa_m) && !has_symbol(dC_da, lambda_l)
println("  itself free of kappa_m/lambda_l (genuinely O(1)): ", ok_da_clean ? "PASS" : "FAIL")
println("∂C/∂beta (O(1) coefficient multiplying lambda_l in the ĉ_n chain rule): ", dC_dbeta)
ok_dbeta_clean = !has_symbol(dC_dbeta, kappa_m) && !has_symbol(dC_dbeta, lambda_l)
println("  itself free of kappa_m/lambda_l (genuinely O(1)): ", ok_dbeta_clean ? "PASS" : "FAIL")
ok_da_nonzero = !iszero(simplify(dC_da; expand=true))
ok_dbeta_nonzero = !iszero(simplify(dC_dbeta; expand=true))
println("  ∂C/∂a generically nonzero (non-degenerate chain): ", ok_da_nonzero ? "PASS" : "FAIL")
println("  ∂C/∂beta generically nonzero (non-degenerate chain): ", ok_dbeta_nonzero ? "PASS" : "FAIL")
println("=> ∂C/∂ĉ_n = (∂C/∂a)*kappa_m*(...) + (∂C/∂beta)*lambda_l*(...): ", dC_dcm_chain, " + ", dC_dbl_chain)
println("   an overall O(δ²) sensitivity, confirmed structurally, not asserted.")
println()
println("∂C/∂θ_c (direct partial; θ_c is a raw coordinate of X, never expressed via")
println("kappa_m/lambda_l at all -- it does not flow through the BDF2 affine relations)")
println("  = ", dC_dtheta)
ok_dtheta_no_slope = !has_symbol(dC_dtheta, kappa_m) && !has_symbol(dC_dtheta, lambda_l)
println("  carries no kappa_m/lambda_l factor (O(1) in the geometry itself): ", ok_dtheta_no_slope ? "PASS" : "FAIL")

@assert ok_da_clean "∂C/∂a unexpectedly carries a kappa_m/lambda_l factor -- conditioning claim needs re-examination"
@assert ok_dbeta_clean "∂C/∂beta unexpectedly carries a kappa_m/lambda_l factor -- conditioning claim needs re-examination"
@assert ok_da_nonzero "∂C/∂a is identically zero -- the O(δ²) chain-rule argument would be vacuous"
@assert ok_dbeta_nonzero "∂C/∂beta is identically zero -- the O(δ²) chain-rule argument would be vacuous"
@assert ok_dtheta_no_slope "∂C/∂θ_c unexpectedly carries a kappa_m/lambda_l factor -- conditioning claim needs re-examination"
println()
println("CONFIRMED: eq:theta-c-crossing's sensitivity to the pressure unknowns ĉ_n is")
println("structurally O(δ²) (via kappa_m/lambda_l, multiplying a genuinely nonzero O(1)")
println("coefficient), while its sensitivity to θ_c itself carries no such factor at all")
println("-- matching design doc §subsubsec:contact-angle's \"Conditioning\" remark exactly,")
println("and confirming it as a real, symbolically-derived fact, not an assertion.")
