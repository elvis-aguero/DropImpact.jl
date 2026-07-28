# Parallel parameter sweep with an auto-selected worker count.
#
# Usage:  julia --project=. -t auto scripts/sweep.jl [options] [We ...]
#
#   --wall=free|pinned|clamped   bath wall condition (default free; design doc §subsubsec:wall)
#   --Bo= --Oh= --M= --L= --N= --nq= --t_end=      physics and truncation
#   --out=PATH           CSV to write/append (default data/sweep_<wall>.csv)
#   --workers=N          skip auto-selection and use N workers
#   --no-ablation        use the memory cap only, do not time-ablate
#
# Two properties this driver is built around, both measured rather than assumed:
#
#   * A single impact's MARGINAL memory footprint is small -- about 12 MiB of peak RSS on
#     top of a ~450 MiB fixed cost for the Julia runtime plus this package -- because the
#     per-case metric reduction below discards `levels`/`diag` before returning. Retaining
#     full histories for every case is what would make a long sweep blow up, so it is
#     precisely what `case_metrics` refuses to do. Consequence: worker count is normally
#     CPU-bound, not memory-bound, and the memory cap only bites at large M, L or nq.
#
#   * Wall-clock per case does NOT fall off linearly with added workers. These runs are
#     allocation-heavy, so they contend on memory bandwidth and on the GC, and past some
#     width more workers buy nothing or lose. The ablation below finds that knee instead of
#     trusting `Sys.CPU_THREADS`.
using SpectralKM, Printf, DelimitedFiles

# ---------------------------------------------------------------- one case, reduced early

"""
    case_metrics(We; kwargs...) -> NamedTuple

Run one impact and return only scalars. The `levels`/`diag` histories go out of scope
before this returns, which is what keeps a sweep's footprint flat in the number of cases
rather than linear in it. Never return `levels` from here.
"""
function case_metrics(We; Bo, Oh, M, L, N, nq, t_end, wall, b=6.0, h0=3.0)
    p = Params(; We, Bo, Oh, M, L, N, b, h0, nq, wall)
    levels, diag, phases = run_simulation(p; t_end=t_end, dt_init=1e-3)
    isempty(diag) && return (; We, contact=false, t_cont=NaN, t_cont_total=NaN, nintervals=0,
                             cor=NaN, depth=NaN, rc_max=NaN, theta_c_max=NaN,
                             nsteps=length(levels))
    times = [lv.t for lv in levels]
    ivs = contact_intervals(times, phases)
    iout = argmax([d.t for d in diag])
    theta_max = maximum(d.theta_c for d in diag)
    # t_cont is the PRIMARY interval, not the first-to-last span: the span is inflated by
    # detachment chatter and by any free-flight excursion between re-contacts. See
    # `contact_intervals`. `nintervals > 1` is the flag that chatter occurred.
    return (; We, contact=true,
            t_cont = primary_contact_time(times, phases),
            t_cont_total = contact_time(times, phases),
            nintervals = length(ivs),
            cor = abs(diag[iout].v / -sqrt(We)),
            depth = max_penetration_depth(levels, p.L),
            rc_max = sin(theta_max),
            theta_c_max = theta_max,
            nsteps = length(levels))
end

const FIELDS = (:We, :contact, :t_cont, :t_cont_total, :nintervals, :cor, :depth,
                :rc_max, :theta_c_max, :nsteps)

# ------------------------------------------------------------------ bounded-width mapping

"""
    pmap_limited(f, items, W) -> Vector

Run `f` over `items` with at most `W` running concurrently, via a semaphore over
`Threads.@spawn`. Julia fixes `nthreads()` at startup, so a script cannot change its
thread count -- but it can choose how much of that pool to *use*, which is what makes a
runtime ablation over `W` possible at all. A case that throws is captured, not propagated:
one bad Weber number must not abandon the rest of the sweep.
"""
function pmap_limited(f, items, W::Int)
    out = Vector{Any}(undef, length(items))
    sem = Base.Semaphore(max(W, 1))
    @sync for (i, item) in enumerate(items)
        Threads.@spawn begin
            Base.acquire(sem)
            try
                out[i] = f(item)
            catch err
                out[i] = err
            finally
                Base.release(sem)
            end
        end
    end
    return out
end

# ----------------------------------------------------------------------- worker selection

"""
    measure_footprint(proxy) -> (bytes, seconds)

Peak-RSS growth and wall time attributable to one case, measured after a warm-up so that
compilation is not billed to the case. `Sys.maxrss()` is a process-wide high-water mark, so
the difference across a single run is an upper bound on one worker's transient working set
-- the conservative direction for a memory budget.
"""
function measure_footprint(proxy::Function)
    proxy()                                   # warm up: compile, touch every code path
    GC.gc(); GC.gc()
    before = Sys.maxrss()
    t = @elapsed proxy()
    growth = Sys.maxrss() - before
    GC.gc()
    return max(growth, 0), t
end

"""
    memory_cap(footprint; fraction=0.5) -> Int

Largest worker count whose combined transient footprint fits in `fraction` of total RAM
after the fixed runtime cost is set aside.

Budgeted against `Sys.total_memory()`, deliberately NOT `Sys.free_memory()`: on macOS the
latter counts only genuinely free pages and reads as ~0 GiB on a warm machine (measured),
which would cap every sweep at one worker. Total memory with a conservative fraction is
crude but does not silently serialise the sweep.
"""
function memory_cap(footprint::Integer; fraction::Float64=0.5)
    footprint <= 0 && return typemax(Int)
    budget = fraction * Sys.total_memory() - Sys.maxrss()
    budget <= 0 && return 1
    return max(1, floor(Int, budget / footprint))
end

"""
    ablate(proxy, Wmax; gain=0.15) -> (W, timings)

Time the proxy case at W = 1, 2, 4, ... up to `Wmax` and return the knee: the largest W
that still improved throughput by more than `gain` over the previous width. Each trial runs
`2W` cases so that every worker gets at least two, otherwise the measurement is dominated
by the ramp-up of the last one.

Doubling rather than scanning every W keeps the ablation to a handful of trials: with
`Wmax = 8` this is four trials, and the whole calibration costs a few case-times -- paid
back on any sweep long enough to be worth parallelising.
"""
function ablate(proxy::Function, Wmax::Int; gain::Float64=0.15)
    timings = Tuple{Int,Float64}[]
    best_W, best_rate = 1, 0.0
    W = 1
    while W <= Wmax
        n = 2W
        t = @elapsed pmap_limited(_ -> proxy(), 1:n, W)
        rate = n / t                                        # cases per second
        push!(timings, (W, rate))
        if rate > best_rate * (1 + gain)
            best_W, best_rate = W, rate
        end
        W *= 2
    end
    return best_W, timings
end

# ------------------------------------------------------------------------------ CSV output

"""Weber numbers already recorded in `path`, so an interrupted sweep can be resumed."""
function completed_cases(path::String)
    isfile(path) || return Set{Float64}()
    rows = readdlm(path, ','; skipstart=1)
    size(rows, 1) == 0 && return Set{Float64}()
    return Set{Float64}(Float64.(rows[:, 1]))
end

function append_row(path::String, row::NamedTuple)
    new = !isfile(path)
    open(path, "a") do io
        new && println(io, join(FIELDS, ","))
        println(io, join((getfield(row, f) for f in FIELDS), ","))
    end
end

# ----------------------------------------------------------------------------------- main

function parse_args(argv)
    opts = Dict{String,String}()
    positional = Float64[]
    for a in argv
        if startswith(a, "--")
            body = a[3:end]
            i = findfirst('=', body)
            i === nothing ? (opts[body] = "true") : (opts[body[1:i-1]] = body[i+1:end])
        else
            push!(positional, parse(Float64, a))
        end
    end
    return opts, positional
end

getopt(opts, key, default, T=String) = haskey(opts, key) ? parse_or(T, opts[key]) : default
parse_or(::Type{String}, s) = s
parse_or(T, s) = parse(T, s)

function main(argv)
    opts, positional = parse_args(argv)

    wall = Symbol(getopt(opts, "wall", "free"))
    Bo = getopt(opts, "Bo", 0.017, Float64)
    Oh = getopt(opts, "Oh", 0.006, Float64)
    M = getopt(opts, "M", 60, Int)
    L = getopt(opts, "L", 60, Int)
    N = getopt(opts, "N", 3, Int)
    nq = getopt(opts, "nq", 40, Int)
    t_end = getopt(opts, "t_end", 14.0, Float64)
    out = getopt(opts, "out", joinpath(@__DIR__, "..", "data", "sweep_$(wall).csv"))

    We_values = isempty(positional) ?
        [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0958, 1.5, 2.0, 2.5, 3.0, 4.0, 4.5] :
        positional

    kw = (; Bo, Oh, M, L, N, nq, t_end, wall)
    @printf("sweep: %d cases, wall=%s, Bo=%.4g Oh=%.4g M=%d L=%d N=%d nq=%d t_end=%.3g\n",
            length(We_values), wall, Bo, Oh, M, L, N, nq, t_end)

    done = completed_cases(out)
    todo = filter(We -> !(We in done), We_values)
    if length(todo) < length(We_values)
        @printf("resuming: %d of %d cases already in %s\n",
                length(We_values) - length(todo), length(We_values), out)
    end
    isempty(todo) && (println("nothing to do"); return)

    # ---- worker selection -------------------------------------------------------------
    W = if haskey(opts, "workers")
        parse(Int, opts["workers"])
    else
        select_workers(kw, length(todo), !haskey(opts, "no-ablation"))
    end
    @printf("\nrunning %d cases at %d worker(s)\n\n", length(todo), W)

    # ---- the sweep --------------------------------------------------------------------
    t0 = time()
    results = pmap_limited(We -> case_metrics(We; kw...), todo, W)
    elapsed = time() - t0

    @printf("%-9s %-8s %-10s %-10s %-6s %-9s %-9s %-9s %-8s\n",
            "We", "contact", "t_cont", "t_c_total", "nivl", "CoR", "depth", "r_c max", "steps")
    for (We, r) in zip(todo, results)
        if r isa Exception
            @printf("%-9.4g FAILED   %s\n", We, sprint(showerror, r))
            continue
        end
        @printf("%-9.4g %-8s %-10.4f %-10.4f %-6d %-9.4f %-9.4f %-9.4f %-8d\n",
                r.We, r.contact, r.t_cont, r.t_cont_total, r.nintervals,
                r.cor, r.depth, r.rc_max, r.nsteps)
        append_row(out, r)
    end
    nok = count(r -> !(r isa Exception), results)
    @printf("\n%d/%d cases in %.1f s (%.1f s/case wall, %.1f s/case serial-equivalent)\n",
            nok, length(todo), elapsed, elapsed / length(todo), elapsed * W / length(todo))
    println("written: ", out)
end

"""Pick a worker count from the memory cap, the core count, and an optional timing knee."""
function select_workers(kw, ncases::Int, do_ablation::Bool)
    cores, threads = Sys.CPU_THREADS, Threads.nthreads()
    if threads == 1
        println("\nnthreads() == 1: the sweep will run serially.")
        println("Relaunch with `julia --project=. -t auto scripts/sweep.jl ...` to parallelise.")
        return 1
    end

    # A cheap but structurally identical proxy: same code path, coarser truncation and a
    # short horizon, so calibration costs a fraction of a real case.
    proxy_kw = merge(kw, (M=16, L=16, nq=16, t_end=1.0))
    proxy = () -> case_metrics(1.0958; proxy_kw...)

    print("calibrating one case ... "); flush(stdout)
    footprint, tcase = measure_footprint(proxy)
    cap = memory_cap(footprint)
    @printf("%.1f MiB peak growth, %.2f s\n", footprint / 2^20, tcase)
    @printf("  memory cap %s worker(s) at 50%% of %.1f GiB total; cores %d, threads %d\n",
            cap == typemax(Int) ? "unbounded" : string(cap),
            Sys.total_memory() / 2^30, cores, threads)

    Wmax = min(threads, cap, ncases)
    Wmax <= 1 && return 1

    if !do_ablation
        @printf("  ablation skipped; using %d worker(s)\n", Wmax)
        return Wmax
    end

    print("ablating worker count ... "); flush(stdout)
    W, timings = ablate(proxy, Wmax)
    println()
    for (w, rate) in timings
        @printf("  W=%-3d %.2f cases/s%s\n", w, rate, w == W ? "   <- selected" : "")
    end
    return W
end

main(ARGS)
