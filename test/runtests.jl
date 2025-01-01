using TestItems
using TestItemRunner
@run_package_tests


@testitem "constructorof" begin
    of = OptimizationFunction(+, Optimization.AutoForwardDiff())
    @test constructorof(typeof(of))(Accessors.getfields(of)...) === of
end

@testitem "Optimization" begin
    using IntervalSets
    using StructArrays
    using StaticArrays
    using OptimizationOptimJL, OptimizationMetaheuristics
    using ReverseDiff
    
    struct ExpModel{A,B}
        scale::A
        shift::B
    end
    
    struct SumModel{T <: Tuple}
        comps::T
    end
    
    (m::ExpModel)(x) = m.scale * exp(-(x - m.shift)^2)
    (m::SumModel)(x) = sum(c -> c(x), m.comps)

    loss(m, data) = sum(r -> abs2(r.y - m(r.x)), data)

    truemod = SumModel((
        ExpModel(2, 5),
        ExpModel(0.5, 2),
        ExpModel(0.5, 8),
    ))

    data = let x = 0:0.2:10
        StructArray(; x, y=truemod.(x) .+ range(-0.01, 0.01, length=length(x)))
    end

    mod0 = SumModel((
        ExpModel(1, 1),
        ExpModel(1, 2),
        ExpModel(1, 3),
    ))
    vars = OptArgs(
        @optic(_.comps[∗].shift) => 0..10.,
        @optic(_.comps[∗].scale) => 0.3..10.,
    )
    prob = OptProblemSpec(Base.Fix2(loss, data), mod0, vars)
    sol = solve(prob, ECA(), maxiters=300)
    @test sol.u isa Vector{Float64}
    @test sol.uobj isa SumModel
    @test getall(sol.uobj, @optic _.comps[∗].shift) |> collect |> sort ≈ [2, 5, 8]  rtol=1e-2

    @testset "no autodiff" begin
    @testset "OptProblemSpec(utype=$(prob.utype))" for prob in (
        OptProblemSpec(Base.Fix2(loss, data), mod0, vars),
        OptProblemSpec(Base.Fix2(loss, data), Vector, mod0, vars),
        OptProblemSpec(Base.Fix2(loss, data), Vector{Float64}, mod0, vars),
        OptProblemSpec(Base.Fix2(loss, data), SVector, mod0, vars),
        OptProblemSpec(Base.Fix2(loss, data), SVector{<:Any, Float64}, mod0, vars),
        OptProblemSpec(Base.Fix2(loss, data), MVector, mod0, vars),
        OptProblemSpec(Base.Fix2(loss, data), MVector{<:Any, Float64}, mod0, vars),
        OptProblemSpec(Base.Fix2(OptimizationFunction{false}(loss), data), mod0, vars),
        OptProblemSpec(Base.Fix2(OptimizationFunction{false}(loss), data), Vector, mod0, vars),
        OptProblemSpec(Base.Fix2(OptimizationFunction{false}(loss), data), Vector{Float64}, mod0, vars),
        OptProblemSpec(Base.Fix2(OptimizationFunction{false}(loss), data), SVector, mod0, vars),
        OptProblemSpec(Base.Fix2(OptimizationFunction{false}(loss), data), SVector{<:Any, Float64}, mod0, vars),
        OptProblemSpec(Base.Fix2(OptimizationFunction{false}(loss), data), MVector, mod0, vars),
        OptProblemSpec(Base.Fix2(OptimizationFunction{false}(loss), data), MVector{<:Any, Float64}, mod0, vars),
    )
        sol = solve(prob, ECA(), maxiters=10)
        @test sol.u isa Vector{Float64}
        @test sol.uobj isa SumModel
    end
    end

    @testset "construct" begin
        vars = OptArgs(
            @optic(_.scale) => 0.3..10.,
            @optic(_.shift) => 0..10.,
        )
        @testset "OptProblemSpec(utype=$(prob.utype))" for prob in (
            OptProblemSpec(Base.Fix2(loss, data), Vector, ExpModel, vars),
            OptProblemSpec(Base.Fix2(loss, data), Vector{Float64}, ExpModel, vars),
            OptProblemSpec(Base.Fix2(loss, data), SVector, ExpModel, vars),
            OptProblemSpec(Base.Fix2(loss, data), SVector{<:Any, Float64}, ExpModel, vars),
            OptProblemSpec(Base.Fix2(loss, data), MVector, ExpModel, vars),
            OptProblemSpec(Base.Fix2(loss, data), MVector{<:Any, Float64}, ExpModel, vars),
            OptProblemSpec(Base.Fix2(OptimizationFunction{false}(loss), data), Vector, ExpModel, vars),
            OptProblemSpec(Base.Fix2(OptimizationFunction{false}(loss), data), Vector{Float64}, ExpModel, vars),
            OptProblemSpec(Base.Fix2(OptimizationFunction{false}(loss), data), SVector, ExpModel, vars),
            OptProblemSpec(Base.Fix2(OptimizationFunction{false}(loss), data), SVector{<:Any, Float64}, ExpModel, vars),
            OptProblemSpec(Base.Fix2(OptimizationFunction{false}(loss), data), MVector, ExpModel, vars),
            OptProblemSpec(Base.Fix2(OptimizationFunction{false}(loss), data), MVector{<:Any, Float64}, ExpModel, vars),
        )
            sol = solve(prob, ECA(), maxiters=10)
            @test sol.u isa Vector{Float64}
            @test sol.uobj isa ExpModel
        end
    end

    @testset "autodiff, cons" begin
        vars = OptArgs(
            @optic(_.comps[∗].shift) => 0..10.,
            @optic(_.comps[∗].scale) => 0.3..10.,
        )
        cons = OptCons(
            ((x, _) -> sum(c -> c.shift, x.comps) / length(x.comps)) => 0.5..4,
        )
        prob = OptProblemSpec(Base.Fix2(OptimizationFunction(loss, Optimization.AutoForwardDiff()), data), Vector{Float64}, mod0, vars, cons)
        sol = solve(prob, Optim.IPNewton(), maxiters=10)
        @test sol.u isa Vector{Float64}
        @test sol.uobj isa SumModel
        for (f, int) in cons.specs
            @test f(sol.uobj, data) ∈ int
        end
    end

    @testset "autodiff, no box" begin
        vars = OptArgs(
            @optic(_.comps[∗].shift),
            @optic(_.comps[∗].scale),
        )
        prob = OptProblemSpec(Base.Fix2(OptimizationFunction(loss, Optimization.AutoForwardDiff()), data), Vector{Float64}, mod0, vars)
        sol = solve(prob, Optim.Newton(), maxiters=10)
        @test sol.u isa Vector{Float64}
        @test sol.uobj isa SumModel
    end
end


@testitem "_" begin
    import CompatHelperLocal as CHL
    CHL.@check()

    using Aqua
    Aqua.test_all(AccessorsExtra, piracies=false, ambiguities=false)
end
