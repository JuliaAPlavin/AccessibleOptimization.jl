module AccessibleOptimization

using Reexport
@reexport using Optimization
@reexport using AccessorsExtra
using DataPipes
using ConstructionBase
using Statistics: mean

export OptArgs, OptCons, OptProblemSpec, solobj


ConstructionBase.constructorof(::Type{<:OptimizationFunction{x}}) where {x} = 
    function(args...)
        kwargs = NamedTuple{fieldnames(OptimizationFunction)}(args)
        OptimizationFunction{x}(kwargs.f, kwargs.adtype; @delete(kwargs[(:f, :adtype)])...)
    end


struct OptArgs{TS}
    specs::TS
    OptArgs(specs...) = new{typeof(specs)}(specs)
end

_optic((o, i)::Pair) = o
_optic(o) = o
_intbound((o, i)::Pair) = i
_intbound(o) = nothing
optic(v::OptArgs) = AccessorsExtra.ConcatOptics(map(_optic, v.specs))

rawu(x0, v::OptArgs) = getall(x0, optic(v))
fromrawu(u, x0, v::OptArgs) = setall(x0, optic(v), u)
rawfunc(f, x0, v::OptArgs) = (u, p) -> f(fromrawu(u, x0, v), p)
rawbounds(x0, v::OptArgs, AT=nothing) =
    if @p v.specs |> any(isnothing(_intbound(_)))
        @assert @p v.specs |> all(isnothing(_intbound(_)))
        return ()
    else
        @p let
            v.specs
            map(fill(_intbound(_), length(getall(x0, _optic(_)))))
            reduce(vcat)
            (lb=_convert(AT, minimum.(__)), ub=_convert(AT, maximum.(__)))
        end
    end

rawu(x0::Type, v::OptArgs) = @p v.specs |> map(_intbound) |> map(mean)
fromrawu(u, x0::Type, v::OptArgs) = @p map(_optic(_1) => _2, v.specs, u) |> construct(x0, __...)
rawbounds(x0::Type, v::OptArgs, AT=nothing) =
    if @p v.specs |> any(isnothing(_intbound(_)))
        @assert @p v.specs |> all(isnothing(_intbound(_)))
        return ()
    else
        @p let
            v.specs
            map(_intbound(_))
            (lb=_convert(AT, minimum.(__)), ub=_convert(AT, maximum.(__)))
        end
    end


struct OptCons{TC,TS}
    ctype::TC
    specs::TS
    
    OptCons(CT::Union{Type,Nothing}, specs...) = new{typeof(CT), typeof(specs)}(CT, specs)
    OptCons(specs...) = OptCons(nothing, specs...)
end
ConstructionBase.constructorof(::Type{<:OptCons}) = (ctype, specs) -> OptCons(ctype, specs...)

rawconsbounds(c::OptCons) = @p let
    c.specs
    map(_[2])
    (lcons=_convert(c.ctype, minimum.(__)), ucons=_convert(c.ctype, maximum.(__)))
end

# rawcons(cons::OptCons, x0, v::OptArgs) = function(u, p)
# 	x = fromrawu(u, x0, v)
# 	map(cons.specs) do (consfunc, consint)
# 		consfunc(x, p)
# 	end |> cons.ctype
# end

_apply(f, args...) = f(args...)

rawcons(cons::OptCons, x0, v::OptArgs) = function(res, u, p)
    x = fromrawu(u, x0, v)
    res .= _apply.(first.(cons.specs), (x,), (p,))
end

Base.summary(cons::OptCons, x, p) = @p let
    cons.specs
    map(enumerate(__)) do (i, (f, int))
        v = f(x, p)
        "cons #$i: $v $(v ∈ int ? '∈' : '∉') $int"
    end
    join(__, '\n')
    Text
end


struct OptProblemSpec{F,D,U,X0,VS<:OptArgs,CS<:Union{OptCons,Nothing}}
    func::F
    data::D
    utype::U
    x0::X0
    vars::VS
    cons::CS
end

function OptProblemSpec(f::Base.Fix2, utype::Type, x0, vars::OptArgs, cons::OptCons)
    cons = @set cons.ctype = something(cons.ctype, utype)
    OptProblemSpec(f.f, f.x, utype, x0, vars, cons)
end
OptProblemSpec(f::Base.Fix2, utype::Union{Type,Nothing}, x0, vars::OptArgs, cons::Union{OptCons,Nothing}) = OptProblemSpec(f.f, f.x, utype, x0, vars, cons)
OptProblemSpec(f::Base.Fix2, utype::Union{Type,Nothing}, x0, vars::OptArgs) = OptProblemSpec(f, utype, x0, vars, nothing)
OptProblemSpec(f::Base.Fix2, x0, vars::OptArgs) = OptProblemSpec(f, nothing, x0, vars)
OptProblemSpec(f::Base.Fix2, x0, vars::OptArgs, cons::Union{OptCons,Nothing}) = OptProblemSpec(f, nothing, x0, vars, cons)

rawfunc(s::OptProblemSpec) = rawfunc(s.func, s.x0, s.vars)
rawfunc(s::OptProblemSpec{<:OptimizationFunction}) = rawfunc(s.func.f, s.x0, s.vars)
rawdata(s::OptProblemSpec) = s.data

rawu(s::OptProblemSpec) = _convert(s.utype, rawu(s.x0, s.vars))
rawbounds(s::OptProblemSpec) = rawbounds(s.x0, s.vars, s.utype)
rawconsbounds(s::OptProblemSpec) = rawconsbounds(s.cons)
rawcons(s::OptProblemSpec) = rawcons(s.cons, s.x0, s.vars)
fromrawu(u, s::OptProblemSpec) = fromrawu(u, s.x0, s.vars)
solobj(sol, s) = fromrawu(sol.u, s)


struct OptSolution{S<:SciMLBase.OptimizationSolution, O<:OptProblemSpec}
    sol::S
    ops::O
end
Base.propertynames(os::OptSolution) = (propertynames(os.sol)..., :uobj)
Base.getproperty(os::OptSolution, s::Symbol) =
    s == :uobj ? solobj(os.sol, os.ops) :
    s == :sol ? getfield(os, s) :
    s == :ops ? getfield(os, s) :
    getproperty(os.sol, s)


SciMLBase.solve(s::OptProblemSpec, args...; kwargs...) =
    OptSolution(solve(OptimizationProblem(s), args...; kwargs...), s)

SciMLBase.OptimizationProblem(s::OptProblemSpec, args...; kwargs...) =
    if isnothing(s.cons)
        OptimizationProblem(rawfunc(s), rawu(s), rawdata(s), args...; rawbounds(s)..., kwargs...)
    else
        OptimizationProblem(OptimizationFunction(rawfunc(s), cons=rawcons(s)), rawu(s), rawdata(s), args...; rawbounds(s)..., rawconsbounds(s)..., kwargs...)
    end

SciMLBase.OptimizationProblem(s::OptProblemSpec{<:OptimizationFunction}, args...; kwargs...) =
    if isnothing(s.cons)
        f = @p s.func |>
            @set(__.f = rawfunc(s))
        OptimizationProblem(f, rawu(s), rawdata(s), args...; rawbounds(s)..., kwargs...)
    else
        f = @p s.func |>
            @set(__.f = rawfunc(s)) |>
            @set __.cons = rawcons(s)
        OptimizationProblem(f, rawu(s), rawdata(s), args...; rawbounds(s)..., rawconsbounds(s)..., kwargs...)
    end


_convert(::Nothing, x) = x

_convert(T::Type{<:Tuple}, x::Tuple) = convert(T, x)
_convert(T::Type{<:Vector}, x::Tuple) = convert(T, collect(x))
_convert(T::Type{<:AbstractVector}, x::Tuple) = convert(T, x)  # SVector, MVector

_convert(T::Type{<:Tuple}, x::AbstractVector) = T(x...)
_convert(T::Type{<:Vector}, x::AbstractVector) = convert(T, x)
_convert(T::Type{<:AbstractVector}, x::AbstractVector) = T(x...)  # SVector, MVector

end
