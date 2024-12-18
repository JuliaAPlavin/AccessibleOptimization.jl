# AccessibleOptimization.jl

Combining `Accessors.jl` + `Optimization.jl` to enable function optimization with arbitrary structs. Vary struct parameters, combinations and transformations of them. Uniform and composable, zero overhead.

Defining features of `AccessibleOptimization`:
- No need to deal with raw vectors of optimized parameters: their values are automatically put into model structs
- Can use arbitrary structs as model definitions, no requirements or limitations on their content, types, or methods
- Can flexibly specify the list of parameters to optimize, as well as their bounds and constrains - independently of the model struct/object creation
- Everything is type-stable and performant

# Usage

Suppose you want to fit a model to data. Here, our model is a sum of squared exponentials.

First, define your model and a way to evaluate it. Regular Julia code, no special types or functions. This is not `Accessors`/`Optimization` specific at all, you may have model definitions already!
```julia
struct ExpModel{A,B}
    scale::A
    shift::B
end

struct SumModel{T <: Tuple}
    comps::T
end

(m::ExpModel)(x) = m.scale * exp(-(x - m.shift)^2)
(m::SumModel)(x) = sum(c -> c(x), m.comps)

loss(m, data) = @p data |> sum(abs2(_.y - m(_.x)))
```

Then, load `AccessibleOptimization`, define optimization parameters, and perform optimization:
```julia
using AccessibleOptimization

# which parameters to optimize, what are their bouds?
vars = OptArgs(
	(@o _.comps[∗].shift) => 0..10.,  # shifts of both components: values from 0..10
	(@o log10(_.comps[∗].scale)) => -1..1,  # component scales: positive-only (using log10 transformation), from 10^-1 to 10^1
)
# create and solve the optimization problem, interface very similar to Optimization.jl
ops = OptProblemSpec(Base.Fix2(loss, data), mod0, vars)
sol = solve(ops, ECA(), maxiters=300)
sol.uobj::SumModel  # the optimal model
```
We use [Accessors.jl](https://github.com/JuliaObjects/Accessors.jl) and [AccessorsExtra.jl](https://github.com/JuliaAplavin/AccessorsExtra.jl) optics to specify parameters to vary during optimization. Basic `@optic` syntax demonstrated in the above example includes property and index access (`_.prop`, `_[1]`) and selecting all elements of a collection with `_[∗]` (type with `\ast<tab>`). Refer to the `Accessors[Extra]` documentation for more details.

See a Pluto notebook with [a walkthrough, more examples and explanations](https://aplavin.github.io/AccessibleOptimization.jl/examples/notebook.html).

# Related packages

`AccessibleOptimization` gains its composability and generality powers (and half its name!) from `Accessors` and `AccessorsExtra` packages.

The optimization part is directly delegated to `Optimization`. Other backends are possible, but best to add them as methods to `Optimization` proper and use from `AccessibleOptimization`.

These packages have generally similar goals, but neither provides all features `AccessibleOptimization` or `Accessors` do:
- [Functors.jl](https://github.com/FluxML/Functors.jl)
- [FlexibleFunctors.jl](https://github.com/Metalenz/FlexibleFunctors.jl)
- [GModelFit.jl](https://github.com/gcalderone/GModelFit.jl)
- [ModelParameters.jl](https://github.com/rafaqz/ModelParameters.jl)
- [ParameterHandling.jl](https://github.com/JuliaGaussianProcesses/ParameterHandling.jl)
- [TransformVariables.jl](https://github.com/tpapp/TransformVariables.jl)
- [ValueShapes.jl](https://github.com/oschulz/ValueShapes.jl)
