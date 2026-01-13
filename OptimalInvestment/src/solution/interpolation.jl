"""
Interpolation utilities for value and policy functions.

Provides efficient interpolation methods used in solution algorithms.
"""

using Interpolations

"""
    linear_interp_1d(x_grid::Vector{Float64}, y_vals::Vector{Float64}, x::Float64) -> Float64

Simple linear interpolation for 1D functions.

# Arguments
- `x_grid`: Grid points (must be sorted)
- `y_vals`: Function values at grid points
- `x`: Point to interpolate at

# Returns
- Interpolated value at x
"""
function linear_interp_1d(x_grid::Vector{Float64}, y_vals::Vector{Float64}, x::Float64)
    n = length(x_grid)
    @assert length(y_vals) == n "Grid and values must have same length"
    @assert issorted(x_grid) "Grid must be sorted"

    # Handle boundary cases
    if x <= x_grid[1]
        return y_vals[1]
    elseif x >= x_grid[end]
        return y_vals[end]
    end

    # Binary search for bracket
    i_low = searchsortedlast(x_grid, x)
    i_high = i_low + 1

    # Linear interpolation
    x_low = x_grid[i_low]
    x_high = x_grid[i_high]
    y_low = y_vals[i_low]
    y_high = y_vals[i_high]

    weight = (x - x_low) / (x_high - x_low)
    return (1 - weight) * y_low + weight * y_high
end

"""
    create_interpolant_1d(x_grid::Vector{Float64}, y_vals::Vector{Float64};
                          method::Symbol=:linear)

Create 1D interpolation object using Interpolations.jl.

# Arguments
- `x_grid`: Grid points
- `y_vals`: Function values
- `method`: :linear or :cubic

# Returns
- Interpolation object that can be called as interp(x)
"""
function create_interpolant_1d(x_grid::Vector{Float64}, y_vals::Vector{Float64};
                              method::Symbol=:linear)
    @assert method in [:linear, :cubic] "Method must be :linear or :cubic"

    if method == :linear
        return LinearInterpolation(x_grid, y_vals, extrapolation_bc=Line())
    else  # cubic
        return CubicSplineInterpolation(x_grid, y_vals, extrapolation_bc=Line())
    end
end

"""
    create_interpolant_3d(x_grid, y_grid, z_grid, vals; method=:linear)

Create 3D interpolation object for V(K, D, σ).

# Arguments
- `x_grid`: Capital grid
- `y_grid`: Demand indices (1:n_D)
- `z_grid`: Volatility indices (1:n_σ)
- `vals`: 3D array of values
- `method`: :linear only (cubic not supported for 3D)

# Returns
- Interpolation object
"""
function create_interpolant_3d(x_grid, y_grid, z_grid, vals; method=:linear)
    if method == :linear
        return LinearInterpolation((x_grid, y_grid, z_grid), vals,
                                   extrapolation_bc=Line())
    else
        @warn "Only linear interpolation supported for 3D. Using linear."
        return LinearInterpolation((x_grid, y_grid, z_grid), vals,
                                   extrapolation_bc=Line())
    end
end

"""
    find_bracket(grid::Vector{Float64}, x::Float64) -> (Int, Int, Float64)

Find interpolation bracket for x in grid.

# Returns
- `i_low`: Lower index
- `i_high`: Upper index
- `weight`: Weight on upper point (0 to 1)
"""
function find_bracket(grid::Vector{Float64}, x::Float64)
    n = length(grid)

    # Boundary cases
    if x <= grid[1]
        return 1, 1, 0.0
    elseif x >= grid[end]
        return n, n, 1.0
    end

    # Binary search
    i_low = searchsortedlast(grid, x)
    i_high = i_low + 1

    # Compute weight
    x_low = grid[i_low]
    x_high = grid[i_high]
    weight = (x - x_low) / (x_high - x_low)

    return i_low, i_high, weight
end

"""
    bilinear_interp(x_grid, y_grid, vals, x, y) -> Float64

Bilinear interpolation on 2D grid.

# Arguments
- `x_grid`: Grid points in x dimension
- `y_grid`: Grid points in y dimension
- `vals`: 2D array of values [i_x, i_y]
- `x, y`: Points to interpolate at

# Returns
- Interpolated value
"""
function bilinear_interp(x_grid, y_grid, vals, x, y)
    # Find brackets
    ix_low, ix_high, wx = find_bracket(x_grid, x)
    iy_low, iy_high, wy = find_bracket(y_grid, y)

    # Handle boundary cases
    if ix_low == ix_high && iy_low == iy_high
        return vals[ix_low, iy_low]
    elseif ix_low == ix_high
        v_low = vals[ix_low, iy_low]
        v_high = vals[ix_low, iy_high]
        return (1 - wy) * v_low + wy * v_high
    elseif iy_low == iy_high
        v_low = vals[ix_low, iy_low]
        v_high = vals[ix_high, iy_low]
        return (1 - wx) * v_low + wx * v_high
    end

    # Bilinear interpolation
    v11 = vals[ix_low, iy_low]
    v12 = vals[ix_low, iy_high]
    v21 = vals[ix_high, iy_low]
    v22 = vals[ix_high, iy_high]

    v1 = (1 - wx) * v11 + wx * v21
    v2 = (1 - wx) * v12 + wx * v22

    return (1 - wy) * v1 + wy * v2
end

"""
    interpolate_on_K(grids::StateGrids, vals::Array{Float64,3},
                     K::Float64, i_D::Int, i_σ::Int) -> Float64

Interpolate 3D array (e.g., value function) at continuous K, discrete (D, σ).

This is a convenience wrapper around the StateGrids interpolation functions.
"""
function interpolate_on_K(grids::StateGrids, vals::Array{Float64,3},
                         K::Float64, i_D::Int, i_σ::Int)
    i_low, i_high, weight = find_K_bracket(grids, K)

    if i_low == i_high
        return vals[i_low, i_D, i_σ]
    else
        v_low = vals[i_low, i_D, i_σ]
        v_high = vals[i_high, i_D, i_σ]
        return (1 - weight) * v_low + weight * v_high
    end
end

"""
    derivative_fd(f, x; h=1e-5) -> Float64

Compute derivative of f at x using finite differences (forward difference).
"""
function derivative_fd(f, x; h=1e-5)
    return (f(x + h) - f(x)) / h
end

"""
    derivative_cd(f, x; h=1e-5) -> Float64

Compute derivative of f at x using central differences.
"""
function derivative_cd(f, x; h=1e-5)
    return (f(x + h) - f(x - h)) / (2 * h)
end

"""
    gradient_fd(f, x::Vector{Float64}; h=1e-5) -> Vector{Float64}

Compute gradient of f at x using finite differences.
"""
function gradient_fd(f, x::Vector{Float64}; h=1e-5)
    n = length(x)
    grad = zeros(n)

    f_x = f(x)

    for i in 1:n
        x_plus = copy(x)
        x_plus[i] += h
        grad[i] = (f(x_plus) - f_x) / h
    end

    return grad
end
