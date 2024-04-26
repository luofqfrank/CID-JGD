using Statistics
using LinearAlgebra
using JuMP 
using ForwardDiff # to use auto gradient

function autoGrad(f_cpt::Function, n, code, x_val) 
    df_dx = ForwardDiff.gradient(x -> f_cpt(n, code, x), x_val)
    return df_dx 
end

# code2point: a dict 
# mapping a code to its representative point
# usage: code2point[code] --> (x, f(x), grad_f(x))

# code_set: the set of all discovered codes
# code_cur: the code of active component at x 
function get_act_codes(f_cpt::Function, code_set, gap, code_cur, x)
    act_code_set = Set()
    n = length(x)
    val = f_cpt(n, code_cur, x)
    for code in code_set
        if f_cpt(n, code, x) <= val + 1e-6 && val - f_cpt(n, code, x) <= gap
            push!(act_code_set, code)
        end
    end
    return act_code_set
end

# x0 the reference point 
# code represents the active component function for x and val = f(x)

# val_x is the function value at x
# r is the radius of the region

# get the maximum distance between x and representative points from code_set
function max_dist(code2point, code_set, x, access_mode)
    dist = 0
    for c in code_set 
        @assert haskey(code2point, c)
        if access_mode == 0
            res = code2point[c][1]
        else 
            res = code2point[c]
        end
        dist = max(dist, norm(x-res)) 
    end
    return dist 
end

# access_mode = 0: extraploation, 1: proactive
# if access_mode = 0, code2point maps code --> (pos, val, grad)
# if access_mode = 1, code2point maps code --> pos
function load_components(f::Function, f_cpt::Function, code2point, x, r_scope, max_cpts, access_mode)
    n = length(x)
    cpts = Dict()
    code_x, val_x = f(n, x) 
    for c in keys(code2point)
        gap = nothing
        if access_mode == 0
            z, val, grad = code2point[c]
            gap = val_x - val - dot(grad, x - z)
        else 
            z = code2point[c]
            gap =  val_x - f_cpt(n, c, x)
        end
        if length(cpts) < max_cpts && norm(x - z) <= r_scope && gap >= 0
            if access_mode == 1
                grad = autoGrad(f_cpt, n, c, x) 
            end
            cpts[c] = (gap, grad) 
        end
    end
    return cpts 
end


# cand_cpts: mapping code --> (gap, grad)
# the gap > 0 is between the component value and the current active value
# return [list of grad], total_gap

# grad_norm: norm of joint gradient
# d is a normalized vector, i.e., norm(d) = 1
function cross_border_bin_search(f::Function, code_set, x, d, alpha_lb, alpha_ub)
    @assert abs(norm(d) - 1) < 1e-8
    n = length(d)
    alpha_0 = alpha_lb 
    while abs(alpha_ub - alpha_lb) > 0.01 * alpha_0
        alpha = 0.5 * (alpha_lb + alpha_ub)
        code, val = f(n, x - alpha * d)
        if in(code, code_set) 
            alpha_lb = alpha 
        else 
            alpha_ub = alpha 
        end
    end
    c1, val1 = f(n, x - alpha_lb * d)
    c2, val2 = f(n, x - alpha_ub * d)
    return c1, val1, alpha_lb, c2, val2, alpha_ub  
end
