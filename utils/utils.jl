using Statistics
using LinearAlgebra
using JuMP 
using ForwardDiff # to use auto gradient
using OSQP

function autoGrad(f_cpt::Function, n, code, x_val) 
    df_dx = ForwardDiff.gradient(x -> f_cpt(n, code, x), x_val)
    return df_dx 
end
