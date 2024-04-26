using ForwardDiff

# encoding rule 2-tuple: 
# (i, 1): the i th branch is active, and the value in the abs() is positive
# (i, -1): the i th branch is active, and the value in the abs() is negative  

function gen_MXHILB(n, x)
    vec = zeros(Float64, n)
    vec_abs = zeros(Float64, n)
    for i in 1:n 
        vec[i] = sum(x[j]/(i+j-1) for j in 1:n)
        vec_abs[i] = abs(vec[i])
    end 
    val = maximum(vec_abs)
    idx = argmax(vec_abs)
    if vec[idx] >= 0
        sgn = 1
    else 
        sgn = -1
    end
    code = (idx, sgn)
    return code, val 
end

function gen_MXHILB_cpt(n, code, x)
    idx, sgn = code
    res = sum(x[j] / (idx + j - 1) for j in 1:n)
    if sgn < 0
        res = - res 
    end 
    return res 
end

function gen_MXHILB_init(n) 
    x_init = zeros(Float64, n)
    for i = 1:n 
        x_init[i] = 1 
    end
    return x_init
end