# encoding rule: n-1 tuple
# (sgn1, sgn2, ..., sgn_(n-1)): sgn_i is 1 or 2

function Chained_LQ(n, x)
    vec = zeros(Float64, n-1, 2)
    code = zeros(Int, n-1)
    res = 0
    for i in 1:(n-1)
        vec[i, 1] = - x[i] - x[i+1]
        vec[i, 2] = - x[i] - x[i+1] + x[i]^2 + x[i+1]^2 - 1 
        if vec[i, 1] >= vec[i, 2]
            code[i] = 1
            res += vec[i, 1]
        else 
            code[i] = 2
            res += vec[i, 2]
        end
    end
    #code = tuple(code)
    return code, res
end

# the component function of Chained_LQ
function Chained_LQ_cpt(n, code, x)
    res = 0
    for i in 1:(n-1)
        if code[i] == 1
            res += - x[i] - x[i+1]
        else 
            res += - x[i] - x[i+1] + x[i]^2 + x[i+1]^2 - 1
        end
    end
    return res
end

function Chained_LQ_init(n)
    x_init = zeros(Float64, n)
    for i in 1:n 
        x_init[i] = - 0.5 
    end
    return x_init
end

